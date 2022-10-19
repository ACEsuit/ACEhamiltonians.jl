using ACEhamiltonians, BlockArrays, HDF5, LinearAlgebra
using JuLIP; Atoms
using ACEhamiltonians.DatabaseIO: _clean_bool


@warn "Do not use this unless you know what you are doing! (currently broken)"

function _block_sizes(atoms, basis_def)
    orbs_per_z = Dict(k=>sum(2v .+ 1) for (k, v) in basis_def)
    block_sizes = Vector{Int64}(undef, length(atoms))
    for (i, z) in enumerate(atoms.Z)
        block_sizes[i] = orbs_per_z[z]
    end
    return block_sizes
end

function real_to_block(m, atoms, basis_def)
    block_sizes = _block_sizes(atoms, basis_def)
    return BlockArray(m, block_sizes, block_sizes, ones(Int64, size(m, 3)))
end

function block_from_atoms(atoms, basis_def, images)
    m = _block_sizes(atoms, basis_def)
    n = ones(Int64, size(images, 2))
    return fill!(BlockArray{Float64}(undef, m, m, n), 0.0)
end

function get_block_indices(n::I, c::Matrix{I}) where I<:Integer
    m = size(c, 2)
    res = Matrix{I}(undef, 5, n^2 * m)
    let a_idxs = 1:n
        @views reshape(res[1, :], (n, n, m)) .= a_idxs'
        @views reshape(res[2, :], (n, n, m)) .= a_idxs
        @views reshape(res[3:5, :], (3, n^2, m)) .= reshape(c, (3, 1, m))
    end
    return res
end


function shift_map(atoms::Atoms)

    x = Matrix{Int64}(undef, 3, length(atoms))
    l_inv = pinv(atoms.cell')
    
    for (i, xᵢ) in enumerate(atoms.X)
        x[:, i] = (l_inv * xᵢ .- 1E-8) .> 0.5
    end

    c̆ = Array{Int64}(undef, 3, length(atoms), length(atoms))

    for i=1:length(atoms)
        for j=i:length(atoms)
            c̆[:, i, j] = x[:, j] - x[:, i]
            c̆[:, j, i] = -c̆[:, i, j]
        end
    end

    return c̆
end



function build_new_images(x::I, y::I, z::I) where I<:Integer

    xₙ, yₙ, zₙ = 2x+1, 2y+1, 2z+1

    res = Matrix{I}(undef, 3, xₙ * yₙ * zₙ)

    # Construct the cell image index vectors
    @views reshape(res[1,:], (zₙ * yₙ,  xₙ)) .= (-x:x)'
    @views reshape(res[2,:], (zₙ,  yₙ,  xₙ)) .= (-y:y)'
    @views reshape(res[3,:], (zₙ,  xₙ * yₙ)) .= -z:z

    # Relocate the origin cell to the start.
    let idx = ((xₙ * yₙ * zₙ) ÷ 2) + 1
        @views res[:, 2:idx] = res[:, 1:idx-1]
        res[:, 1] .= 0
    end

    return res

end



function cull_redundant_edge_cells(matrix, images)
    # Locate cells in the real-space matrix `matrix` that contain no data. It is important
    # that only cells that are actually zero are selected here, as opposed to those which
    # are "near-zero".
    cell_is_sparse = iszero.(eachslice(matrix, dims=3))
    
    # Remove these cells form the matrix and update image list accordingly. 
    matrix = matrix[:, :, .!cell_is_sparse]
    images = images[:,    .!cell_is_sparse]
    
    # It is possible that the above culling operation may remove non-edge cells from the
    # matrix. This is not a problem so long as the cells "above" it are also removed as
    # this just corresponds to recursively removing empty edge cells until consistency.
    # However, one must ensure that there are no widow or orphan cells.  
    
    # Check for widowed cells; these are situations where a cell, such as [ 2,  0,  0],
    # is present but its symmetrically equivalent parter cell, [-2,  0,  0], is not.
    # While widow cells are not unphysical, they should not be found in the real-space
    # matrices produced by FHI-aims due to the storage format it uses.
    let image_set = eachcol(images)
        widow_cells_found = [image for image in image_set if -image ∉ image_set]
        if length(widow_cells_found) ≠ 0
            error("At least one widowed cell has been encountered $(widow_cells_found[1])")
        end
    end
    
    # Check for orphaned cells; these are situations where a cell, such as [ 2,  0,  0],
    # is present but its parent cell, [ 1,  0,  0], is not. Orphan cells are unphysical
    # and as such their presence indicates an error has occurred. The approach taken to
    # perform this check is rather... abstract but works.
    xyz =  maximum.(eachrow(abs.(images)))

    image_map = fill(false, 2xyz .+ 1...)
    for i in eachcol(images)
        image_map[i + xyz .+ 1 ...]
    end

    for i in 1:3
        # Only test axes along which orphaned cells can actually exist.
        if xyz[i] ≠ 0
            # Try to locate the first orphaned cell
            idx = findfirst(abs.(diff(diff(image_map, dims=i), dims=i)) .≥ 2)
            # If it exists, identify its index and throw an error
            if !isnothing(idx)
                missing_idx = [idx.I...] .- xyz .- 1
                error("At least one orphaned cell has been encountered $missing_idx")
            end
        end
    end

    # Return the refined real-space matrix and its corresponding image list
    return matrix, images 
end


function remap_real_space_matrix(matrix::AbstractArray{F, 3}, images, atoms, basis_def) where F<:AbstractFloat
    # Get the block-indices of each atom-atom block within the real-space matrix.   
    block_indices = get_block_indices(length(atoms), images)
    
    # Create a copy of the block-indices and remap it to account for the fractional
    # coordinate domain shift. Specifically that of (-0.5, 5.0] → (0.0, 1.0], where
    # the former is used by FHI-aims and the latter by ACEhamiltonians. If an atom's
    # fractional coordinate exceeds 0.5 then it is wrapped. Thus the shortest periodic
    # distance between a pair of atoms may switch between crossing and not-crossing a
    # periodic boundary when shifting fractional coordinate domain.
    new_block_indices = copy(block_indices)
    let c̆ = shift_map(atoms)
        for i in eachcol(new_block_indices)
            i[3:5] -= c̆[:, i[1], i[2]]
        end
    end

    # The image list is regenerated to account for situations where the remapping process
    # maps blocks into cells that were not present in the original matrix. 
    new_images = build_new_images(maximum.(eachrow(abs.(new_block_indices[3:5, :])))...)

    # # Swap the atomic indices to account for FHI-aims calculating the bond distances via
    # # `F(u⃗,v⃗)=u⃗-v⃗` which is in opposition to ACEhamiltonians which uses `F(u⃗,v⃗)=v⃗-u⃗`.
    @inbounds for i = 1:size(block_indices, 2)
        block_indices[1, i], block_indices[2, i] = block_indices[2, i], block_indices[1, i]
    end

    # Convert the original real-space matrix into a block-wise form and create an
    # analogous one for the new matrix. 
    b_matrix = real_to_block(matrix, atoms, basis_def)
    new_b_matrix = real_to_block(
        zeros(F, size(matrix)[1:2]..., size(new_images, 2)), atoms, basis_def)
    
    # A pair of dictionaries used to map from a cell coordinate to its index in the real-
    # space matrix. For example `matrix[:, : cell_to_idx[[-1, 0, 2]]]` should return a
    # slice of `matrix` corresponding to the cell [-1, 0, 2]
    cell_to_idx = Dict(cell=>i for (i, cell) in enumerate(eachcol(images)))
    new_cell_to_idx = Dict(cell=>i for (i, cell) in enumerate(eachcol(new_images)))

    # Map the data from the original real-space matrix block by block into its corresponding
    # position in the new real-space matrix.
    for (i, j) in zip(eachcol(block_indices), eachcol(new_block_indices))
        source_block = Block(i[1:2]..., cell_to_idx[i[3:5]])
        target_block = Block(j[1:2]..., new_cell_to_idx[j[3:5]])

        # The source block must be transposed to account for the atom index exchange earlier.
        @views new_b_matrix[target_block][:, :] = b_matrix[source_block][:, :]'
    end

    # The remapping process is agnostic with respect to the contents of the matrix & will
    # therefore map all blocks regardless of whether or not they actually contain data.
    # Thus, some of the outer most cells in the new matrix may contain no meaningful data.
    # Such redundant cells are scanned for and removed from the matrix prior to returning
    # it to the user.  
    return cull_redundant_edge_cells(Array(new_b_matrix), new_images)
    
end

"""

This function constructs the block-map required to transform a real-space matrix as
generated by FHI-aims into the form expected by ACEhamiltonians.


# Arguments
- `atoms::Atoms`: an atoms object representing the system for which the real-space matrix
  map should be constructed.
- `images:AbstractMatrix`: cell index vectors identifying the cells present in the current
  real-space matrix.
- `matrices::Varargs{BlockArray}`: optional block-arrays representing the real-space
  matrices upon which the block-map will be used that can be used to detect sparsity
  and therefore remove redundant cells from the block-map.


# Returns:
- `block_map::Vector{Pair{Block, Block}}`: the block-map which maps a block in the
  original real-space matrix to that in the new real-space matrix.
- `images_out::Matrix`: cell index vectors identifying the cells present in the new real-
  space matrix.

Notes:
    By default, the block-map generation process is agnostic with respect to the contents
    of the real-space matrices upon they will be used. Therefore, all atomic blocks will
    be mapped regardless of whether or not they actually contain data. Thus, some of the
    outer most cells in remapped matrices may contain no meaningful data. The redundant
    cells can be scanned for and their removal factored into the block-map by supplying
    `generate_map` with block-wise real-space matrices, via the `matrices` argument, from
    which block sparsity can be determined. One danger in doing this is that the sparsity
    in one matrix, say the Hamiltonian, may differ from that of another matrix, such as
    the overlap matrix, and thus their block-maps may diverge. To account for this the
    `matrices` argumetn supports and arbitrary number of matrices; allowing form sparsity
    to be determined from all matrices upon which the block-map will be used.  
    
"""
function generate_map(atoms::Atoms, images::AbstractMatrix, matrices::T...) where T<:BlockArray

    # If multiple matrices are to be used then they must have matching dimensions 
    if length(matrices) > 1
        @assert all(reduce(.==, size.(matrices))) "Matrices must have matching dimensions!"
    end

    # Construct a dictionary which maps a cell coordinate to its index in the real-space
    # matrix. For example `matrix[:, : cell_to_idx[[-1, 0, 2]]]` should return the slice
    # of `matrix` corresponding to the cell [-1, 0, 2].
    cell_to_idx = Dict(cell=>i for (i, cell) in enumerate(eachcol(images)))

    # Get the indices of all non-zero atom-blocks. It is important that only blocks that
    # are truly zero get filtered out here, as opposed to those which are mearley close
    # to zero. This filtering operation is performed to prevent the formation of redundant
    # cells which frequently crop up during the remapping process. The `block_indices`
    # variable is a matrix where each column is a vector of the form:
    #   [atomic_index_atom_1, atomic_index_atom_2, cell_x, cell_y, cell_z]
    block_indices = get_block_indices(length(atoms), images)

    if length(matrices) ≠ 0
        mask = BitVector(undef, size(block_indices, 2))
        let array_mask = fill!(BitArray(undef, blocksize(matrices[1])), 0)
            for m in matrices
                array_mask .|= .!iszero.(blocks(m))
            end
            for (i, bᵢ) in enumerate(eachcol(block_indices))
                # When checking the sparsity of a block the atomic indices are exchanged for
                # reasons that are discussed later on.
                mask[i] = array_mask[bᵢ[2], bᵢ[1], cell_to_idx[bᵢ[3:5]]]
            end
        end
        block_indices = block_indices[:, mask]
    end
    
    
    # Create a copy of the block-indices and remap it to account for the fractional
    # coordinate domain shift. Specifically that of (-0.5, 0.5] → (0.0, 1.0], where
    # the former is used by FHI-aims and the latter by ACEhamiltonians. If an atom's
    # fractional coordinate exceeds 0.5 then it is wrapped. Thus the shortest periodic
    # distance between a pair of atoms may switch between crossing and not-crossing a
    # periodic boundary when shifting fractional coordinate domain.
    new_block_indices = copy(block_indices)
    let c̆ = shift_map(atoms)
        for i in eachcol(new_block_indices)
            i[3:5] -= c̆[:, i[1], i[2]]
        end
    end

    # The image list is regenerated to account for situations where the remapping process
    # maps blocks into cells that were not present in the original matrix. 
    new_images = build_new_images(maximum.(eachrow(abs.(new_block_indices[3:5, :])))...)

    # Construct the new matrix equivalent of `cell_to_idx`
    new_cell_to_idx = Dict(cell=>i for (i, cell) in enumerate(eachcol(new_images)))

    # Construct the block map which maps blocks in the original real-space matrix to those
    # in the new restructured one.
    block_map = Vector{Pair{Block{3, Int64}, Block{3, Int64}}}(undef, size(block_indices, 2))
    for (i, (bᵢ, bⱼ)) in enumerate(zip(eachcol(block_indices), eachcol(new_block_indices)))
        # The atomic indices of the source block are exchanged to account for FHI-aims
        # calculating the bond distance vector as `F(u⃗,v⃗)=u⃗-v⃗` which is in opposition
        # to ACEhamiltonians which uses `F(u⃗,v⃗)=v⃗-u⃗`.
        source_block = Block(bᵢ[2], bᵢ[1], cell_to_idx[bᵢ[3:5]])
        target_block = Block(bⱼ[1:2]..., new_cell_to_idx[bⱼ[3:5]])
        block_map[i] = source_block => target_block
    end

    return block_map, new_images
end


function apply_map!(target::BlockArray, source::BlockArray, block_map)
    for i in block_map
        # The source block must be transposed to account for the atom index exchange
        # performed in generate_map.
        @views target[i[2]][:, :] = source[i[1]][:, :]'
    end
end

function is_a_data_node(node::Union{HDF5.Group,HDF5.Dataset})
    # Returns `true` if the supplied HDF5 node is i) a Group and not a Dataset,
    # and ii) contains information for ACEhamiltonians to fit too.
    return (
        node isa HDF5.Group
        && haskey(node, "Structure")
        && haskey(node, "Info")
        && haskey(node, "Data"))
end


function remap_database(path)
    # Loop over the systems present in the databases
    
    h5open(path, "r+") do database
        
        # Loop over the top level nodes of the database.
        for node in database

            # Only operate on nodes which are HDF5.Groups containing ACEhamiltonians fitting
            # data where the Hamiltonian and overlap matrices are real-space matrices.
            if is_a_data_node(node) && !gamma_only(node)
                # Check the group to make sure that it has not already been mapped.
                if !haskey(attributes(node), "remapped") || !_clean_bool(attributes(node)["remapped"])
                    # Load in required data, along with the real-space matrices that are
                    # to be remapped, 
                    images_in = load_cell_translations(node)
                    basis_def = load_basis_set_definition(node)
                    atoms = load_atoms(node)
                    Hr_in = real_to_block(load_hamiltonian(node), atoms, basis_def)
                    Sr_in = real_to_block(load_overlap(node), atoms, basis_def)

                    # Construct the remapped real-space matrices.
                    block_map, images_out = generate_map(atoms, images_in, Hr_in, Sr_in)
                    Hr_out = block_from_atoms(atoms, basis_def, images_out)
                    Sr_out = block_from_atoms(atoms, basis_def, images_out)
                    apply_map!(Hr_out, Hr_in, block_map), apply_map!(Sr_out, Sr_in, block_map)
                    

                    # Delete the existing real-space matrices and the cell image vector
                    # matrix and overwrite them with the new ones
                    delete_object(node, "Data/H")
                    delete_object(node, "Data/S")
                    delete_object(node, "Info/Translations")
                    
                    node["Data/H"] = Array(Hr_out)
                    node["Data/S"] = Array(Sr_out)
                    node["Info/Translations"] = images_out

                    # Set the remapped tag to 'true'
                    attributes(node)["remapped"] = true
                end
            end
        end
    end
end

