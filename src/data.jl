module MatrixManipulation
using ACEhamiltonians
using JuLIP: Atoms

export BlkIdx, atomic_block_idxs, repeat_atomic_block_idxs, filter_on_site_idxs,
       filter_off_site_idxs, filter_upper_idxs, filter_lower_idxs, get_sub_blocks,
       set_sub_blocks!, get_blocks, set_blocks!, locate_and_get_sub_blocks

# ╔═════════════════════╗
# ║ Matrix Manipulation ║
# ╚═════════════════════╝

# ╭─────────────────────┬────────╮
# │ Matrix Manipulation │ BlkIdx │
# ╰─────────────────────┴────────╯
"""
An alias for `AbstractMatrix` used to signify a block index matrix. Given the frequency &
significance of block index matrices it became prudent to create an alias for it. This
helps to i) make it clear when and where a block index is used and ii) prevent having to
repeatedly explained what a block index matrix was each time one was used.

As the name suggests, these are matrices which specifies the indices of a series of atomic
blocks. The 1ˢᵗ row specifies the atomic indices of the 1ˢᵗ atom in each block and the 2ⁿᵈ
row the indices of the 2ⁿᵈ atom. That is to say `block_index_matrix[:,i]` would yield the
atomic indices of the atoms associated with the iᵗʰ block listed in `block_index_matrix`.
The 3ʳᵈ row, if present, specifies the index of the cell in in which the second atom lies;
i.e. it indexes the cell translation vector list.

For example; `BlkIdx([1 3; 2 4])` specifies two atomic blocks, the first being between
atoms 1&2, and the second between atoms 3&4; `BlkIdx([5; 6; 10])` represents the atomic
block between atoms 5&6, however in this case there is a third number 10 which give the
cell number. The cell number is can be used to help 3D real-space matrices or indicate
which cell translation vector should be applied.

It is important to note that the majority of functions that take `BlkIdx` as an argument
will assume the first and second species are consistent across all atomic-blocks. 
"""
BlkIdx = AbstractMatrix


# ╭─────────────────────┬─────────────────────╮
# │ Matrix Manipulation │ BlkIdx:Constructors │
# ╰─────────────────────┴─────────────────────╯
# These are the main methods by which `BlkIdx` instances are are constructed & expanded.

"""
    atomic_block_idxs(z_1, z_2, z_s[; order_invariant=false])

Construct a block index matrix listing all atomic-blocks present in the supplied system
where the first and second species are `z_1` and `z_2` respectively.

# Arguments
- `z_1::Int`: first interacting species
- `z_2::Int`: second interacting species
- `z_s::Vector`: atomic numbers present in system
- `order_invariant::Bool`: by default, `block_idxs` only indexes atomic blocks in which the
  1ˢᵗ & 2ⁿᵈ species are `z_1` & `z_2` respectively. However, if `order_invariant` is enabled
  then `block_idxs` will index all atomic blocks between the two species irrespective of
  which comes first.

# Returns
- `block_idxs::BlkIdx`: a 2×N matrix in which each column represents the index of an atomic block.
  With `block_idxs[:, i]` yielding the atomic indices associated with the iᵗʰ atomic-block.

# Notes
Enabling `order_invariant` is analogous to the following compound call:
`hcat(atomic_block_idxs(z_1, z_2, z_s), atomic_block_idxs(z_2, z_1, z_s))`
Furthermore, only indices associated with the origin cell are returned; if extra-cellular
blocks are required then `repeat_atomic_block_idxs` should be used.

# Examples
```
julia> atomic_numbers = [1, 1, 8]
julia> atomic_block_idxs(1, 8, atomic_numbers)
2×2 Matrix{Int64}:
 1  2
 3  3

julia> atomic_block_idxs(8, 1, atomic_numbers)
2×2 Matrix{Int64}:
 3  3
 1  2

julia> atomic_block_idxs(8, 1, atomic_numbers; order_invariant=true)
2×4 Matrix{Int64}:
 3  3  1  2
 1  2  3  3
```
"""
function atomic_block_idxs(z_1::I, z_2::I, z_s::Vector; order_invariant::Bool=false) where I<:Integer
    # This function uses views, slices and reshape operations to construct the block index
    # list rather than an explicitly nested for-loop to reduce speed.
    z_1_idx, z_2_idx = findall(==(z_1), z_s), findall(==(z_2), z_s)
    n, m = length(z_1_idx), length(z_2_idx)
    if z_1 ≠ z_2 && order_invariant
        res = Matrix{I}(undef, 2, n * m * 2)
        @views let res = res[:, 1:end ÷ 2]
            @views reshape(res[1, :], (m, n)) .= z_1_idx'
            @views reshape(res[2, :], (m, n)) .= z_2_idx
        end

        @views let res = res[:, 1 + end ÷ 2:end]
            @views reshape(res[1, :], (n, m)) .= z_2_idx'
            @views reshape(res[2, :], (n, m)) .= z_1_idx
        end
    else
        res = Matrix{I}(undef, 2, n * m)
        @views reshape(res[1, :], (m, n)) .= z_1_idx'
        @views reshape(res[2, :], (m, n)) .= z_2_idx
    end
    return res
end

function atomic_block_idxs(z_1, z_2, z_s::Atoms; kwargs...)
    return atomic_block_idxs(z_1, z_2, convert(Vector{Int}, z_s.Z); kwargs...)
end


"""
    repeat_atomic_block_idxs(block_idxs, n)

Repeat the atomic blocks indices `n` times and adds a new row specifying image number.
This is primarily intended to be used as a way to extend an atom block index list to
account for periodic images as they present in real-space matrices.

# Arguments
- `block_idxs::BlkIdx`: the block indices which are to be expanded.

# Returns
- `block_indxs_expanded::BlkIdx`: expanded block indices.

# Examples
```
julia> block_idxs = [10 10 20 20; 10 20 10 20]
julia> repeat_atomic_block_idxs(block_idxs, 2)
3×8 Matrix{Int64}:
 10  10  20  20  10  10  20  20
 10  20  10  20  10  20  10  20
  1   1   1   1   2   2   2   2
```
"""
function repeat_atomic_block_idxs(block_idxs::BlkIdx, n::T) where T<:Integer
    @assert size(block_idxs, 1) != 3 "`block_idxs` has already been expanded."
    m = size(block_idxs, 2)
    res = Matrix{T}(undef, 3, m * n)
    @views reshape(res[3, :], (m, n)) .= (1:n)'
    @views reshape(res[1:2, :], (2, m, n)) .= block_idxs
    return res
end


# ╭─────────────────────┬──────────────────╮
# │ Matrix Manipulation │ BlkIdx:Ancillary │
# ╰─────────────────────┴──────────────────╯
# Internal functions that operate on `BlkIdx`.

"""
    _block_starts(block_idxs, atoms, basis_def)

This function takes a series of atomic block indices, `block_idxs`, and returns the index
of the first element in each atomic block. This is helpful when wanting to locate atomic
blocks in a Hamiltonian or overlap matrix associated with a given pair of atoms.

# Arguments
- `block_idxs::BlkIdx`: block indices specifying the blocks whose starts are to be returned.
- `atoms::Atoms`: atoms object of the target system.
- `basis_def::BasisDef`: corresponding basis set definition.

# Returns
- `block_starts::Matrix`: A copy of `block_idxs` where the first & second rows now provide an
  index specifying where the associated block starts in the Hamiltonian/overlap matrix. The
  third row, if present, is left unchanged.
"""
function _block_starts(block_idxs::BlkIdx, atoms::Atoms, basis_def::BasisDef)
    n_orbs = Dict(k=>sum(2v .+ 1) for (k,v) in basis_def) # N∘ orbitals per species
    n_orbs_per_atom = [n_orbs[z] for z in atoms.Z] # N∘ of orbitals on each atom
    block_starts = copy(block_idxs)
    @views block_starts[1:2, :] = (
        cumsum(n_orbs_per_atom) - n_orbs_per_atom .+ 1)[block_idxs[1:2, :]]

    return block_starts
end

"""
    _sub_block_starts(z_1, z_2, s_i, s_j, basis_def)


Get the index of the first element of the sub-block formed between shells `s_i` and `s_j`
of species `z_1` and `z_2` respectively. The results of this method are commonly added to
those of `_block_starts` to give the first index of a desired sub-block in some arbitrary
Hamiltonian or overlap matrix.

# Arguments
- `z_1::Int`: species on which shell `s_i` resides.
- `z_2::Int`: species on which shell `s_j` resides.
- `s_i::Int`: first shell of the sub-block.
- `s_j::Int`: second shell of the sub-block.
- `basis_def::BasisDef`: corresponding basis set definition.

# Returns
- `sub_block_starts::Vector`: vector specifying the index in a `z_1`-`z_2` atom-block at which
  the first element of the `s_i`-`s_j` sub-block is found.

"""
function _sub_block_starts(z_1, z_2, s_i::I, s_j::I, basis_def::BasisDef) where I<:Integer
    sub_block_starts = Vector{I}(undef, 2)
    sub_block_starts[1] = sum(2basis_def[z_1][1:s_i-1] .+ 1) + 1
    sub_block_starts[2] = sum(2basis_def[z_2][1:s_j-1] .+ 1) + 1
    return sub_block_starts
end


# ╭─────────────────────┬────────────────╮
# │ Matrix Manipulation │ BlkIdx:Filters │
# ╰─────────────────────┴────────────────╯
# Filtering operators to help with differentiating between and selecting specific block
# indices or collections thereof. 

"""
    filter_on_site_idxs(block_idxs)

Filter out all but the on-site block indices.

# Arguments
- `block_idxs::BlkIdx`: block index matrix to be filtered.

# Returns
- `filtered_block_idxs::BlkIdx`: copy of `block_idxs` with only on-site block indices remaining.

"""
function filter_on_site_idxs(block_idxs::BlkIdx)
    # When `block_idxs` is a 2×N matrix then the only requirement for an interaction to be
    # on-site is that the two atomic indices are equal to one another. If `block_idxs` is a
    # 3×N matrix then the interaction must lie within the origin cell.
    if size(block_idxs, 1) == 2
        return block_idxs[:, block_idxs[1, :] .≡ block_idxs[2, :]]
    else 
        return block_idxs[:, block_idxs[1, :] .≡ block_idxs[2, :] .&& block_idxs[3, :] .== 1]
    end
end

"""
    filter_off_site_idxs(block_idxs)

Filter out all but the off-site block indices.

# Arguments
- `block_idxs::BlkIdx`: block index matrix to be filtered.

# Returns
- `filtered_block_idxs::BlkIdx`: copy of `block_idxs` with only off-site block indices remaining.

"""
function filter_off_site_idxs(block_idxs::BlkIdx)
    if size(block_idxs, 1) == 2  # Locate where atomic indices not are equal
        return block_idxs[:, block_idxs[1, :] .≠ block_idxs[2, :]]
    else  # Find where atomic indices are not equal or the cell≠1.
        return block_idxs[:, block_idxs[1, :] .≠ block_idxs[2, :] .|| block_idxs[3, :] .≠ 1]
    end
end


"""
    filter_upper_idxs(block_idxs)

Filter out atomic-blocks that reside in the lower triangle of the matrix. This is useful
for removing duplicate data in some cases. (blocks on the diagonal are retained)

# Arguments
- `block_idxs::BlkIdx`: block index matrix to be filtered.

# Returns
- `filtered_block_idxs::BlkIdx`: copy of `block_idxs` with only blocks from the upper
   triangle remaining.
"""
filter_upper_idxs(block_idxs::BlkIdx) = block_idxs[:, block_idxs[1, :] .≤ block_idxs[2, :]]


"""
    filter_lower_idxs(block_idxs)

Filter out atomic-blocks that reside in the upper triangle of the matrix. This is useful
for removing duplicate data in some cases. (blocks on the diagonal are retained)

# Arguments
- `block_idxs::BlkIdx`: block index matrix to be filtered.

# Returns
- `filtered_block_idxs::BlkIdx`: copy of `block_idxs` with only blocks from the lower
   triangle remaining.
"""
filter_lower_idxs(block_idxs::BlkIdx) = block_idxs[:, block_idxs[1, :] .≥ block_idxs[2, :]]


# ╭─────────────────────┬─────────────────╮
# │ Matrix Manipulation │ Data Assignment │
# ╰─────────────────────┴─────────────────╯

# The `_get_blocks!` methods are used to collect either atomic-blocks or sub-blocks from
# a Hamiltonian or overlap matrix.

"""
    _get_blocks!(src, target, starts)

Gather blocks from a `src` matrix and store them in the array `target`. This method is
to be used when gathering data from two-dimensional, single k-point, matrices. 

# Arguments
- `src::Matrix`: matrix from which data is to be drawn.
- `target::Array`: array in which data should be stored.
- `starts::BlkIdx`: a matrix specifying where each target block starts.

# Notes
The size of each block to ge gathered is worked out form the size of `target`.

"""
function _get_blocks!(src::Matrix{T}, target::AbstractArray{T, 3}, starts::BlkIdx) where T
    for i in 1:size(starts, 2)
        @views target[:, :, i] = src[
            starts[1, i]:starts[1, i] + size(target, 1) - 1,
            starts[2, i]:starts[2, i] + size(target, 2) - 1]
    end
end

"""
    _get_blocks!(src, target, starts)

Gather blocks from a `src` matrix and store them in the array `target`. This method is
to be used when gathering data from three-dimensional, real-space, matrices.

# Arguments
- `src::Matrix`: matrix from which data is to be drawn.
- `target::Array`: array in which data should be stored.
- `starts::BlkIdx`: a matrix specifying where each target block starts. Note that in
  this case the cell index, i.e. the third row, specifies the cell index.

# Notes
The size of each block to ge gathered is worked out form the size of `target`.

"""
function _get_blocks!(src::AbstractArray{T, 3}, target::AbstractArray{T, 3}, starts::BlkIdx) where T
    for i in 1:size(starts, 2)
        @views target[:, :, i] = src[
            starts[1, i]:starts[1, i] + size(target, 1) - 1,
            starts[2, i]:starts[2, i] + size(target, 2) - 1,
            starts[3, i]]
    end
end

# The `_set_blocks!` methods perform the inverted operation of their `_get_blocks!`
# counterparts as they place data **into** the Hamiltonian or overlap matrix. 

"""
    _set_blocks!(src, target, starts)

Scatter blocks from the `src` matrix into the `target`. This method is to be used when
assigning data to two-dimensional, single k-point, matrices. 

# Arguments
- `src::Matrix`: matrix from which data is to be drawn.
- `target::Array`: array in which data should be stored.
- `starts::BlkIdx`: a matrix specifying where each target block starts.

# Notes
The size of each block to ge gathered is worked out form the size of `target`.

"""
function _set_blocks!(src::AbstractArray{T, 3}, target::Matrix{T}, starts::BlkIdx) where T
    for i in 1:size(starts, 2)
        @views target[
            starts[1, i]:starts[1, i] + size(src, 1) - 1,
            starts[2, i]:starts[2, i] + size(src, 2) - 1,
            ] = src[:, :, i]
    end
end

"""
    _set_blocks!(src, target, starts)

Scatter blocks from the `src` matrix into the `target`. This method is to be used when
assigning data to three-dimensional, real-space, matrices.

# Arguments
- `src::Matrix`: matrix from which data is to be drawn.
- `target::Array`: array in which data should be stored.
- `starts::BlkIdx`: a matrix specifying where each target block starts. Note that in
  this case the cell index, i.e. the third row, specifies the cell index.

# Notes
The size of each block to ge gathered is worked out form the size of `target`.

"""
function _set_blocks!(src::AbstractArray{T, 3}, target::AbstractArray{T, 3}, starts::BlkIdx) where T
    for i in 1:size(starts, 2)
        @views target[
            starts[1, i]:starts[1, i] + size(src, 1) - 1,
            starts[2, i]:starts[2, i] + size(src, 2) - 1,
            starts[3, i]
            ] = src[:, :, i]
    end
end




"""
    get_sub_blocks(matrix, block_idxs, s_i, s_j, atoms, basis_def)

Collect sub-blocks of a given type from select atom-blocks in a provided matrix.

This method will collect, from `matrix`, the `s_i`-`s_j` sub-block of each atom-block
listed in `block_idxs`. It is assumed that all atom-blocks are between identical pairs
of species.

# Arguments
- `matrix::Array`: matrix from which to draw. This may be in either the 3D real-space N×N×C form
  or the single k-point N×N form; where N & C are the N∘ of orbitals & images respectively.
- `block_idxs::BlkIdx`: atomic-blocks from which sub-blocks are to be gathered.
- `s_i::Int`: first shell
- `s_j::Int`: second shell
- `atoms::Atoms`: target system's `JuLIP.Atoms` objects
- `basis_def:BasisDef`: corresponding basis set definition object (`BasisDef`)

# Returns
- `sub_blocks`: an array containing the collected sub-blocks.

# Notes
If `matrix` is supplied in its 3D real-space form then it is imperative to ensure that
the origin cell is first.
"""
function get_sub_blocks(matrix::AbstractArray{T}, block_idxs::BlkIdx, s_i, s_j, atoms::Atoms, basis_def) where T
    z_1, z_2 = atoms.Z[block_idxs[1:2, 1]]

    # Identify where each target block starts (first column and row)
    starts = _block_starts(block_idxs, atoms, basis_def)

    # Shift `starts` so it points to the start of the **sub-blocks** rather than the block
    starts[1:2, :] .+= _sub_block_starts(z_1, z_2, s_i, s_j, basis_def) .- 1

    data = Array{T, 3}(  # Array in which the resulting sub-blocks are to be collected
        undef, 2basis_def[z_1][s_i] + 1, 2basis_def[z_2][s_j] + 1, size(block_idxs, 2))
    
    # Carry out the assignment operation.
    _get_blocks!(matrix, data, starts)
    
    return data
end


"""
    set_sub_blocks(matrix, values, block_idxs, s_i, s_j, atoms, basis_def)

Place sub-block data from `values` representing the interaction between shells `s_i` &
`s_j` into the matrix at the atom-blocks listed in `block_idxs`. This is this performs
the inverse operation to `set_sub_blocks`.

# Arguments
- `matrix::Array`: matrix from which to draw. This may be in either the 3D real-space N×N×C form
  or the single k-point N×N form; where N & C are the N∘ of orbitals & images respectively.
- `values::Array`: sub-block values.
- `block_idxs::BlkIdx`: atomic-blocks from which sub-blocks are to be gathered.
- `s_i::Int`: first shell
- `s_j::Int`: second shell
- `atoms::Atoms`: target system's `JuLIP.Atoms` objects
- `basis_def:BasisDef`: corresponding basis set definition object (`BasisDef`)

# Notes
If `matrix` is supplied in its 3D real-space form then it is imperative to ensure that
the origin cell is first.
"""
function set_sub_blocks!(matrix::AbstractArray, values, block_idxs::BlkIdx, s_i, s_j, atoms::Atoms, basis_def)
    
    if size(values, 3) != size(block_idxs, 2)
        throw(DimensionMismatch(
            "The last dimensions of `values` & `block_idxs` must be of the same length."))
    end
    
    z_1, z_2 = atoms.Z[block_idxs[1:2, 1]]

    # Identify where each target block starts (first column and row)
    starts = _block_starts(block_idxs, atoms, basis_def)

    # Shift `starts` so it points to the start of the **sub-blocks** rather than the block
    starts[1:2, :] .+= _sub_block_starts(z_1, z_2, s_i, s_j, basis_def) .- 1

    # Carry out the scatter operation.
    _set_blocks!(values, matrix, starts)
end


"""
    get_blocks(matrix, block_idxs, atoms, basis_def)

Collect, from `matrix`, the blocks listed in `block_idxs`.

# Arguments
- `matrix::Array`: matrix from which to draw. This may be in either the 3D real-space N×N×C form
  or the single k-point N×N form; where N & C are the N∘ of orbitals & images respectively.
- `block_idxs::BlkIdx`: the atomic-blocks to gathered.
- `atoms::Atoms`: target system's `JuLIP.Atoms` objects
- `basis_def:BasisDef`: corresponding basis set definition object (`BasisDef`)

# Returns
- `sub_blocks`: an array containing the collected sub-blocks.

# Notes
If `matrix` is supplied in its 3D real-space form then it is imperative to ensure that
the origin cell is first.
"""
function get_blocks(matrix::AbstractArray{T}, block_idxs::BlkIdx, atoms::Atoms, basis_def) where T
    z_1, z_2 = atoms.Z[block_idxs[1:2, 1]]

    # Identify where each target block starts (first column and row)
    starts = _block_starts(block_idxs, atoms, basis_def)

    data = Array{T, 3}(  # Array in which the resulting blocks are to be collected
        undef, sum(2basis_def[z_1].+ 1), sum(2basis_def[z_2].+ 1), size(block_idxs, 2))
    
    # Carry out the assignment operation.
    _get_blocks!(matrix, data, starts)
    
    return data
end


"""
    set_sub_blocks(matrix, values, block_idxs, s_i, s_j, atoms, basis_def)

Place atom-block data from `values` into the matrix at the atom-blocks listed in `block_idxs`.
This is this performs the inverse operation to `set_blocks`.

# Arguments
- `matrix::Array`: matrix from which to draw. This may be in either the 3D real-space N×N×C form
  or the single k-point N×N form; where N & C are the N∘ of orbitals & images respectively.
- `values::Array`: sub-block values.
- `block_idxs::BlkIdx`: atomic-blocks from which sub-blocks are to be gathered.
- `s_i::Int`: first shell
- `s_j::Int`: second shell
- `atoms::Atoms`: target system's `JuLIP.Atoms` objects
- `basis_def:BasisDef`: corresponding basis set definition object (`BasisDef`)

# Notes
If `matrix` is supplied in its 3D real-space form then it is imperative to ensure that
the origin cell is first.
"""
function set_blocks!(matrix::AbstractArray, values, block_idxs::BlkIdx, atoms::Atoms, basis_def)

    if size(values, 3) != size(block_idxs, 2)
        throw(DimensionMismatch(
            "The last dimensions of `values` & `block_idxs` must be of the same length."))
    end

    # Identify where each target block starts (first column and row)
    starts = _block_starts(block_idxs, atoms, basis_def)

    # Carry out the scatter operation.
    _set_blocks!(values, matrix, starts)
end


"""
    locate_and_get_sub_blocks(matrix, z_1, z_2, s_i, s_j, atoms, basis_def)

Collects sub-blocks from the supplied matrix that correspond to off-site interactions
between the `s_i`'th shell on species `z_1` and the `s_j`'th shell on species `z_2`.

# Arguments
- `matrix`: matrix from which to draw. This may be in either the 3D real-space N×N×C form
  or the single k-point N×N form; where N & C are the N∘ of orbitals & images respectively.
- `z_1`: 1ˢᵗ species (atomic number) 
- `z_2`: 2ⁿᵈ species (atomic number)
- `s_i`: shell on 1ˢᵗ species
- `s_j`: shell on 2ⁿᵈ species
- `atoms`: target system's `JuLIP.Atoms` objects
- `basis_def`: corresponding basis set definition object (`BasisDef`)

# Returns
- `sub_blocks`: an Nᵢ×Nⱼ×M array containing the collected sub-blocks; where Nᵢ & Nⱼ are
  the number of orbitals on the `s_i`'th & `s_j`'th shells of species `z_1` & `z_2`
  respectively, and M is the N∘ of sub-blocks found.
- `block_idxs`: A matrix specifying which atomic block each sub-block in `sub_blocks`
  was taken from. If `matrix` is a 3D real space matrix then `block_idxs` will also
  include the cell index.

# Notes
If `matrix` is supplied in its 3D real-space form then it is imperative to ensure that
the origin cell is first. 
"""
locate_and_get_sub_blocks(matrix, z_1, z_2, s_i, s_j, atoms::Atoms, basis_def) = _locate_and_get_sub_blocks(matrix, z_1, z_2, s_i, s_j, atoms, basis_def)

"""
    locate_and_get_sub_blocks(matrix, z, s_i, s_j, atoms, basis_def)

Collects sub-blocks from the supplied matrix that correspond to on-site interactions
between the `s_i`'th & `s_j`'th shells on species `z`.

# Arguments
- `matrix`: matrix from which to draw. This may be in either the 3D real-space N×N×C form
  or the single k-point N×N form; where N & C are the N∘ of orbitals & images respectively.
- `z_1`: target species (atomic number) 
- `s_i`: 1ˢᵗ shell
- `s_j`: 2ⁿᵈ shell
- `atoms`: target system's `JuLIP.Atoms` objects
- `basis_def`: corresponding basis set definition object (`BasisDef`)

# Returns
- `sub_blocks`: an Nᵢ×Nⱼ×M array containing the collected sub-blocks; where Nᵢ & Nⱼ are
  the number of orbitals on the `s_i`'th & `s_j`'th shells of species `z_1` & `z_2`
  respectively, and M is the N∘ of sub-blocks found.
- `block_idxs`: A matrix specifying which atomic block each sub-block in `sub_blocks`
  was taken from. If `matrix` is a 3D real space matrix then `block_idxs` will also
  include the cell index.

# Notes
If `matrix` is supplied in its 3D real-space form then it is imperative to ensure that
the origin cell is first. 
"""
locate_and_get_sub_blocks(matrix, z, s_i, s_j, atoms::Atoms, basis_def) = _locate_and_get_sub_blocks(matrix, z, s_i, s_j, atoms, basis_def)

# Multiple dispatch is used to avoid the type instability in `locate_and_get_sub_blocks`
# associated with the creation of the `block_idxs` variable. It is also used to help
# distinguish between on-site and off-site collection operations. The following
# `_locate_and_get_sub_blocks` functions differ only in how they construct `block_idxs`.

# Off site _locate_and_get_sub_blocks functions
function _locate_and_get_sub_blocks(matrix::AbstractArray{T, 2}, z_1, z_2, s_i, s_j, atoms::Atoms, basis_def) where T
    block_idxs = atomic_block_idxs(z_1, z_2, atoms.Z)
    block_idxs = filter_off_site_idxs(block_idxs)
    # Duplicate blocks present when gathering off-site homo-atomic homo-orbital interactions
    # must be purged. 
    if (z_1 == z_2) && (s_i == s_j)
        block_idxs = filter_upper_idxs(block_idxs) 
    end
    return get_sub_blocks(matrix, block_idxs, s_i, s_j, atoms, basis_def), block_idxs
end

function _locate_and_get_sub_blocks(matrix::AbstractArray{T, 3}, z_1, z_2, s_i, s_j, atoms::Atoms, basis_def) where T
    block_idxs = atomic_block_idxs(z_1, z_2, atoms.Z)
    block_idxs = repeat_atomic_block_idxs(block_idxs, size(matrix, 3))
    block_idxs = filter_off_site_idxs(block_idxs)
    if (z_1 == z_2) && (s_i == s_j)
        block_idxs = filter_upper_idxs(block_idxs) 
    end
    return get_sub_blocks(matrix, block_idxs, s_i, s_j, atoms, basis_def), block_idxs
end

# On site _locate_and_get_sub_blocks functions
function _locate_and_get_sub_blocks(matrix::AbstractArray{T, 2}, z, s_i, s_j, atoms::Atoms, basis_def) where T
    block_idxs = atomic_block_idxs(z, z, atoms.Z)
    block_idxs = filter_on_site_idxs(block_idxs)
    return get_sub_blocks(matrix, block_idxs, s_i, s_j, atoms, basis_def), block_idxs
end

function _locate_and_get_sub_blocks(matrix::AbstractArray{T, 3}, z, s_i, s_j, atoms::Atoms, basis_def) where T
    block_idxs = atomic_block_idxs(z, z, atoms.Z)
    block_idxs = filter_on_site_idxs(block_idxs)
    block_idxs = repeat_atomic_block_idxs(block_idxs, 1)
    return get_sub_blocks(matrix, block_idxs, s_i, s_j, atoms, basis_def), block_idxs
end

end
