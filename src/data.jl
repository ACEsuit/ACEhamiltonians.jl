

module MatrixManipulation
using ACEhamiltonians
using JuLIP: Atoms

export BlkIdx, atomic_blk_idxs, repeat_atomic_blk_idxs, filter_on_site_idxs,
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
    atomic_blk_idxs(z_1, z_2, z_s[; order_invariant=false])

Construct a block index matrix listing all atomic-blocks present in the supplied system
where the first and second species are `z_1` and `z_2` respectively.

# Arguments
- `z_1::Int`: first interacting species
- `z_2::Int`: second interacting species
- `z_s::Vector`: atomic numbers present in system
- `order_invariant::Bool`: by default, `blk_idxs` only indexes atomic blocks in which the
  1ˢᵗ & 2ⁿᵈ species are `z_1` & `z_2` respectively. However, if `order_invariant` is enabled
  then `blk_idxs` will index all atomic blocks between the two species irrespective of
  which comes first.

# Returns
- `blk_idxs::BlkIdx`: a 2×N matrix in which each column represents the index of an atomic block.
  With `blk_idxs[:, i]` yielding the atomic indices associated with the iᵗʰ atomic-block.

# Notes
Enabling `order_invariant` is analogous to the following compound call:
`hcat(atomic_blk_idxs(z_1, z_2, z_s), atomic_blk_idxs(z_2, z_1, z_s))`
Furthermore, only indices associated with the origin cell are returned; if extra-cellular
blocks are required then `repeat_atomic_blk_idxs` should be used.

# Examples
```
julia> atomic_numbers = [1, 1, 8]
julia> atomic_blk_idxs(1, 8, atomic_numbers)
2×2 Matrix{Int64}:
 1  2
 3  3

julia> atomic_blk_idxs(8, 1, atomic_numbers)
2×2 Matrix{Int64}:
 3  3
 1  2

julia> atomic_blk_idxs(8, 1, atomic_numbers; order_invariant=true)
2×4 Matrix{Int64}:
 3  3  1  2
 1  2  3  3
```
"""
function atomic_blk_idxs(z_1::I, z_2::I, z_s::Vector; order_invariant::Bool=false) where I<:Integer
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

function atomic_blk_idxs(z_1, z_2, z_s::Atoms; kwargs...)
    return atomic_blk_idxs(z_1, z_2, convert(Vector{Int}, z_s.Z); kwargs...)
end


"""
    repeat_atomic_blk_idxs(blk_idxs, n)

Repeat the atomic blocks indices `n` times and adds a new row specifying image number.
This is primarily intended to be used as a way to extend an atom block index list to
account for periodic images as they present in real-space matrices.

# Arguments
- `blk_idxs::BlkIdx`: the block indices which are to be expanded.

# Returns
- `blk_indxs_expanded::BlkIdx`: expanded block indices.

# Examples
```
julia> blk_idxs = [10 10 20 20; 10 20 10 20]
julia> repeat_atomic_blk_idxs(blk_idxs, 2)
3×8 Matrix{Int64}:
 10  10  20  20  10  10  20  20
 10  20  10  20  10  20  10  20
  1   1   1   1   2   2   2   2
```
"""
function repeat_atomic_blk_idxs(blk_idxs::BlkIdx, n::T) where T<:Integer
    @assert size(blk_idxs, 1) != 3 "`blk_idxs` has already been expanded."
    m = size(blk_idxs, 2)
    res = Matrix{T}(undef, 3, m * n)
    @views reshape(res[3, :], (m, n)) .= (1:n)'
    @views reshape(res[1:2, :], (2, m, n)) .= blk_idxs
    return res
end


# ╭─────────────────────┬──────────────────╮
# │ Matrix Manipulation │ BlkIdx:Ancillary │
# ╰─────────────────────┴──────────────────╯
# Internal functions that operate on `BlkIdx`.

"""
    _blk_starts(blk_idxs, atoms, basis_def)

This function takes a series of atomic block indices, `blk_idxs`, and returns the index
of the first element in each atomic block. This is helpful when wanting to locate atomic
blocks in a Hamiltonian or overlap matrix associated with a given pair of atoms.

# Arguments
- `blk_idxs::BlkIdx`: block indices specifying the blocks whose starts are to be returned.
- `atoms::Atoms`: atoms object of the target system.
- `basis_def::BasisDef`: corresponding basis set definition.

# Returns
- `block_starts::Matrix`: A copy of `blk_idxs` where the first & second rows now provide an
  index specifying where the associated block starts in the Hamiltonian/overlap matrix. The
  third row, if present, is left unchanged.
"""
function _blk_starts(blk_idxs::BlkIdx, atoms::Atoms, basis_def::BasisDef)
    n_orbs = Dict(k=>sum(2v .+ 1) for (k,v) in basis_def) # N∘ orbitals per species
    n_orbs_per_atom = [n_orbs[z] for z in atoms.Z] # N∘ of orbitals on each atom
    block_starts = copy(blk_idxs)
    @views block_starts[1:2, :] = (
        cumsum(n_orbs_per_atom) - n_orbs_per_atom .+ 1)[blk_idxs[1:2, :]]

    return block_starts
end

"""
    _sblk_starts(z_1, z_2, s_i, s_j, basis_def)


Get the index of the first element of the sub-block formed between shells `s_i` and `s_j`
of species `z_1` and `z_2` respectively. The results of this method are commonly added to
those of `_blk_starts` to give the first index of a desired sub-block in some arbitrary
Hamiltonian or overlap matrix.

# Arguments
- `z_1::Int`: species on which shell `s_i` resides.
- `z_2::Int`: species on which shell `s_j` resides.
- `s_i::Int`: first shell of the sub-block.
- `s_j::Int`: second shell of the sub-block.
- `basis_def::BasisDef`: corresponding basis set definition.

# Returns
- `sblk_starts::Vector`: vector specifying the index in a `z_1`-`z_2` atom-block at which
  the first element of the `s_i`-`s_j` sub-block is found.

"""
function _sblk_starts(z_1, z_2, s_i::I, s_j::I, basis_def::BasisDef) where I<:Integer
    sblk_starts = Vector{I}(undef, 2)
    sblk_starts[1] = sum(2basis_def[z_1][1:s_i-1] .+ 1) + 1
    sblk_starts[2] = sum(2basis_def[z_2][1:s_j-1] .+ 1) + 1
    return sblk_starts
end


# ╭─────────────────────┬────────────────╮
# │ Matrix Manipulation │ BlkIdx:Filters │
# ╰─────────────────────┴────────────────╯
# Filtering operators to help with differentiating between and selecting specific block
# indices or collections thereof. 

"""
    filter_on_site_idxs(blk_idxs)

Filter out all but the on-site block indices.

# Arguments
- `blk_idxs::BlkIdx`: block index matrix to be filtered.

# Returns
- `filtered_blk_idxs::BlkIdx`: copy of `blk_idxs` with only on-site block indices remaining.

"""
function filter_on_site_idxs(blk_idxs::BlkIdx)
    # When `blk_idxs` is a 2×N matrix then the only requirement for an interaction to be
    # on-site is that the two atomic indices are equal to one another. If `blk_idxs` is a
    # 3×N matrix then the interaction must lie within the origin cell.
    if size(blk_idxs, 1) == 2
        return blk_idxs[:, blk_idxs[1, :] .≡ blk_idxs[2, :]]
    else 
        return blk_idxs[:, blk_idxs[1, :] .≡ blk_idxs[2, :] .&& blk_idxs[3, :] .== 1]
    end
end

"""
    filter_off_site_idxs(blk_idxs)

Filter out all but the off-site block indices.

# Arguments
- `blk_idxs::BlkIdx`: block index matrix to be filtered.

# Returns
- `filtered_blk_idxs::BlkIdx`: copy of `blk_idxs` with only off-site block indices remaining.

"""
function filter_off_site_idxs(blk_idxs::BlkIdx)
    if size(blk_idxs, 1) == 2  # Locate where atomic indices not are equal
        return blk_idxs[:, blk_idxs[1, :] .≠ blk_idxs[2, :]]
    else  # Find where atomic indices are not equal or the cell≠1.
        return blk_idxs[:, blk_idxs[1, :] .≠ blk_idxs[2, :] .|| blk_idxs[3, :] .≠ 1]
    end
end


"""
    filter_upper_idxs(blk_idxs)

Filter out atomic-blocks that reside in the lower triangle of the matrix. This is useful
for removing duplicate data in some cases. (blocks on the diagonal are retained)

# Arguments
- `blk_idxs::BlkIdx`: block index matrix to be filtered.

# Returns
- `filtered_blk_idxs::BlkIdx`: copy of `blk_idxs` with only blocks from the upper
   triangle remaining.
"""
filter_upper_idxs(blk_idxs::BlkIdx) = blk_idxs[:, blk_idxs[1, :] .≤ blk_idxs[2, :]]


"""
    filter_lower_idxs(blk_idxs)

Filter out atomic-blocks that reside in the upper triangle of the matrix. This is useful
for removing duplicate data in some cases. (blocks on the diagonal are retained)

# Arguments
- `blk_idxs::BlkIdx`: block index matrix to be filtered.

# Returns
- `filtered_blk_idxs::BlkIdx`: copy of `blk_idxs` with only blocks from the lower
   triangle remaining.
"""
filter_lower_idxs(blk_idxs::BlkIdx) = blk_idxs[:, blk_idxs[1, :] .≥ blk_idxs[2, :]]


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
    get_sub_blocks(matrix, blk_idxs, s_i, s_j, atoms, basis_def)

Collect sub-blocks of a given type from select atom-blocks in a provided matrix.

This method will collect, from `matrix`, the `s_i`-`s_j` sub-block of each atom-block
listed in `blk_idxs`. It is assumed that all atom-blocks are between identical pairs
of species.

# Arguments
- `matrix::Array`: matrix from which to draw. This may be in either the 3D real-space N×N×C form
  or the single k-point N×N form; where N & C are the N∘ of orbitals & images respectively.
- `blk_idxs::BlkIdx`: atomic-blocks from which sub-blocks are to be gathered.
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
function get_sub_blocks(matrix::AbstractArray{T}, blk_idxs::BlkIdx, s_i, s_j, atoms::Atoms, basis_def) where T
    z_1, z_2 = atoms.Z[blk_idxs[1:2, 1]]

    # Identify where each target block starts (first column and row)
    starts = _blk_starts(blk_idxs, atoms, basis_def)

    # Shift `starts` so it points to the start of the **sub-blocks** rather than the block
    starts[1:2, :] .+= _sblk_starts(z_1, z_2, s_i, s_j, basis_def) .- 1

    data = Array{T, 3}(  # Array in which the resulting sub-blocks are to be collected
        undef, 2basis_def[z_1][s_i] + 1, 2basis_def[z_2][s_j] + 1, size(blk_idxs, 2))
    
    # Carry out the assignment operation.
    _get_blocks!(matrix, data, starts)
    
    return data
end


"""
    set_sub_blocks(matrix, values, blk_idxs, s_i, s_j, atoms, basis_def)

Place sub-block data from `values` representing the interaction between shells `s_i` &
`s_j` into the matrix at the atom-blocks listed in `blk_idxs`. This is this performs
the inverse operation to `set_sub_blocks`.

# Arguments
- `matrix::Array`: matrix from which to draw. This may be in either the 3D real-space N×N×C form
  or the single k-point N×N form; where N & C are the N∘ of orbitals & images respectively.
- `values::Array`: sub-block values.
- `blk_idxs::BlkIdx`: atomic-blocks from which sub-blocks are to be gathered.
- `s_i::Int`: first shell
- `s_j::Int`: second shell
- `atoms::Atoms`: target system's `JuLIP.Atoms` objects
- `basis_def:BasisDef`: corresponding basis set definition object (`BasisDef`)

# Notes
If `matrix` is supplied in its 3D real-space form then it is imperative to ensure that
the origin cell is first.
"""
function set_sub_blocks!(matrix::AbstractArray, values, blk_idxs::BlkIdx, s_i, s_j, atoms::Atoms, basis_def)
    
    if size(values, 3) != size(blk_idxs, 2)
        throw(DimensionMismatch(
            "The last dimensions of `values` & `blk_idxs` must be of the same length."))
    end
    
    z_1, z_2 = atoms.Z[blk_idxs[1:2, 1]]

    # Identify where each target block starts (first column and row)
    starts = _blk_starts(blk_idxs, atoms, basis_def)

    # Shift `starts` so it points to the start of the **sub-blocks** rather than the block
    starts[1:2, :] .+= _sblk_starts(z_1, z_2, s_i, s_j, basis_def) .- 1

    # Carry out the scatter operation.
    _set_blocks!(values, matrix, starts)
end


"""
    get_blocks(matrix, blk_idxs, atoms, basis_def)

Collect, from `matrix`, the blocks listed in `blk_idxs`.

# Arguments
- `matrix::Array`: matrix from which to draw. This may be in either the 3D real-space N×N×C form
  or the single k-point N×N form; where N & C are the N∘ of orbitals & images respectively.
- `blk_idxs::BlkIdx`: the atomic-blocks to gathered.
- `atoms::Atoms`: target system's `JuLIP.Atoms` objects
- `basis_def:BasisDef`: corresponding basis set definition object (`BasisDef`)

# Returns
- `sub_blocks`: an array containing the collected sub-blocks.

# Notes
If `matrix` is supplied in its 3D real-space form then it is imperative to ensure that
the origin cell is first.
"""
function get_blocks(matrix::AbstractArray{T}, blk_idxs::BlkIdx, atoms::Atoms, basis_def) where T
    z_1, z_2 = atoms.Z[blk_idxs[1:2, 1]]

    # Identify where each target block starts (first column and row)
    starts = _blk_starts(blk_idxs, atoms, basis_def)

    data = Array{T, 3}(  # Array in which the resulting blocks are to be collected
        undef, sum(2basis_def[z_1].+ 1), sum(2basis_def[z_2].+ 1), size(blk_idxs, 2))
    
    # Carry out the assignment operation.
    _get_blocks!(matrix, data, starts)
    
    return data
end


"""
    set_sub_blocks(matrix, values, blk_idxs, s_i, s_j, atoms, basis_def)

Place atom-block data from `values` into the matrix at the atom-blocks listed in `blk_idxs`.
This is this performs the inverse operation to `set_blocks`.

# Arguments
- `matrix::Array`: matrix from which to draw. This may be in either the 3D real-space N×N×C form
  or the single k-point N×N form; where N & C are the N∘ of orbitals & images respectively.
- `values::Array`: sub-block values.
- `blk_idxs::BlkIdx`: atomic-blocks from which sub-blocks are to be gathered.
- `s_i::Int`: first shell
- `s_j::Int`: second shell
- `atoms::Atoms`: target system's `JuLIP.Atoms` objects
- `basis_def:BasisDef`: corresponding basis set definition object (`BasisDef`)

# Notes
If `matrix` is supplied in its 3D real-space form then it is imperative to ensure that
the origin cell is first.
"""
function set_blocks!(matrix::AbstractArray, values, blk_idxs::BlkIdx, atoms::Atoms, basis_def)

    if size(values, 3) != size(blk_idxs, 2)
        throw(DimensionMismatch(
            "The last dimensions of `values` & `blk_idxs` must be of the same length."))
    end

    # Identify where each target block starts (first column and row)
    starts = _blk_starts(blk_idxs, atoms, basis_def)

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
- `blk_idxs`: A matrix specifying which atomic block each sub-block in `sub_blocks`
  was taken from. If `matrix` is a 3D real space matrix then `blk_idxs` will also
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
- `blk_idxs`: A matrix specifying which atomic block each sub-block in `sub_blocks`
  was taken from. If `matrix` is a 3D real space matrix then `blk_idxs` will also
  include the cell index.

# Notes
If `matrix` is supplied in its 3D real-space form then it is imperative to ensure that
the origin cell is first. 
"""
locate_and_get_sub_blocks(matrix, z, s_i, s_j, atoms::Atoms, basis_def) = _locate_and_get_sub_blocks(matrix, z, s_i, s_j, atoms, basis_def)

# Multiple dispatch is used to avoid the type instability in `locate_and_get_sub_blocks`
# associated with the creation of the `blk_idxs` variable. It is also used to help
# distinguish between on-site and off-site collection operations. The following
# `_locate_and_get_sub_blocks` functions differ only in how they construct `blk_idxs`.

# Off site _locate_and_get_sub_blocks functions
function _locate_and_get_sub_blocks(matrix::AbstractArray{T, 2}, z_1, z_2, s_i, s_j, atoms::Atoms, basis_def) where T
    blk_idxs = atomic_blk_idxs(z_1, z_2, atoms.Z)
    blk_idxs = filter_off_site_idxs(blk_idxs)
    # Duplicate blocks present when gathering off-site homo-atomic homo-orbital interactions
    # must be purged. 
    if (z_1 == z_2) && (s_i == s_j)
        blk_idxs = filter_upper_idxs(blk_idxs) 
    end
    return get_sub_blocks(matrix, blk_idxs, s_i, s_j, atoms, basis_def), blk_idxs
end

function _locate_and_get_sub_blocks(matrix::AbstractArray{T, 3}, z_1, z_2, s_i, s_j, atoms::Atoms, basis_def) where T
    blk_idxs = atomic_blk_idxs(z_1, z_2, atoms.Z)
    blk_idxs = repeat_atomic_blk_idxs(blk_idxs, size(matrix, 3))
    blk_idxs = filter_off_site_idxs(blk_idxs)
    if (z_1 == z_2) && (s_i == s_j)
        blk_idxs = filter_upper_idxs(blk_idxs) 
    end
    return get_sub_blocks(matrix, blk_idxs, s_i, s_j, atoms, basis_def), blk_idxs
end

# On site _locate_and_get_sub_blocks functions
function _locate_and_get_sub_blocks(matrix::AbstractArray{T, 2}, z, s_i, s_j, atoms::Atoms, basis_def) where T
    blk_idxs = atomic_blk_idxs(z, z, atoms.Z)
    blk_idxs = filter_on_site_idxs(blk_idxs)
    return get_sub_blocks(matrix, blk_idxs, s_i, s_j, atoms, basis_def), blk_idxs
end

function _locate_and_get_sub_blocks(matrix::AbstractArray{T, 3}, z, s_i, s_j, atoms::Atoms, basis_def) where T
    blk_idxs = atomic_blk_idxs(z, z, atoms.Z)
    blk_idxs = filter_on_site_idxs(blk_idxs)
    blk_idxs = repeat_atomic_blk_idxs(blk_idxs, 1)
    return get_sub_blocks(matrix, blk_idxs, s_i, s_j, atoms, basis_def), blk_idxs
end



end

module States
using ACEhamiltonians, NeighbourLists, JuLIP
using ACEhamiltonians.MatrixManipulation: BlkIdx
using StaticArrays: SVector
using LinearAlgebra: norm, normalize
using ACE: AbstractState, CylindricalBondEnvelope, BondEnvelope, _evaluate_bond, _evaluate_env

import ACEhamiltonians.Parameters: ison
import ACE: _inner_evaluate

export BondState, AtomState, reflect, get_state

# ╔════════╗
# ║ States ║
# ╚════════╝
"""
    BondState(rr, rr0, bond)

State entities used when representing the environment about a bond.  

# Fields
- `rr`: environmental atom's position relative to the midpoint of the bond.
- `rr0`: vector between the two "bonding" atoms, i.e. the bond vector.
- `bond`: a boolean, which if true indicates the associated `BondState` entity represents
  the bond itself. If false, then the state is taken to represent an environmental atom
  about the bond rather than the bond itself.   

# Notes
If `bond == true` then `rr` should be set to `rr0/2`. If an environmental atom lies too
close to bond's midpoint then ACE may crash. Thus a small offset may be required in some
cases.

# Developers Notes
An additional field will be added at a later data to facilitate multi-species support. It
is possible that the `BondState` structure will have to be split into two sub-structures. 
"""
struct BondState{T<:SVector{3, <:AbstractFloat}, B<:Bool} <: AbstractState
    rr::T
    rr0::T
    bond::B
end


"""
    AtomState(rr)

State entity representing the environment about an atom.

# Fields
- `rr`: environmental atom's position relative to the host atom. 

# Developers Notes
An additional field will be added at a later data to facilitate multi-species support.

"""
struct AtomState{T<:SVector{3, <:AbstractFloat}} <: AbstractState
    rr::T
end

# ╭────────┬───────────────────────╮
# │ States │ General Functionality │
# ╰────────┴───────────────────────╯

# Display methods to help alleviate endless terminal spam.
function Base.show(io::IO, state::BondState)
    rr = string([round.(state.rr, digits=5)...])
    rr0 = string([round.(state.rr0, digits=5)...])
    print(io, "BondState(rr:$rr, rr0:$rr0, bond:$(state.bond))")
end

function Base.show(io::IO, state::AtomState)
    rr = string([round.(state.rr, digits=5)...])
    print(io, "AtomState(rr:$rr)")
end

# Allow for equality checks (will otherwise default to equivalency)  
Base.:(==)(x::T, y::T) where T<:BondState = x.rr == y.rr && y.rr0 == y.rr0 && x.bond == y.bond
Base.:(==)(x::T, y::T) where T<:AtomState = x.rr == y.rr

# The ≈ operator is commonly of more use for State entities than the equality
Base.isapprox(x::T, y::T; kwargs...) where T<:BondState = isapprox(x.rr, y.rr; kwargs...) && isapprox(x.rr0, y.rr0; kwargs...) && x.bond == y.bond
Base.isapprox(x::T, y::T; kwargs...) where T<:AtomState = isapprox(x.rr, y.rr; kwargs...)
Base.isapprox(x::T, y::T; kwargs...) where T<:AbstractVector{<:BondState} = all(x .≈ y)
Base.isapprox(x::T, y::T; kwargs...) where T<:AbstractVector{<:AtomState} = all(x .≈ y)

# ACE requires the `zero` method to be defined for states.
Base.zero(::Type{BondState{T, S}}) where {T, S} = BondState{T, S}(zero(T), zero(T), false)
Base.zero(::Type{AtomState{T}}) where T = AtomState{T}(zero(T))
Base.zero(::B) where B<:BondState = zero(B)
Base.zero(::B) where B<:AtomState = zero(B)
# Todo:
#   - Identify why `zero(BondState)` is being called and if the location where it is used
#     is effected by always choosing `bond` to be false. [PRIORITY:LOW]

"""
    ison(state)    

Return a boolean indicating whether the state entity is associated with either an on-site
or off-site interaction.
"""
ison(::T) where T<:AtomState = true
ison(::T) where T<:BondState = false


"""
    reflect(state)

Reflect `BondState` across the bond's midpoint. Calling on a state representing the bond A→B
will return the symmetrically B→A state. For states where `bond=true` this will flip the
sign on `rr` & `rr0`; whereas only `rr0` is flipped for the `bond=false` case.

# Arguments
- `state::BondState`: the state to be reflected.

# Returns
- `reflected_state::BondState`: a view of `state` reflected across the midpoint.

# Warnings
This is only valid for bond states whose atomic positions are given relative to the midpoint
of the bond; i.e. `envelope.λ≡0`.
"""
function reflect(state::T) where T<:BondState
    if state.bond
        return T(-state.rr, -state.rr0, true)
    else
        return T(state.rr, -state.rr0, false)
    end
end

# `reflect` is just an identify function for `AtomState` instances. This is included to
# alleviate the need for branching elsewhere in the code.
reflect(state::AtomState) = state

# ╭───────┬───────────╮
# │ State │ Factories │
# ╰───────┴───────────╯
"""
    get_state(i, atoms[; r=16.0])

Construct a state representing the environment about atom `i`.

# Arguments
- `i::Integer`: index of the atom whose state is to be constructed.
- `atoms::Atoms`: the `Atoms` object in which atom `i` resides.
- `r::AbstractFloat`: maximum distance up to which neighbouring atoms
  should be considered.

# Returns
- `state::Vector{::AtomState}`: state objects representing the environment about atom `i`.

"""
function get_state(i::Integer, atoms::Atoms; r::AbstractFloat=16.0)
    # Construct the neighbour list (this is cached so speed is not an issue)
    pair_list = JuLIP.neighbourlist(atoms, r; fixcell=false)

    # Extract environment about each relevant atom from the pair list. These will be tuples
    # of the form: (atomic-index, relative-position)
    idxs, vecs = NeighbourLists.neigs(pair_list, i) 

    # Neighbour list caching results in atoms outside the cutoff being present in `vecs`.
    filtered_vecs = filter(k -> norm(k) <= r, vecs)

    # Once multi-species support is added the atomic numbers will need to be retrieved.

    # Construct & return the `AtomState`` vector
    return AtomState.(filtered_vecs)
end


"""
    get_state(i, j, atoms, envelope[, image])

Construct a state representing the environment about the "bond" between atoms `i` & `j`.

# Arguments
- `i::Int`: atomic index of the first bonding atom.
- `j::Int`: atomic index of the second bonding atom.
- `atoms::Atoms`: the `Atoms` object in which atoms `i` and `j` reside.
- `envelope::CylindricalBondEnvelope:` an envelope specifying the volume to consider when
  constructing the state. This must be centred at the bond's midpoint; i.e. `envelope.λ≡0`.
- `image::Optional{Vector}`: a vector specifying the image in which atom `j`
  should reside; i.e. the cell translation vector. This defaults to `nothing` which will
  result in the closets periodic image of `j` being used. 

# Returns
- `state::Vector{::BondState}`: state objects representing the environment about the bond
  between atoms `i` and `j`.

# Notes
It is worth noting that a state will be constructed for the ij bond even when the distance
between them exceeds the bond-cutoff specified by the `envelope`. The maximum cutoff
distance for neighbour list construction is handled automatically.

# Warnings
- The neighbour list for the bond is constructed by applying an offset to the first atom's
  neighbour list. As such spurious states will be encountered when the ij distance exceeds
  the bond cutoff value `envelope.r0cut`. Do not ignore this warning!
- It is vital to ensure that when an `image` is supplied that all atomic coordinates are
  correctly wrapped into the unit cell. If fractional coordinates lie outside of the range
  [0, 1] then the results of this function will not be correct.

"""
function get_state(
    i::I, j::I, atoms::Atoms, envelope::CylindricalBondEnvelope,
    image::Union{AbstractVector{I}, Nothing}=nothing) where {I<:Integer}

    # Todo:
    #   - Combine the neighbour lists of atom i and j rather than just the former. This
    #     will reduce the probably of spurious state construction. But will increase run
    #     time as culling of duplicate states and bond states will be required.

    # Guard against instances where a non-zero envelope.λ value is used. When λ is set to
    # zero it means that all positions are relative to the mid-point of the bond. If this
    # is not the case then must more work is required elsewhere in the code. 
    @assert envelope.λ == 0.0 "Non-zero envelope λ values are not supported" 

    # Neighbour list cutoff distance; accounting for distances being relative relative to
    # atom `i` rather than the bond's mid-point. 
    r = sqrt((envelope.r0cut + envelope.zcut)^2 + envelope.rcut^2)

    # Neighbours list construction (about atom `i`)
    idxs, vecs, cells = _neighbours(i, atoms, r)

    # Get the bond vector between atoms i & j; where i is in the origin cell & j resides
    # in either i) closest periodic image, or ii) that specified by `image` if provided.
    if isnothing(image)
        # Identify the shortest i→j vector account for PBC.
        idx = _locate_minimum_image(j, idxs, vecs)
        rr0 = vecs[idx]
    else
        @assert length(image) == 3 "image must be a vector of length three"
        # Find the vector between atom i in the origin cell and atom j in cell `image`.
        idx = _locate_target_image(j, idxs, cells, image)
        if idx != 0
            rr0 = vecs[idx]
        else  # Special case where the cutoff was too short to catch the desired i→j bond.
            # In this case we must calculate rr0 manually. 
            rr0 = atoms.X[j] - atoms.X[i] + (adjoint(image .* atoms.pbc) * atoms.cell).parent
        end
    end

    # The i→j bond vector must be removed from `vecs` so that it does not get treated as
    # an environmental atom in the for loop later on. This operation is done even if the
    # `idx==0` to maintain type stability.
    @views vecs_no_bond = vecs[1:end .!= idx]
    
    # As the mid-point of the bond is used as the origin an offset is needed to shift
    # vectors so they're relative to the bond's midpoint and not atom `i`.
    offset =  rr0 / 2.0

    # Construct the `BondState` objects for the environmental atoms; i.e. where `bond=false`.
    env_states = Vector{BondState{typeof(rr0), Bool}}(undef, length(vecs_no_bond))
    
    for k=1:length(vecs_no_bond)
        # Offset the vectors as needed and guard against erroneous positions.
        vec = _guard_position(vecs_no_bond[k] - offset, rr0, i, j)
        env_states[k] = BondState{typeof(rr0), Bool}(vec, rr0, false)
    end
    
    # Cull states outside of the bond envelope. A double filter is required as the
    # inbuilt filter operation deviates from standard julia behaviour.
    env_states = Base.filter(x -> filter(envelope, x), env_states)
    # This line accounts for much of the run time and could do being optimised.

    # Construct the `BondState` representing the bond vector itself; i.e. `bond=true`.
    # and return it along with the other environmental `BondState` entities.
    return [BondState(rr0/2.0, rr0, true); env_states]

end


# Commonly one will need to collect multiple states rather than single states on their
# own. Hence the `get_state[s]` functions. These functions have been tagged for internal
# use only until they can be polished up. 

"""
    _get_states(blk_idxs, atoms, envelope[, images])

Get the states describing the environments about a collection of bonds as defined by the
block index list `blk_idxs`. This is effectively just a fancy wrapper for `get_state'.

# Arguments
- `blk_idxs`: atomic index matrix in which the first & second rows specify the indices of the
  two "bonding" atoms. The third row, if present, is used to index `images` to collect
  cell in which the second atom lies.
- `atoms`: the `Atoms` object in which that atom pair resides.
- `envelope`: an envelope specifying the volume to consider when constructing the states.
- `images`: image/cell translation vector look up list, only required when the `blk_idxs`
  supplies a cell index value. Defaults to `nothing`.

# Returns
- `bond_states::Vector{::Vector{::BondState}}`: a vector providing the requested bond states.


# Developers Notes
This is currently set to private until it is cleaned up.

"""
function _get_states(blk_idxs::BlkIdx, atoms::Atoms, envelope::CylindricalBondEnvelope,
    images::Union{AbstractMatrix{I}, Nothing}=nothing) where I
    if isnothing(images)
        if size(blk_idxs, 1) == 3 && any blk_idxs[3, :] != 1
            throw(ArgumentError("`idxs` provides non-origin cell indices but no 
            `images` argument was given!"))
        end
        return get_state.(blk_idxs[1, :], blk_idxs[2, :], atoms, envelope)
    else
        # If size(blk_idxs,1) == 2, i.e. no cell index is supplied then this will error out.
        # Thus not manual error handling is required. If images are supplied then blk_idxs
        # must contain the image index.
        return get_state.(blk_idxs[1, :], blk_idxs[2, :], atoms, envelope, images[blk_idxs[3, :], :])
    end
end


"""
    _get_states(blk_idxs, atoms[; r=16.0])

Get states describing the environments around each atom block specified in `blk_idxs`.
Note that `blk_idx` is assumed to contain only on-site blocks. This is just a wrapper
for `get_state'.

# Developers Notes
This is currently set to private until it is cleaned up.

"""
function _get_states(blk_idxs::BlkIdx, atoms::Atoms; r=16.0)
    if @views blk_idxs[1, :] != blk_idxs[2, :]
        throw(ArgumentError(
            "The supplied `blk_idxs` represent a hetroatomic interaction. But the function 
            called is for retrieving homoatomic states."))
    end
    return get_state.(blk_idxs[1, :], atoms; r=16.0)
end



# ╭────────┬──────────────────────────╮
# │ States │ Factory Helper Functions │
# ╰────────┴──────────────────────────╯


"""
    _neighbours(i, atoms, r)

Identify and return information about atoms neighbouring atom `i` in system `atoms`.  

# Arguments
- `i::Int`: index of the atom for which the neighbour list is to be constructed.
- `atoms::Atoms`: system in which atom `i` is located.
- `r::AbstractFloat`: cutoff distance to for the neighbour list. Due to the effects off
  cacheing this should be treated as if it were a lower bounds for the cutoff rather than
  the cutoff itself. 

# Returns
- `idxs`: atomic index of each neighbour.
- `vecs`: distance vector to each neighbour.
- `cells`: index specifying the cell in which the neighbouring atom resides.

# Warnings
Due to the effects of caching there is a high probably that the returned neighbour list
will contain neighbours at distances greater than `r`.

"""
function _neighbours(i::Integer, atoms::Atoms, r::AbstractFloat)
    pair_list = JuLIP.neighbourlist(atoms, r; fixcell=false)
    return NeighbourLists.neigss(pair_list, i)
end


"""
    _locate_minimum_image(j, idxs, vecs)

Index of the closest `j` neighbour accounting for periodic boundary conditions.   

# Arguments
- `j::Integer`: Index of the atom for for whom the minimum image is to be identified.
- `idxs::Vector{::Integer}`: Integers specifying the indices of the atoms to two which
  the distances in `vecs` correspond.
- `vecs::Vector{SVector{3, ::AbstractFloat}}`: Vectors between the the source atom and
  the target atom.

# Returns
- `index::Integer`: an index `k` for which `vecs[k]` will yield the vector between the
  source atom and the closest periodic image of atom `j`.

# Notes
If multiple minimal vectors are found, then the first one will be returned. 

"""
function _locate_minimum_image(j::I, idxs::AbstractVector{I}, vecs::AbstractVector{<:AbstractVector{F}})::I where {F<:AbstractFloat, I<:Integer}
    # Locate all entries in the neighbour list that correspond to atom `j`
    js = findall(==(j), idxs)
    # Identify which image of atom `j` is closest
    return js[findmin(norm, vecs[js])[2]]
end

"""
    _locate_target_image(j, idxs, images, image)

Search through the neighbour list for atoms with the atomic index `j` that reside in
the specified `image` and return its index. If no such match is found, as can happen
when the cutoff distance is too short, then an index of zero is returned.

# Arguments
- `j`: index of the desired neighbour.
- `idxs`: atomic indices of atoms in the neighbour list.
- `images`: image in which the neighbour list atoms reside.
- `image`: image in which the target neighbour should reside.

# Returns
- `idx::Int`: index of the first entry in the neighbour list representing an atom with the
  atom index `j` residing in the image `image`. Zero if no matches are found.

# Notes
The `images` argument is set vector of vectors here as this is represents the type returned
by the neighbour list constructor. Blocking other types prevents any misunderstandings.

# Todo:
- Test for type instabilities

"""
function _locate_target_image(j::I, idxs::AbstractVector{I}, images::AbstractVector{<:AbstractVector{I}}, image::AbstractVector{I})::I where I<:Integer
    js = findall(==(j), idxs)
    idx = findfirst(i -> all(i .== image), images[js])
    return isnothing(idx) ? zero(I) : js[idx]
end


"""
    _guard_position(vec, rr0, i, j[; cutoff=0.05])

Clean environmental bond state positions to prevent erroneous behaviour.

If, when constructing a `BondState` entity, an environmental atom is found to lie too close
to the midpoint between the two bonding atoms then ACE will crash. Thus, such cases must be
carefully checked for and, when encountered, a small offset applied.

# Arguments
- `vec`: the vector to be checked and, if necessary, cleaned. This should be relative to
  the midpoint of the bond.
- `rr0`: bond vector.
- `i`: index of the first bonding atom.
- `j`: index of the second bonding atom.
- `cutoff`: the minimum permitted distance from the origin. Vectors closer to the origin
  than this value will be offset. 

# Returns
- `vec`: the original vector `vec` or a safely offset version thereof.

# Notes
While best efforts have been made to make the offset as reproducible as possible it is not
invariant to permutation for instances where the atom lies **exactly** at the midpoint.

"""
function _guard_position(vec::T, rr0::T, i::I, j::I; cutoff=0.05) where {I<:Integer, T}
    # If the an environmental atom lies too close to the origin it must be offset to avoid
    # errors. While this introduces noise, it is better than not being able to fit. A
    # better solution should be found where and if possible. 
    vec_norm = norm(vec)
    
    # If the vector is outside of the cutoff region then no action is required 
    if vec_norm >= cutoff
        return vec
    # If the atom is inside of the cutoff region, but not at exactly at the mid-point then
    # scale the vector so that it lies outside of the cutoff region.
    elseif 0 < vec_norm
        return normalize(vec) * cutoff
    # If the atom lies exactly at the bond origin, then offset it along the bond vector.
    elseif vec_norm == 0 
        # Shift vector is in direction of the atom with the lowest atomic index, or if both
        # are the same then the first atom. This helps to make the offset a little more
        # consistent and reproducible.
        o = i <= j ? -one(I) : one(I)
        return normalize(rr0) * (cutoff * o)
    end
end


# ╭────────┬───────────╮
# │ States │ Overrides │
# ╰────────┴───────────╯
# Local override to account for the `BondState` field `be::Symbol` being replaced with the
# field `bond::Bool`.  
function _inner_evaluate(env::BondEnvelope, state::BondState)
    if state.bond
        return _evaluate_bond(env, state)
    else
        return _evaluate_env(env, state)
    end
 end

end