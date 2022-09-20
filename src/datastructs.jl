module DataSets
using ACEhamiltonians
using LinearAlgebra: norm
using JuLIP: Atoms
using ACEhamiltonians.MatrixManipulation: BlkIdx, _distance_mask
using ACEhamiltonians.States: reflect, _get_states, _neighbours, _locate_minimum_image, _locate_target_image

import ACEhamiltonians.Parameters: ison

# ╔══════════╗
# ║ DataSets ║
# ╚══════════╝

# `AbstractFittingDataSet` based structures contain all data necessary to perform a fit. 
abstract type AbstractFittingDataSet end

export DataSet, filter_sparse, filter_bond_distance, get_dataset, AbstractFittingDataSet

"""
    DataSet(values, blk_idxs, states)

A structure for storing collections of sub-blocks & states representing the environments
from which they came. These are intended to be used during the model fitting process.
While the block index matrix is not strictly necessary to the fitting process it is useful
enough to merit inclusion.

# Fields
- `values::AbstractArray`: an i×j×n matrix containing extracted the sub-block values;
  where i & j are the number of orbitals associate with the two shells, and n the number
  of sub-blocks. 
- `blk_idxs::BlkIdx`: a block index matrix specifying from which block each sub-block in
  `values` was taken from. This acts mostly as meta-data.
- `states`: states representing the atomic-block from which each sub-block was taken.

# Notes
Some useful properties of `DataSet` instances have been highlighted below:
- Addition can be used to combine one or more datasets.
- Indexing can be used to take a sub-set of a dataset.
- `size` acts upon `DataSet.values`.
- `length` returns `size(DataSet.values, 3)`, i.e. the number of sub-blocks.
- The adjoint of a `DataSet` will return a copy where:
    - `values` is the hermitian conjugate of its parent.
    - atomic indices `blk_idxs` have been exchanged, i.e. rows 1 & 2 are swapped.
    - `states` are reflected (where appropriate) .

"""
struct DataSet{V<:AbstractArray{<:Any, 3}, B<:BlkIdx, S<:AbstractVector} <: AbstractFittingDataSet
    values::V
    blk_idxs::B
    states::S
end

# ╭──────────┬───────────────────────╮
# │ DataSets │ General Functionality │
# ╰──────────┴───────────────────────╯

function Base.show(io::IO, data_set::T) where T<:AbstractFittingDataSet
    F = valtype(data_set.values)
    mat_shape = join(size(data_set), '×')
    print(io, "$(nameof(T)){$F}($mat_shape)")
end

function Base.:(==)(x::T, y::T) where T<:DataSet
    return x.blk_idxs == y.blk_idxs && x.values == y.values && x.states == y.states
end

# Two or more AbstractFittingDataSet entities can be added together via the `+` operator
Base.:(+)(x::T, y::T) where T<:AbstractFittingDataSet = T(
    (cat(getfield(x, fn), getfield(y, fn), dims=ndims(getfield(x, fn)))
    for fn in fieldnames(T))...)

# Allow AbstractFittingDataSet objects to be indexed so that a subset may be selected. This is
# mostly used when filtering data.
function Base.getindex(data_set::T, idx) where T<:AbstractFittingDataSet
    return T((_getindex_helper(data_set, fn, idx) for fn in fieldnames(T))...)
end


function _getindex_helper(data_set, fn, idx)
    # This abstraction helps to speed up calls to Base.getindex.
    return let data = getfield(data_set, fn)
        collect(selectdim(data, ndims(data), idx))
    end
end

Base.lastindex(data_set::AbstractFittingDataSet) = length(data_set)
Base.length(data_set::AbstractFittingDataSet) = size(data_set, 3)
Base.size(data_set::AbstractFittingDataSet, dim::Integer) = size(data_set.values, dim)
Base.size(data_set::AbstractFittingDataSet) = size(data_set.values)

"""
Return a copy of the provided `DataSet` in which i) the sub-blocks (i.e. the `values`
field) have set to their adjoint, ii) the atomic indices in the `blk_idxs` field have
been exchanged, and iii) all `BondStates` in the `states` field have been reflected.
"""
function Base.adjoint(data_set::T) where T<:DataSet
    swapped_blk_idxs = copy(data_set.blk_idxs); _swaprows!(swapped_blk_idxs, 1, 2)
    return T(
        # Transpose and take the complex conjugate of the sub-blocks
        conj(permutedims(data_set.values, (2, 1, 3))),
        # Swap the atomic indices in `blk_idxs` i.e. atom block 1-2 is now block 2-1
        swapped_blk_idxs,
        # Reflect bond states across the mid-point of the bond.
        [reflect.(i) for i in data_set.states])
end

function _swaprows!(matrix::AbstractMatrix, i::Integer, j::Integer)
    @inbounds for k = 1:size(matrix, 2)
        matrix[i, k], matrix[j, k] = matrix[j, k], matrix[i, k]
    end
end


"""
    ison(dataset)    

Return a boolean indicating whether the `DataSet` entity contains on-site data.
"""
ison(x::T) where T<:DataSet = ison(x.states[1][1])



# ╭──────────┬─────────╮
# │ DataSets │ Filters │
# ╰──────────┴─────────╯


"""
    filter_sparse(dataset[, threshold=1E-8])

Filter out data-points with fully sparse sub-blocks. Only data-points with sub-blocks
containing at least one element whose absolute value is greater than the specified
threshold will be retained.

# Arguments
- `dataset::AbstractFittingDataSet`: the dataset that is to be filtered.
- `threshold::AbstractFloat`: value below which an element will be considered to be
  sparse. This will default to 1E-8 if omitted.

# Returns
- `filtered_dataset`: a copy of the, now filtered, dataset.

"""
function filter_sparse(dataset::AbstractFittingDataSet, threshold::AbstractFloat=1E-8)
    return dataset[vec(any(abs.(dataset.values) .>= threshold, dims=(1,2)))]
end


"""
filter_bond_distance(dataset, distance)

Filter out data-points whose bond-vectors exceed the supplied cutoff distance. This allows
for states that will not be used to be removed during the data selection process rather
than during evaluation. Note that this is only applicable to off-site datasets. 

# Arguments
- `dataset::AbstractFittingDataSet`: tje dataset that is to be filtered.
- `distance::AbstractFloat`: data-points with bond distances exceeding this value will be
  filtered out.  

# Returns
- `filtered_dataset`: a copy of the, now filtered, dataset.

"""
function filter_bond_distance(dataset::AbstractFittingDataSet, distance::AbstractFloat)
    if length(dataset) != 0  # Don't try and check the state unless one exists. 
        # Throw an error if the user tries to apply the a bond filter to an on-site dataset
        # where there is no bond to be filtered.
        @assert !ison(dataset) "Only applicable to off-site datasets" 
    end
    return dataset[[norm(i[1].rr0) <= distance for i in dataset.states]]
end




# ╭──────────┬───────────╮
# │ DataSets │ Factories │
# ╰──────────┴───────────╯
# This section will hold the factory methods responsible for automating the construction
# of `DataSet` entities. The `get_dataset` methods will be implemented once the `Basis`
# structures have been implemented.


# This is just a reimplementation of `filter_idxs_by_bond_distance` that allows for `blocks`
# to get filtered as well
function _filter_bond_idxs(blocks, block_idxs::BlkIdx, distance::AbstractFloat, atoms::Atoms, images)
    let mask = _distance_mask(block_idxs::BlkIdx, distance::AbstractFloat, atoms::Atoms, images) 
        return blocks[:, :, mask], block_idxs[:, mask]
    end
end



"""
# Todo
 - This could be made more performant.
"""
function _filter_sparse(values, block_idxs, tolerance)
    mask = vec(any(abs.(values) .>= tolerance, dims=(1,2)))
    return values[:, :, mask], block_idxs[:, mask]
end


"""

# Arguments
- `matrix`:
- `atoms`:
- `basis`:
- `basis_def`:
- `tolerance`:
- `filter_bonds`:

"""
function get_dataset(
    matrix::AbstractArray, atoms::Atoms, basis::Basis, basis_def,
    images::Union{Matrix, Nothing}=nothing;
    tolerance::Union{Nothing, <:AbstractFloat}=nothing, filter_bonds::Bool=false)

    if ndims(matrix) == 3 && isnothing(images)
        throw("`images` must be provided when provided with a real space `matrix`.")
    end

    # Locate and gather the sub-blocks correspond the interaction associated with `basis`  
    blocks, block_idxs = locate_and_get_sub_blocks(matrix, basis.id..., atoms, basis_def)

    # If gathering off-site data and `filter_bonds` is `true` then remove data-points
    # associated with interactions between atom pairs whose bond-distance exceeds the
    # cutoff as specified by the bond envelope. This prevents having to construct states
    # (which is an expensive process) for interactions which will just be deleted later
    # on. Enabling this can save a non-trivial amount of time and memory. 
    if !ison(basis) && filter_bonds
        blocks, block_idxs = _filter_bond_idxs(
            blocks, block_idxs, envelope(basis).r0cut, atoms, images) 
    end

    if !isnothing(tolerance) # Filter out sparse sub-blocks; but only if instructed to 
        blocks, block_idxs = _filter_sparse(blocks, block_idxs, tolerance)
    end

    # Construct states for each of the sub-blocks.
    if ison(basis)
        # For on-site states the cutoff radius is provided; this results in redundant
        # information being culled here rather than later on; thus saving on memory.
        states = _get_states(block_idxs, atoms; r=radial(basis).R.ru)
    else
        # For off-site states the basis' bond envelope must be provided.
        states = _get_states(block_idxs, atoms, envelope(basis), images)
    end

    # Construct and return the requested DataSet object
    dataset = DataSet(blocks, block_idxs, states)

    return dataset
end

 
end
# Notes
# - The matrix and array versions of `get_dataset` could easily be combined.
# - The `get_dataset` method is likely to suffer from type instability issues as it is
#   unlikely that Julia will know ahead of time whether the `DataSet` structure returned
#   will contain on or off-states states; each having different associated structures.
#   Thus type ambiguities in the `Basis` structures should be alleviated.


