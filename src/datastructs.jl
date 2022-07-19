module DataSets
using ACEhamiltonians
using LinearAlgebra: norm
using JuLIP: Atoms
using ACEhamiltonians.MatrixManipulation: BlkIdx
using ACEhamiltonians.States: reflect

import ACEhamiltonians.Parameters: ison

# ╔══════════╗
# ║ DataSets ║
# ╚══════════╝

# `AbstractFittingDataSet` based structures contain all data necessary to perform a fit. 
abstract type AbstractFittingDataSet end

export DataSet, filter_sparse, filter_bond_distance

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

end