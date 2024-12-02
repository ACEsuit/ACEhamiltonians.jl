module DataSets
using ACEhamiltonians
using Random: shuffle
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

export DataSet, filter_sparse, filter_bond_distance, get_dataset, AbstractFittingDataSet, random_split, random_sample, random_distance_sample

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
function Base.getindex(data_set::T, idx::UnitRange) where T<:AbstractFittingDataSet
    return T((_getindex_helper(data_set, fn, idx) for fn in fieldnames(T))...)
end

function Base.getindex(data_set::T, idx::Vararg{<:Integer}) where T<:AbstractFittingDataSet
    return data_set[collect(idx)]
end

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
  sparse. This will defaudfgflt to 1E-8 if omitted.

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

"""
    random_sample(dataset, n[; with_replacement])

Select a random subset of size `n` from the supplied `dataset`.

# Arguments
- `dataset::AbstractFittingDataSet`: dataset to be sampled.
- `n::Integer`: number of sample points
- `with_replacement::Bool`: if true (default) then duplicate samples will not be drawn.

# Returns
- `sample_dataset::AbstractFittingDataSet`: a randomly selected subset of `dataset`
  of size `n`.
"""
function random_sample(dataset::AbstractFittingDataSet, n::Integer; with_replacement=true)
    if with_replacement
        @assert n ≤ length(dataset) "Sample size cannot exceed dataset size"
        return dataset[shuffle(1:length(dataset))[1:n]]
    else
        return dataset[rand(1:length(dataset), n)]
    end
end

"""
   random_split(dataset, x) 

Split the `dataset` into a pair of randomly selected subsets.

# Arguments
- `dataset::AbstractFittingDataSet`: dataset to be partitioned.
- `x::AbstractFloat`: partition ratio; the fraction of samples to be
  placed into the first subset. 

"""
function random_split(dataset::DataSet, x::AbstractFloat)
    split_index = Int(round(length(dataset)x))
    idxs = shuffle(1:length(dataset))
    return dataset[idxs[1:split_index]], dataset[idxs[split_index + 1:end]]
    
end



"""
    random_distance_sample(dataset, n[; with_replacement=true, rng=rand])

Select a random subset of size `n` from the supplied `dataset` via distances.

This functions in a similar manner to `random_sample` but selects points based on their
bond length. This is intended to ensure a more even sample.

# Arguments
- `dataset::AbstractFittingDataSet`: dataset to be sampled.
- `n::Integer`: number of sample points
- `with_replacement::Bool`: if true (default) then duplicate samples will not be drawn.
- `rng::Function`: function to generate random numbers.

"""
function random_distance_sample(dataset, n; with_replacement=true, rng=rand)
    
    @assert length(dataset) ≥ n
    # Construct an array storing the bond lengths of each state, sort it and generate
    # the sort permutation array to allow elements in r̄ to be mapped back to their
    # corresponding state in the dataset.
    r̄ = [norm(i[1].rr0) for i in dataset.states]
    r̄_perm = sortperm(r̄)
    r̄[:] = r̄[r̄_perm]
    m = length(dataset)

    # Work out the maximum & minimum bond distance as well as the range
    r_min = minimum(r̄)
    r_max = maximum(r̄)
    r_range = r_max - r_min
    
    # Preallocate transient index storage array
    selected_idxs = zeros(Int, n)

    for i=1:n
        # Select a random distance r ∈ [min(r̄), max(r̄)]
        r = rng() * r_range + r_min

        # Identify the first element of r̄ ≥ r and the last element ≤ r 
        idxs = searchsorted(r̄, r)
        idx_i, idx_j = minmax(first(idxs), last(idxs))

        # Expand the window by one each side, but don't exceed the array's bounds
        idx_i = max(idx_i-1, 1)
        idx_j = min(idx_j+1, m)

        # Identify which element is closest to r and add the associated index
        # to the selected index array.
        idx = last(findmin(j->abs(r-j), r̄[idx_i:idx_j])) + idx_i - 1

        # If this state has already been selected then replace it with the next
        # closest one.
 
        if with_replacement && r̄_perm[idx] ∈ selected_idxs

            # Identify the indices corresponding to the first states with longer and shorter
            # bond lengths than the current, duplicate, state.
            lb = max(idx-1, 1)
            ub = min(idx+1, m)
            
            while lb >= 1 && r̄_perm[lb] ∈ selected_idxs
                lb -= 1
            end
  
            while ub <= m && r̄_perm[ub] ∈ selected_idxs
                ub += 1
            end
            
            # Select the closets valid state
            new_idx = 0
            dx = Inf

            if lb != 0 && lb != idx
                new_idx = lb
                dx = abs(r̄[lb] - r)
            end

            if ub != m+1 && (abs(r̄[ub] - r) < dx) && ub != idx
                new_idx = ub
            end

            idx = new_idx
 
        end


        selected_idxs[i] = r̄_perm[idx]
    end

    return dataset[selected_idxs]

end

# ╭──────────┬───────────╮
# │ DataSets │ Factories │
# ╰──────────┴───────────╯
# This section will hold the factory methods responsible for automating the construction
# of `DataSet` entities. The `get_dataset` methods will be implemented once the `AHSubModel`
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
get_dataset(matrix, atoms, submodel, basis_def[, images; tolerance, filter_bonds, focus])

Construct and return a `DataSet` entity containing the minimal data required to fit a
`AHSubModel` entity.

# Arguments
- `matrix`: matrix from which sub-blocks are to be gathered.
- `atoms`: atoms object representing the system to which the matrix pertains.
- `submodel`: `AHSubModel` entity for the desired sub-block; the `id` field is used to identify
  which sub-blocks should be gathered and how they should be gathered.
- `basis_def`: a basis definition specifying what orbitals are present on each species.
- `images`: cell translation vectors associated with the matrix, this is only required
  when the `matrix` is in the three-dimensional real-space form.

# Keyword Arguments
- `tolerance`: specifying a float value will enact sparse culling in which only sub-blocks
  with at least one element greater than the permitted tolerance will be included. This
  is used to remove all zero, or near zero, sub-blocks. This is disabled by default.
- `filter_bonds`: if set to `true` then only interactions within the permitted cutoff
  distance will be returned. This is only valid off-site interactions and is disabled
  by default. The cut-off distance is extracted from the bond envelope contained within
  the `basis` object. This defaults to `true` for off-site interactions.
- `focus`: the `focus` argument allows the `get_dataset` call to return only a sub-set
  of possible data-points. If a vector of atomic indices is provided then only on/off-
  site sub-blocks for/between those atoms will be returned; i.e. [1, 2] would return
  on-sites 1-on, 2-on and off-sites 1-1-off, 1-2-off, 2-1-off, & 2-2-off. If a matrix
  is provided, like so [1 2; 3 4] then only the associated off-site sub-blocks will be
  returned, i.e. 1-2-off and 3-4-off. Note that the matrix form is only valid when
  retrieving off-site sub-blocks.
- `no_reduce`: by default symmetrically redundant sub-blocks will not be gathered; this
  equivalent blocks from be extracted from the upper and lower triangles of the Hamiltonian
  and overlap matrices. This will default to `false`, however it is sometimes useful to
  disable this when debugging.

# Todo:
 - Warn that only the upper triangle is returned and discuss how this effects "focus".
 
"""
function get_dataset(
    matrix::AbstractArray, atoms::Atoms, submodel::AHSubModel, basis_def,
    images::Union{Matrix, Nothing}=nothing;
    tolerance::Union{Nothing, <:AbstractFloat}=nothing, filter_bonds::Bool=true,
    focus::Union{Vector{<:Integer}, Matrix{<:Integer}, Nothing}=nothing,
    no_reduce=false)

    if ndims(matrix) == 3 && isnothing(images)
        throw("`images` must be provided when provided with a real space `matrix`.")
    end

    # Locate and gather the sub-blocks correspond the interaction associated with `basis`  
    blocks, block_idxs = locate_and_get_sub_blocks(matrix, submodel.id..., atoms, basis_def; focus=focus, no_reduce=no_reduce)

    if !isnothing(focus)
        mask = ∈(focus).(block_idxs[1, :]) .& ∈(focus).(block_idxs[2, :])
        block_idxs = block_idxs[:, mask]
        blocks = blocks[:, :, mask]
    end

    # If gathering off-site data and `filter_bonds` is `true` then remove data-points
    # associated with interactions between atom pairs whose bond-distance exceeds the
    # cutoff as specified by the bond envelope. This prevents having to construct states
    # (which is an expensive process) for interactions which will just be deleted later
    # on. Enabling this can save a non-trivial amount of time and memory. 
    if !ison(submodel) && filter_bonds
        blocks, block_idxs = _filter_bond_idxs(
            blocks, block_idxs, envelope(submodel).r0cut, atoms, images) 
    end

    if !isnothing(tolerance) # Filter out sparse sub-blocks; but only if instructed to 
        blocks, block_idxs = _filter_sparse(blocks, block_idxs, tolerance)
    end

    # Construct states for each of the sub-blocks.
    if ison(submodel)
        # For on-site states the cutoff radius is provided; this results in redundant
        # information being culled here rather than later on; thus saving on memory.
        states = _get_states(block_idxs, atoms; r=radial(submodel).R.ru)
    else
        if size(block_idxs, 2) == 0
            states = zeros(0)
        else
            states = _get_states(block_idxs, atoms, envelope(submodel), images)
        end
        # # For off-site states the basis' bond envelope must be provided.
        # states = _get_states(block_idxs, atoms, envelope(submodel), images)
    end

    # Construct and return the requested DataSet object
    dataset = DataSet(blocks, block_idxs, states)

    return dataset
end



"""
Construct a collection of `DataSet` instances storing the information required to fit
their associated `AHSubModel` entities. This convenience function will call the original
`get_dataset` method for each and every basis in the supplied model and return a
dictionary storing once dataset for each basis in the model.

"""
function get_dataset(
    matrix::AbstractArray, atoms::Atoms, model::Model,
    images::Union{Matrix, Nothing}=nothing; kwargs...)

    basis_def = model.basis_definition
    on_site_data = Dict(
        basis.id => get_dataset(matrix, atoms, basis, basis_def, images; kwargs...)
        for basis in values(model.on_site_submodels))

    off_site_data = Dict(
        basis.id => get_dataset(matrix, atoms, basis, basis_def, images; kwargs...)
        for basis in values(model.off_site_submodels))

    return on_site_data, off_site_data
end


 
end

# Notes
# - The matrix and array versions of `get_dataset` could easily be combined.
# - The `get_dataset` method is likely to suffer from type instability issues as it is
#   unlikely that Julia will know ahead of time whether the `DataSet` structure returned
#   will contain on or off-states states; each having different associated structures.
#   Thus type ambiguities in the `AHSubModel` structures should be alleviated.
