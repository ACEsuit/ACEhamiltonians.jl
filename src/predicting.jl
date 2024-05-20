module Predicting

using ACE, ACEbase, ACEhamiltonians, LinearAlgebra
using JuLIP: Atoms, neighbourlist
using ACE: ACEConfig, AbstractState, evaluate

using ACEhamiltonians.States: _get_states
using ACEhamiltonians.Fitting: _evaluate_real

using ACEhamiltonians: DUAL_BASIS_MODEL

export predict, predict!, cell_translations


"""
    cell_translations(atoms, cutoff)

Translation indices of all cells in within range of the origin. Multiplying any translation
index by the lattice vector will return the cell translation vector associated with said
cell. The results of this function are most commonly used in constructing the real space
matrix.

# Arguments
- `atoms::Atoms`: system for which the cell translation index vectors are to be constructed.
- `cutoff::AbstractFloat`: cutoff distance for diatomic interactions.

# Returns
- `cell_indices::Matrix{Int}`: a 3×N matrix specifying all cell images that are within
  the cutoff distance of the origin cell.

# Notes
The first index provided in `cell_indices` is always that of the origin cell; i.e. [0 0 0].
A cell is included if, and only if, at least one atom within it is within range of at least
one atom in the origin cell. Mirror image cell are always included, that is to say if the
cell [i, j, k] is present then the cell [-i, -j, -k] will also be present.
"""
function cell_translations(atoms::Atoms{T}, cutoff) where T<:AbstractFloat

    l⃗, x⃗ = atoms.cell, atoms.X
    # n_atoms::Int = size(x⃗, 2)
    n_atoms::Int = size(x⃗, 1)
    
    # Identify how many cell images can fit within the cutoff distance.
    aₙ, bₙ, cₙ = convert.(Int, cld.(cutoff, norm.(eachrow(l⃗))))
    
    # Matrix in which the resulting translation indices are to be stored
    cellᵢ = Matrix{Int}(undef, 3, (2aₙ + 1) * (2bₙ + 1) * (2cₙ + 1))

    # The first cell is always the origin cell
    cellᵢ[:, 1] .= 0
    i = 1

    # Loop over all possible cells within the cutoff range.
    for n₁=-aₙ:aₙ, n₂=-bₙ:bₙ, n₃=-cₙ:cₙ

        # Origin cell is skipped over when encountered as it is already defined.
        if n₁ ≠ 0 || n₂ ≠ 0 || n₃ ≠ 0

            # Construct the translation vector
            t⃗ = l⃗[1, :]n₁ + l⃗[2, :]n₂ + l⃗[3, :]n₃

            # Check if any atom in the shifted cell, n⃗, is within the cutoff distance of
            # any other atom in the origin cell, [0,0,0]. 
            min_distance = 2cutoff
            for atomᵢ=1:n_atoms, atomⱼ=1:n_atoms
                min_distance = min(min_distance, norm(x⃗[atomᵢ] - x⃗[atomⱼ] + t⃗))
            end

            # If an atom in the shifted cell is within the cutoff distance of another in
            # the origin cell then the cell should be included.
            if min_distance ≤ cutoff
                i += 1
                cellᵢ[:, i] .= n₁, n₂, n₃ 
            end
        end
    end

    # Return a truncated view of the cell translation index matrix. 
    return cellᵢ[:, 1:i]

end

"""
cell_translations(atoms, model)

Translation indices of all cells in within range of the origin. Note, this is a wrapper
for the base `cell_translations(atoms, cutoff)` method which automatically selects an
appropriate cutoff distance. See the base method for more info.


# Argument
- `atoms::Atoms`: system for which the cell translation index vectors are to be constructed.
- `model::Model`: model instance from which an appropriate cutoff distance is to be derived.


# Returns
- `cell_indices::Matrix{Int}`: a 3×N matrix specifying all cell images that are within
  the cutoff distance of the origin cell.

"""
function cell_translations(atoms::Atoms, model::Model)
    # Loop over the interaction cutoff distances and identify the maximum recognise
    # interaction distance and use that as the cutoff.
    return cell_translations(
        atoms, maximum(values(model.off_site_parameters.b_cut)))
end


"""
    predict!(values, basis, state)

Predict the values for a given sub-block by evaluating the provided basis on the specified
state; or more accurately the descriptor that is to be constructed from said state. Results
are placed directly into the supplied matrix `values.`

# Arguments
 - `values::AbstractMatrix`: matrix into which the results should be placed.
 - `basis::AHSubModel`: basis to be evaluated.
 - `state::Vector{States}`: state upon which the `basis` should be evaluated. 
"""
function predict!(values::AbstractMatrix, submodel::T, state::Vector{S}) where {T<:AHSubModel, S<:AbstractState}
    # If the model has been fitted then use it to predict the results; otherwise just
    # assume the results are zero.
    if is_fitted(submodel)
        # Construct a descriptor representing the supplied state and evaluate the
        # basis on it to predict the associated sub-block.  
        A = evaluate(submodel.basis, ACEConfig(state))
        B = _evaluate_real(A)
        values .= (submodel.coefficients' * B) + submodel.mean

        @static if DUAL_BASIS_MODEL
            if T<: AnisoSubModel
                A = evaluate(submodel.basis_i, ACEConfig(reflect.(state)))
                B = _evaluate_real(A)
                values .= (values + ((submodel.coefficients_i' * B) + submodel.mean_i)') / 2.0
            elseif !ison(submodel) && (submodel.id[1] == submodel.id[2]) && (submodel.id[3] == submodel.id[4])
                # If the dual basis model is being used then it is assumed that the symmetry
                # issue has not been resolved thus an additional symmetrisation operation is
                # required.
                A = evaluate(submodel.basis, ACEConfig(reflect.(state)))
                B = _evaluate_real(A)
                values .= (values + ((submodel.coefficients' * B) + submodel.mean)') / 2.0
            end
        end

    else
        fill!(values, 0.0)
    end
end


# Construct and fill a matrix with the results of a single state

"""
"""
function predict(submodel::AHSubModel, states::Vector{<:AbstractState})
    # Create a results matrix to hold the predicted values. The shape & type information
    # is extracted from the basis. However, complex types will be converted to their real
    # equivalents as results in ACEhamiltonians are always real. With the current version
    # of ACE this is the the easiest way to reliably identify the shape and float type of
    # the sub-blocks; at least that Julia is happy with.
    n, m, type = ACE.valtype(submodel.basis).parameters[3:5]
    values = Matrix{real(type)}(undef, n, m)
    predict!(values, submodel, states)
    return values
end


"""
Predict the values for a collection of sub-blocks by evaluating the provided basis on the
specified states. This is a the batch operable variant of the primary `predict!` method. 

"""
# function predict!(values::AbstractArray{<:Any, 3}, submodel::AHSubModel, states::Vector{<:Vector{<:AbstractState}})
#     for i=1:length(states)
#         @views predict!(values[:, :, i], submodel, states[i])
#     end
# end


# using Base.Threads
# function predict!(values::AbstractArray{<:Any, 3}, submodel::AHSubModel, states::Vector{<:Vector{<:AbstractState}})
#     @threads for i=1:length(states)
#         @views predict!(values[:, :, i], submodel, states[i])
#     end
# end


using Distributed
function predict!(values::AbstractArray{<:Any, 3}, submodel::AHSubModel, states::Vector{<:Vector{<:AbstractState}})
    @sync begin
        for i=1:length(states)
            @async begin
                fetch(@spawn @views predict!(values[:, :, i], submodel, states[i]))
            end
        end
    end
end


"""
"""
# function predict(submodel::AHSubModel, states::Vector{<:Vector{<:AbstractState}})
#     # Construct and fill a matrix with the results from multiple states
#     n, m, type = ACE.valtype(submodel.basis).parameters[3:5]
#     values = Array{real(type), 3}(undef, n, m, length(states))
#     predict!(values, submodel, states)
#     return values
# end


using SharedArrays
function predict(submodel::AHSubModel, states::Vector{<:Vector{<:AbstractState}})
    # Construct and fill a matrix with the results from multiple states
    n, m, type = ACE.valtype(submodel.basis).parameters[3:5]
    # values = Array{real(type), 3}(undef, n, m, length(states))
    A = SharedArray{real(type), 3}(n, m, length(states))
    predict!(values, submodel, states)
    return values
end


# Special version of the batch operable `predict!` method that is used when scattering data
# into a Vector of AbstractMatrix types rather than into a three dimensional tensor. This
# is implemented to facilitate the scattering of data into collection of sub-view arrays.
function predict!(values::Vector{<:Any}, submodel::AHSubModel, states::Vector{<:Vector{<:AbstractState}})
    for i=1:length(states)
        @views predict!(values[i], submodel, states[i])
    end
end

# using Base.Threads
# function predict!(values::Vector{<:Any}, submodel::AHSubModel, states::Vector{<:Vector{<:AbstractState}})
#     @threads for i=1:length(states)
#         @views predict!(values[i], submodel, states[i])
#     end
# end

"""
"""
function predict(model::Model, atoms::Atoms, cell_indices::Union{Nothing, AbstractMatrix}=nothing; kwargs...)
    # Pre-build neighbour list to avoid edge case which can degrade performance
    _preinitialise_neighbour_list(atoms, model)

    if isnothing(cell_indices)
        return _predict(model, atoms; kwargs...)
    else
        return _predict(model, atoms, cell_indices; kwargs...)
    end
end


function _predict(model, atoms, cell_indices)

    # Todo:-
    #   - use symmetry to prevent having to compute data for cells reflected
    #     cell pairs; i.e. [ 0,  0,  1] & [ 0,  0, -1]
    #   - Setting the on-sites to an identity should be determined by the model
    #     rather than just assuming that the user always wants on-site overlap
    #     blocks to be identity matrices.

    basis_def = model.basis_definition
    n_orbs = number_of_orbitals(atoms, basis_def)

    # Matrix into which the final results will be placed
    matrix = zeros(n_orbs, n_orbs, size(cell_indices, 2))
    
    # Mirror index map array required by `_reflect_block_idxs!`
    mirror_idxs = _mirror_idxs(cell_indices)

    # The on-site blocks of overlap matrices are approximated as identity matrix.
    if model.label ≡ "S"
        matrix[1:n_orbs+1:n_orbs^2] .= 1.0
    end

    for (species₁, species₂) in species_pairs(atoms::Atoms)  

        # Matrix containing the block indices of all species₁-species₂ atom-blocks
        blockᵢ = repeat_atomic_block_idxs(
            atomic_block_idxs(species₁, species₂, atoms), size(cell_indices, 2))

        # Identify on-site sub-blocks now as they as static over the shell pair loop.
        # Note that when `species₁≠species₂` `length(on_blockᵢ)≡0`. 
        on_blockᵢ = filter_on_site_idxs(blockᵢ)

        for (shellᵢ, shellⱼ) in shell_pairs(species₁, species₂, basis_def)

            
            # Get the off-site basis associated with this interaction
            basis_off = model.off_site_submodels[(species₁, species₂, shellᵢ, shellⱼ)]

            # Identify off-site sub-blocks with bond-distances less than the specified cutoff
            off_blockᵢ = filter_idxs_by_bond_distance(
                filter_off_site_idxs(blockᵢ), 
                envelope(basis_off).r0cut, atoms, cell_indices)
            
            # Blocks in the lower triangle are redundant in the homo-orbital interactions
            if species₁ ≡ species₂ && shellᵢ ≡ shellⱼ
                off_blockᵢ = filter_upper_idxs(off_blockᵢ)
            end

            off_site_states = _get_states( # Build states for the off-site atom-blocks
                off_blockᵢ, atoms, envelope(basis_off), cell_indices)
            
            # Don't try to compute off-site interactions if none exist
            if length(off_site_states) > 0
                let values = predict(basis_off, off_site_states) # Predict off-site sub-blocks
                    set_sub_blocks!( # Assign off-site sub-blocks to the matrix
                        matrix, values, off_blockᵢ, shellᵢ, shellⱼ, atoms, basis_def)

                    
                    _reflect_block_idxs!(off_blockᵢ, mirror_idxs)
                    values = permutedims(values, (2, 1, 3))
                    set_sub_blocks!(  # Assign data to symmetrically equivalent blocks
                        matrix, values, off_blockᵢ, shellⱼ, shellᵢ, atoms, basis_def)
                end
            end

            
            # Evaluate on-site terms for homo-atomic interactions; but only if not instructed
            # to approximate the on-site sub-blocks as identify matrices.
            if species₁ ≡ species₂ && model.label ≠ "S"
                # Get the on-site basis and construct the on-site states
                basis_on = model.on_site_submodels[(species₁, shellᵢ, shellⱼ)]
                on_site_states = _get_states(on_blockᵢ, atoms; r=radial(basis_on).R.ru)
                
                # Don't try to compute on-site interactions if none exist
                if length(on_site_states) > 0
                    let values = predict(basis_on, on_site_states) # Predict on-site sub-blocks
                        set_sub_blocks!( # Assign on-site sub-blocks to the matrix
                            matrix, values, on_blockᵢ, shellᵢ, shellⱼ, atoms, basis_def)
                        
                        values = permutedims(values, (2, 1, 3))
                        set_sub_blocks!(  # Assign data to the symmetrically equivalent blocks
                            matrix, values, on_blockᵢ, shellⱼ, shellᵢ, atoms, basis_def)
                    end
                end
            end

        end
    end
    
    return matrix    
end


# function _predict(model, atoms, cell_indices)

#     # Todo:-
#     #   - use symmetry to prevent having to compute data for cells reflected
#     #     cell pairs; i.e. [ 0,  0,  1] & [ 0,  0, -1]
#     #   - Setting the on-sites to an identity should be determined by the model
#     #     rather than just assuming that the user always wants on-site overlap
#     #     blocks to be identity matrices.

#     basis_def = model.basis_definition
#     n_orbs = number_of_orbitals(atoms, basis_def)

#     # Matrix into which the final results will be placed
#     matrix = zeros(n_orbs, n_orbs, size(cell_indices, 2))
    
#     # Mirror index map array required by `_reflect_block_idxs!`
#     mirror_idxs = _mirror_idxs(cell_indices)

#     # The on-site blocks of overlap matrices are approximated as identity matrix.
#     if model.label ≡ "S"
#         matrix[1:n_orbs+1:n_orbs^2] .= 1.0
#     end

#     for (species₁, species₂) in species_pairs(atoms::Atoms)  

#         # Matrix containing the block indices of all species₁-species₂ atom-blocks
#         blockᵢ = repeat_atomic_block_idxs(
#             atomic_block_idxs(species₁, species₂, atoms), size(cell_indices, 2))

#         # Identify on-site sub-blocks now as they as static over the shell pair loop.
#         # Note that when `species₁≠species₂` `length(on_blockᵢ)≡0`. 
#         on_blockᵢ = filter_on_site_idxs(blockᵢ)

#         Threads.@threads for (shellᵢ, shellⱼ) in shell_pairs(species₁, species₂, basis_def)

            
#             # Get the off-site basis associated with this interaction
#             basis_off = model.off_site_submodels[(species₁, species₂, shellᵢ, shellⱼ)]

#             # Identify off-site sub-blocks with bond-distances less than the specified cutoff
#             off_blockᵢ = filter_idxs_by_bond_distance(
#                 filter_off_site_idxs(blockᵢ), 
#                 envelope(basis_off).r0cut, atoms, cell_indices)
            
#             # Blocks in the lower triangle are redundant in the homo-orbital interactions
#             if species₁ ≡ species₂ && shellᵢ ≡ shellⱼ
#                 off_blockᵢ = filter_upper_idxs(off_blockᵢ)
#             end

#             off_site_states = _get_states( # Build states for the off-site atom-blocks
#                 off_blockᵢ, atoms, envelope(basis_off), cell_indices)
            
#             # Don't try to compute off-site interactions if none exist
#             if length(off_site_states) > 0
#                 let values = predict(basis_off, off_site_states) # Predict off-site sub-blocks
#                     set_sub_blocks!( # Assign off-site sub-blocks to the matrix
#                         matrix, values, off_blockᵢ, shellᵢ, shellⱼ, atoms, basis_def)

                    
#                     _reflect_block_idxs!(off_blockᵢ, mirror_idxs)
#                     values = permutedims(values, (2, 1, 3))
#                     set_sub_blocks!(  # Assign data to symmetrically equivalent blocks
#                         matrix, values, off_blockᵢ, shellⱼ, shellᵢ, atoms, basis_def)
#                 end
#             end

            
#             # Evaluate on-site terms for homo-atomic interactions; but only if not instructed
#             # to approximate the on-site sub-blocks as identify matrices.
#             if species₁ ≡ species₂ && model.label ≠ "S"
#                 # Get the on-site basis and construct the on-site states
#                 basis_on = model.on_site_submodels[(species₁, shellᵢ, shellⱼ)]
#                 on_site_states = _get_states(on_blockᵢ, atoms; r=radial(basis_on).R.ru)
                
#                 # Don't try to compute on-site interactions if none exist
#                 if length(on_site_states) > 0
#                     let values = predict(basis_on, on_site_states) # Predict on-site sub-blocks
#                         set_sub_blocks!( # Assign on-site sub-blocks to the matrix
#                             matrix, values, on_blockᵢ, shellᵢ, shellⱼ, atoms, basis_def)
                        
#                         values = permutedims(values, (2, 1, 3))
#                         set_sub_blocks!(  # Assign data to the symmetrically equivalent blocks
#                             matrix, values, on_blockᵢ, shellⱼ, shellᵢ, atoms, basis_def)
#                     end
#                 end
#             end

#         end
#     end
    
#     return matrix    
# end


function _predict(model, atoms)
    # Currently this method has the tendency to produce non-positive definite overlap
    # matrices when working with the aluminum systems, however this is not observed in
    # the silicon systems. As such this function should not be used for periodic systems
    # until the cause of this issue can be identified. 
    @warn "This function is not to be trusted"
    # TODO:
    #   - It seems like the `filter_idxs_by_bond_distance` method is not working as intended
    #     as results change based on whether this is enabled or disabled.

    # See comments in the real space matrix version of `predict` more information. 
    basis_def = model.basis_definition
    n_orbs = number_of_orbitals(atoms, basis_def)

    matrix = zeros(n_orbs, n_orbs)

    # If constructing an overlap matrix then the on-site blocks can just be set to
    # an identify matrix.
    if model.label ≡ "S" 
        matrix[1:n_orbs+1:end] .= 1.0
    end
    
    for (species₁, species₂) in species_pairs(atoms::Atoms)  

        blockᵢ = atomic_block_idxs(species₁, species₂, atoms)

        on_blockᵢ = filter_on_site_idxs(blockᵢ)

        for (shellᵢ, shellⱼ) in shell_pairs(species₁, species₂, basis_def)

            basis_off = model.off_site_submodels[(species₁, species₂, shellᵢ, shellⱼ)]

            off_blockᵢ = filter_idxs_by_bond_distance(
                filter_off_site_idxs(blockᵢ), 
                envelope(basis_off).r0cut, atoms)
            
            if species₁ ≡ species₂ && shellᵢ ≡ shellⱼ
                off_blockᵢ = filter_upper_idxs(off_blockᵢ)
            end

            off_site_states = _get_states(
                off_blockᵢ, atoms, envelope(basis_off))
            
            if length(off_site_states) > 0 
                let values = predict(basis_off, off_site_states)
                    set_sub_blocks!(
                        matrix, values, off_blockᵢ, shellᵢ, shellⱼ, atoms, basis_def)

                    
                    _reflect_block_idxs!(off_blockᵢ)
                    values = permutedims(values, (2, 1, 3))
                    set_sub_blocks!(
                        matrix, values, off_blockᵢ, shellⱼ, shellᵢ, atoms, basis_def)
                end
            end

            
            if species₁ ≡ species₂ && model.label ≠ "S"
                basis_on = model.on_site_submodels[(species₁, shellᵢ, shellⱼ)]
                on_site_states = _get_states(on_blockᵢ, atoms; r=radial(basis_on).R.ru)
                

                if length(on_site_states) > 0
                    let values = predict(basis_on, on_site_states)
                        set_sub_blocks!(
                            matrix, values, on_blockᵢ, shellᵢ, shellⱼ, atoms, basis_def)
                        
                        values = permutedims(values, (2, 1, 3))
                        set_sub_blocks!(
                            matrix, values, on_blockᵢ, shellⱼ, shellᵢ, atoms, basis_def)
                    end
                end
            end

        end
    end
    
    return matrix    
end


# ╭───────────────────────────╮
# │ Internal Helper Functions │
# ╰───────────────────────────╯

"""
Construct the mirror index map required by `_reflect_block_idxs!`.
"""
function _mirror_idxs(cell_indices)
    mirror_idxs = Vector{Int}(undef, size(cell_indices, 2))
    let cell_to_index = Dict(cell=>idx for (idx, cell) in enumerate(eachcol(cell_indices)))
        for i=1:length(mirror_idxs)
            mirror_idxs[i] = cell_to_index[cell_indices[:, i] * -1]
        end 
    end
    return mirror_idxs
end


"""
This function takes in a `BlkIdx` entity as an argument & swaps the atomic block indices;
i.e. [1, 2] → [2, 1].
"""
function _reflect_block_idxs!(block_idxs::BlkIdx)
    @inbounds for i=1:size(block_idxs, 2)
        block_idxs[1, i], block_idxs[2, i] = block_idxs[2, i], block_idxs[1, i]
    end
    nothing
end

"""
Inverts a `BlkIdx` instance by swapping the atomic-indices and substitutes the cell index
for its reflected counterpart; i.e. [i, j, k] → [j, i, idx_mirror[k]].
"""
function _reflect_block_idxs!(block_idxs::BlkIdx, idx_mirror::AbstractVector)
    @inbounds for i=1:size(block_idxs, 2)
        block_idxs[1, i], block_idxs[2, i] = block_idxs[2, i], block_idxs[1, i]
        block_idxs[3, i] = idx_mirror[block_idxs[3, i]]
    end
    nothing
end


function _maximum_distance_estimation(model::Model)
    # Maximum radial distance (on-site)
    max₁ = maximum(values(model.on_site_parameters.e_cut_out))
    # Maximum radial distance (off-site)
    max₂ = maximum(values(model.off_site_parameters.e_cut_out))
    # Maximum effective envelope distance 
    max₃ = maximum(
        [sqrt((env.r0cut + env.zcut)^2 + (env.rcut/2)^2)
        for env in envelope.(values(model.off_site_submodels))])
    
    return max(max₁, max₂, max₃)

end

"""
The construction of neighbour lists can be computationally intensive. As such lists are
used frequently by the lower levels of the code, they are cached & only every recomputed
recomputed if the requested cutoff distance exceeds that used when building the cached
version. It has been found that because each basis can have a different cutoff distance
it is possible that, due to the effects of evaluation order, that the neighbour list can
end up being reconstructed many times. This can be mitigated by anticipating what the
largest cutoff distance is likely to be and pre-building the neighbour list ahead of time.
Hence this function.
"""
function _preinitialise_neighbour_list(atoms::Atoms, model::Model)
    # Get a very rough approximation for what the largest cutoff distance might be when
    # constructing the neighbour list. 
    r = _maximum_distance_estimation(model) * 1.1

    # Construct construct and cache the maximum likely neighbour list 
    neighbourlist(atoms, r; fixcell=false);

    nothing
end


end
