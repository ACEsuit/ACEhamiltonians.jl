module Fitting2
using HDF5, ACE, ACEbase, ACEhamiltonians, StaticArrays, Statistics, LinearAlgebra, SparseArrays
using HDF5: Group
using JuLIP: Atoms
using ACE: ACEConfig, evaluate, scaling, AbstractState, SymmetricBasis
using ACEhamiltonians.Fitting: evaluateval_real, solve_ls
using ACEhamiltonians.Common: number_of_orbitals
using ACEhamiltonians.Bases: envelope
using ACEhamiltonians.DatabaseIO: load_hamiltonian_gamma, load_overlap_gamma

export fit!

# Once the bond inversion issue has been resolved the the redundant models will no longer
# be required. The changes needed to be made in this file to remove the redundant model
# are as follows:
#   - Remove inverted state condition in single model `fit!` method.
#   - `_assemble_ls` should take `Basis` entities.
#   - Remove inverted state condition from the various `predict` methods.

# Todo:
#   - Need to make sure that the acquire_B! function used by ACE does not actually modify the
#     basis function. Otherwise there may be some issues with sharing basis functions.
#   - ACE should be modified so that `valtype` inherits from Base. This way there should be
#     no errors caused when importing it.
#   - Remove hard coded matrix type from the predict function.

function Ctran(l::Int64,m::Int64,μ::Int64)
   if abs(m) ≠ abs(μ)
      return 0
   elseif abs(m) == 0
      return 1
   elseif m > 0 && μ > 0
      return 1/sqrt(2)
   elseif m > 0 && μ < 0
      return (-1)^m/sqrt(2)
   elseif m < 0 && μ > 0
      return  - im * (-1)^m/sqrt(2)
   else
      return im/sqrt(2)
   end
end

Ctran(l::Int64) = sparse(Matrix{ComplexF64}([ Ctran(l,m,μ) for m = -l:l, μ = -l:l ]))

function _evaluate_real(Aval)
   L1,L2 = size(Aval[1])
   L1 = Int((L1-1)/2)
   L2 = Int((L2-1)/2)
   C1 = Ctran(L1)
   C2 = Ctran(L2)
   return real([ C1 * Aval[i].val * C2' for i = 1:length(Aval)])
end

# function _evaluate_real(Aval)
#     # This would be far more efficient if it were batch operable.
#     # The logic used here is rerun continuously which is highly wasteful.

#     n₁, n₂ = size(Aval[1])
#     ℓ₁, ℓ₂ = Int((n₁ - 1) / 2), Int((n₂ - 1) / 2)

#     # allocate Aval_real
#     Aval_real = [zeros(ComplexF64, n₁, n₂) for i = 1:length(Aval)]
#     # reconstruct real A
#     # TODO: I believe that there must exist a matrix form... But I will keep it for now...
#     for k=1:length(Aval)

#         A = Aval[k].val
#         for i=1:n₁, j=1:n₂
#             # Magnetic quantum numbers
#             m₁, m₂ = i - ℓ₁ - 1, j - ℓ₂ - 1
            
#             val = A[i,j]
            
#             if m₁ < 0
#                 val -= (-1)^(m₁) * A[end-i+1,j]
#             elseif m₁ > 0
#                 val += (-1)^(m₁) * A[end-i+1,j]
#             end

#             if (m₁ > 0 < m₂) || (m₁ < 0 > m₂)
#                 val += (-1)^(m₁ + m₂) * A[end-i+1, end-j+1]
#             elseif (m₁ > 0 > m₂) || (m₁ < 0 < m₂)
#                 val -= (-1)^(m₁ + m₂) * A[end-i+1, end-j+1]
#             end
#             if m₂ < 0
#                 val -= (-1)^(m₂) * A[i, end-j+1]
#             elseif m₂ > 0
#                 val += (-1)^(m₂) * A[i, end-j+1]
#             end

#             # This could be applied as a matrix to the full block
#             s = ((m₁ >= 0) && (m₂ < 0)) ? -1 : 1 # Account for sign flip until mess is sorted
#             plane = (m₁ >= 0) ⊻ (m₂ >= 0) ? im : 1
#             scale = m₁ == 0 ? (m₂ == 0 ? 1 : 1/√2) : (m₂ == 0 ? 1/√2 : 1/2)

#             Aval_real[k][i,j] = scale * plane * (s * (val))
#         end

#         # Prefactor "scale" could be applied here as a matrix operation
#     end
#     #return Aval_real
#     if norm(Aval_real - real(Aval_real))<1e-12
#         return real(Aval_real)
#     else
#         error("norm = $(norm(Aval_real - real(Aval_real))), please recheck...")
#     end
# end

"""
"""
function _assemble_ls(basis::SymmetricBasis, data::T, nonzero_mean::Bool=false) where T<:AbstractFittingDataSet
    # This will be rewritten once the other code has been refactored.

    # Should `A` not be constructed using `acquire_B!`?

    n₁, n₂, n₃ = size(data)
    # Currently the code desires "A" to be an X×Y matrix of Nᵢ×Nⱼ matrices, where X is
    # the number of sub-block samples, Y is equal to `size(bos.basis.A2Bmap)[1]`, and
    # Nᵢ×Nⱼ is the sub-block shape; i.e. 3×3 for pp interactions. This may be refactored
    # at a later data if this layout is not found to be strictly necessary.
    cfg = ACEConfig.(data.states)
    Aval = evaluate.(Ref(basis), cfg)
    A = permutedims(reduce(hcat, _evaluate_real.(Aval)), (2, 1))
    
    Y = [data.values[:, :, i] for i in 1:n₃]

    # Calculate the mean value x̄
    if nonzero_mean && n₁ ≡ n₂ && ison(data) 
        x̄ = mean(diag(mean(Y)))*I(n₁)
    else
        x̄ = zeros(n₁, n₂)
    end

    Y .-= Ref(x̄)
    return A, Y, x̄


end


###################
# Fitting Methods #
###################

"""
    fit!(basis, data;[ nonzero_mean])

Fits a specified model with the supplied data.

# Arguments
- `basis`: basis that is to be fitted.
- `data`: data that the basis is to be fitted to.
- `nonzero_mean::Bool`: setting this flag to true enables a non-zero mean to be
    used.
"""
function fit!(basis::T₁, data::T₂; nonzero_mean::Bool=false) where {T₁<:Basis, T₂<:AbstractFittingDataSet}
    # Lambda term should not be hardcoded to 1e-7!

    # Get the basis function's scaling factor (?)
    Γ = Diagonal(scaling(basis.basis, 2))

    # Setup the least squares problem
    Φ, Y, x̄ = _assemble_ls(basis.basis, data, nonzero_mean)
    
    # Assign the mean value to the basis set
    basis.mean .= x̄

    # Solve the least squares problem and get the coefficients
    basis.coefficients .= collect(solve_ls(Φ, Y, 1e-7, Γ, "LSQR"))

    # >>>>>>>>>>REMOVE UPON BASIS SYMMETRY ISSUE RESOLUTON>>>>>>>>>>
    if T₁<:AnisoBasis
        Γ = Diagonal(scaling(basis.basis_i, 2))
        Φ, Y, x̄ = _assemble_ls(basis.basis_i, data', nonzero_mean)
        basis.mean_i .= x̄
        basis.coefficients_i .= collect(solve_ls(Φ, Y, 1e-7, Γ, "LSQR"))
    end
    # <<<<<<<<<<REMOVE UPON BASIS SYMMETRY ISSUE RESOLUTON<<<<<<<<<<

    nothing
end


# Convenience function for appending data to a dictionary
function _append_data!(dict, key, value)
    if haskey(dict, key)
        dict[key] = dict[key] + value
    else
        dict[key] = value
    end
end


"""
    fit!(model, systems, target[; tolerance, filter_bonds, recenter])

Fits a specified model to the supplied data.

# Arguments
- `model::Model`: Model to be fitted.
- `systems::Vector{Group}`: HDF5 groups storing data with which the model should
    be fitted.
- `target::Symbol`: a symbol indicating which matrix should be fitted. This may be either
    `H` or `S`.
- `tolerance::AbstractFloat`: only sub-blocks where at least one value is greater than
    or equal to `tolerance` will be fitted. This argument permits spars blocks to be
    ignored.
- `filter_bonds::Bool`: Ignores interactions beyond the specified cutoff.
- `recenter::Bool`: Enabling this will re-wrap atomic coordinates to be consistent with
    the geometry layout used internally by FHI-aims. This should be used whenever loading
    real-space matrices generated by FHI-aims.

"""
function fit!(
    model::Model, systems::Vector{Group}, target::Symbol;
    tolerance::F=0.0, filter_bonds::Bool=true, recenter::Bool=false) where F<:AbstractFloat
    
    # Todo:
    #   - Check that the relevant data exists before trying to extract it; i.e. don't bother
    #     trying to gather carbon on-site data from an H2 system.
    #   - Currently the basis set definition is loaded from the first system under the
    #     assumption that it is constant across all systems. However, this will break down
    #     if different species are present in each system.
    #   - The approach currently taken limits io overhead by reducing redundant operations.
    #     However, this will likely use considerably more memory.

    # Section 1: Gather the data

    get_matrix = Dict(  # Select an appropriate function to load the target matrix
        :H=>load_hamiltonian, :S=>load_overlap,
        :Hg=>load_hamiltonian_gamma, :Sg=>load_overlap_gamma)[target]

    fitting_data = Dict{Basis, DataSet}()

    # Loop over the specified systems
    for system in systems

        # Load the required data from the database entry
        matrix = get_matrix(system), load_atoms(system; recentre=recentre)
        images = ndims(matrix) == 2 ? nothing : load_cell_translations(system)
        
        # Loop over the on site bases and collect the appropriate data
        for basis in values(model.on_site_bases)
            data_set = get_dataset(matrix, atoms, basis, model.basis_definition, images; tolerance=tolerance)
            _append_data!(fitting_data,basis, data_set)
        end 

        # Repeat for the off-site models
        for basis in values(model.off_site_bases)
            data_set = get_dataset(
                matrix, atoms, basis, model.basis_definition, images;
                tolerance=tolerance, filter_bonds=filter_bonds)
            
            # >>>>>>>>>>REMOVE UPON BASIS SYMMETRY ISSUE RESOLUTON>>>>>>>>>>
            # As ACE does not currently obey bond inversion symmetry the symmetrically
            # equivalent datasets must also be trained on. This is done intrinsically for
            # hetro-orbital interactions due to the use of dual models. However, the
            # "inverse" data must be added manually for homo-orbital interactions. 
            if (basis.id[1] == basis.id[2]) && (basis.id[3] == basis.id[4])
                data_set += data_set'
            end
            # <<<<<<<<<<REMOVE UPON BASIS SYMMETRY ISSUE RESOLUTON<<<<<<<<<<

            _append_data!(fitting_data, basis, data_set)

        end         
    end

    # Fit the on/off-site models
    for (basis, data_set) in fitting_data
        if length(data_set) ≡ 0
            @warn "Cannot fit $(basis.id): no matching data-points found (filters may be too aggressive)"
            continue
        elseif is_fitted(basis)
            @warn "Skipping $(basis.id): basis already fitted"
        else
            fit!(basis, data_set)
        end
        

    end

end



end
