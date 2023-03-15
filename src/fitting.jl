module Fitting2
using HDF5, ACE, ACEbase, ACEhamiltonians, StaticArrays, Statistics, LinearAlgebra, SparseArrays
using HDF5: Group
using JuLIP: Atoms
using ACE: ACEConfig, evaluate, scaling, AbstractState, SymmetricBasis
using ACEhamiltonians.Fitting: evaluateval_real, solve_ls
using ACEhamiltonians.Common: number_of_orbitals
using ACEhamiltonians.Bases: envelope
using ACEhamiltonians.DatabaseIO: load_hamiltonian_gamma, load_overlap_gamma

using ACEhamiltonians: DUAL_BASIS_MODEL

export fit!

# Once the bond inversion issue has been resolved the the redundant models will no longer
# be required. The changes needed to be made in this file to remove the redundant model
# are as follows:
#   - Remove inverted state condition in single model `fit!` method.
#   - `_assemble_ls` should take `AHBasis` entities.
#   - Remove inverted state condition from the various `predict` methods.

# Todo:
#   - Need to make sure that the acquire_B! function used by ACE does not actually modify the
#     basis function. Otherwise there may be some issues with sharing basis functions.
#   - ACE should be modified so that `valtype` inherits from Base. This way there should be
#     no errors caused when importing it.
#   - Remove hard coded matrix type from the predict function.

function _ctran(l::Int64,m::Int64,μ::Int64)
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

_ctran(l::Int64) = sparse(Matrix{ComplexF64}([ _ctran(l,m,μ) for m = -l:l, μ = -l:l ]))

function _evaluate_real(Aval)
   L1,L2 = size(Aval[1])
   L1 = Int((L1-1)/2)
   L2 = Int((L2-1)/2)
   C1 = _ctran(L1)
   C2 = _ctran(L2)
   return real([ C1 * Aval[i].val * C2' for i = 1:length(Aval)])
end

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
- `λ::AbstractFloat`: regularisation term to be used (default=1E-7).
- `solver::String`: solver to be used (default="LSQR")
"""
function fit!(basis::T₁, data::T₂; nonzero_mean::Bool=false, λ=1E-7, solver="LSQR") where {T₁<:AHBasis, T₂<:AbstractFittingDataSet}

    # Get the basis function's scaling factor
    Γ = Diagonal(scaling(basis.basis, 2))

    # Setup the least squares problem
    Φ, Y, x̄ = _assemble_ls(basis.basis, data, nonzero_mean)
    
    # Assign the mean value to the basis set
    basis.mean .= x̄

    # Solve the least squares problem and get the coefficients

    basis.coefficients .= collect(solve_ls(Φ, Y, λ, Γ, solver))

    @static if DUAL_BASIS_MODEL
        if T₁<:AnisoBasis
            Γ = Diagonal(scaling(basis.basis_i, 2))
            Φ, Y, x̄ = _assemble_ls(basis.basis_i, data', nonzero_mean)
            basis.mean_i .= x̄
            basis.coefficients_i .= collect(solve_ls(Φ, Y, λ, Γ, solver))
        end
    end

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
- `tolerance::AbstractFloat`: only sub-blocks where at least one value is greater than
    or equal to `tolerance` will be fitted. This argument permits spars blocks to be
    ignored.
- `filter_bonds::Bool`: Ignores interactions beyond the specified cutoff.
- `recentre::Bool`: Enabling this will re-wrap atomic coordinates to be consistent with
    the geometry layout used internally by FHI-aims. This should be used whenever loading
    real-space matrices generated by FHI-aims.
- `refit::Bool`: By default already fitted bases will not be refitted, but this behaviour
    can be suppressed by setting `refit=true`.
- `target::String`: a string indicating which matrix should be fitted. This may be either
    `H` or `S`. If unspecified then the model's `.label` field will be read and used. 
"""
function fit!(
    model::Model, systems::Vector{Group}; tolerance::Union{F, Nothing}=nothing,
    filter_bonds::Bool=true, recentre::Bool=false, 
    target::Union{String, Nothing}=nothing, refit::Bool=false) where F<:AbstractFloat
    
    # Todo:
    #   - Modify so that redundant data is not extracted; i.e. both A[0,0,0] -> A[1,0,0] and
    #     A[0,0,0] -> A[-1,0,0]
    #   - Check that the relevant data exists before trying to extract it; i.e. don't bother
    #     trying to gather carbon on-site data from an H2 system.
    #   - Currently the basis set definition is loaded from the first system under the
    #     assumption that it is constant across all systems. However, this will break down
    #     if different species are present in each system.
    #   - The approach currently taken limits io overhead by reducing redundant operations.
    #     However, this will likely use considerably more memory.
    #   - Don't fit on-site bases for overlap matrix models.

    # Section 1: Gather the data

    # If no target has been specified; then default to that given by the model's label.
    
    target = isnothing(target) ? model.label : target

    get_matrix = Dict(  # Select an appropriate function to load the target matrix
        "H"=>load_hamiltonian, "S"=>load_overlap,
        "Hg"=>load_hamiltonian_gamma, "Sg"=>load_overlap_gamma)[target]

    fitting_data = Dict{Any, DataSet}()

    # Loop over the specified systems
    for system in systems

        # Load the required data from the database entry
        matrix, atoms = get_matrix(system), load_atoms(system; recentre=recentre)
        images = ndims(matrix) == 2 ? nothing : load_cell_translations(system)
        
        # Loop over the on site bases and collect the appropriate data
        for basis in values(model.on_site_bases)
            data_set = get_dataset(matrix, atoms, basis, model.basis_definition, images; tolerance=tolerance)
            _append_data!(fitting_data, basis.id, data_set)
        end 

        # Repeat for the off-site models
        for basis in values(model.off_site_bases)
            data_set = get_dataset(
                matrix, atoms, basis, model.basis_definition, images;
                tolerance=tolerance, filter_bonds=filter_bonds)

            _append_data!(fitting_data, basis.id, data_set)

        end         
    end

    # Fit the on/off-site models
    fit!(model, fitting_data; refit=refit)
    
end


"""
    fit!(model, fitting_data[; refit])


Fit the specified model using the provided data.

# Arguments
- `model::Model`: the model that should be fitted.
- `fitting_data`: dictionary providing the data to which the supplied model should be
  fitted. This should hold one entry for each basis that is to be fitted and should take
  the form `{Basis.id, DataSet}`.
- `refit::Bool`: By default, already fitted bases will not be refitted, but this behaviour
  can be suppressed by setting `refit=true`.
"""
function fit!(
    model::Model, fitting_data; refit::Bool=false)

    @info "Fitting off site bases:"
    for (id, basis) in model.off_site_bases
        if !haskey(fitting_data, id)
            @info "Skipping $(id): no fitting data provided"
        elseif is_fitted(basis) && !refit
            @info "Skipping $(id): basis already fitted"
        elseif length(fitting_data) ≡ 0
            @info "Skipping $(id): fitting dataset is empty"
        else
            @info "Fitting $(id): using $(length(fitting_data[id])) fitting points"
            fit!(basis, fitting_data[id])
        end    
    end

    @info "Fitting on site bases:"
    for (id, basis) in model.on_site_bases
        if !haskey(fitting_data, id)
            @info "Skipping $(id): no fitting data provided"
        elseif is_fitted(basis) && !refit
            @info "Skipping $(id): basis already fitted"
        elseif length(fitting_data) ≡ 0
            @info "Skipping $(id): fitting dataset is empty"
        else
            @info "Fitting $(id): using $(length(fitting_data[id])) fitting points"
            fit!(basis, fitting_data[id]; nonzero_mean=ison(basis))
        end    
    end
end


# The following code was added to `fitting.jl` to allow data to be fitted on databases
# structured using the original database format.
using ACEhamiltonians.DatabaseIO: _load_old_atoms, _load_old_hamiltonian, _load_old_overlap
using Serialization

function old_fit!(
    model::Model, systems, target::Symbol;
    tolerance::F=0.0, filter_bonds::Bool=true, recentre::Bool=false,
    refit::Bool=false) where F<:AbstractFloat
    
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
        :H=>_load_old_hamiltonian, :S=>_load_old_overlap)[target]

    fitting_data = IdDict{AHBasis, DataSet}()

    # Loop over the specified systems
    for (database_path, index_data) in systems
        
        # Load the required data from the database entry
        matrix, atoms = get_matrix(database_path), _load_old_atoms(database_path)

        println("Loading: $database_path")

        # Loop over the on site bases and collect the appropriate data
        if haskey(index_data, "atomic_indices")
            println("Gathering on-site data:")
            for basis in values(model.on_site_bases)
                println("\t- $basis")
                data_set = get_dataset(
                    matrix, atoms, basis, model.basis_definition;
                    tolerance=tolerance, focus=index_data["atomic_indices"])
                _append_data!(fitting_data, basis, data_set)
  
            end
            println("Finished gathering on-site data")
        end

        # Repeat for the off-site models
        if haskey(index_data, "atom_block_indices")
            println("Gathering off-site data:")
            for basis in values(model.off_site_bases)
                println("\t- $basis")
                data_set = get_dataset(
                    matrix, atoms, basis, model.basis_definition;
                    tolerance=tolerance, filter_bonds=filter_bonds, focus=index_data["atom_block_indices"])
                _append_data!(fitting_data, basis, data_set)
            end
            println("Finished gathering off-site data")
        end
    end

    # Fit the on/off-site models
    fit!(model, fitting_data; refit=refit) 

end

end
