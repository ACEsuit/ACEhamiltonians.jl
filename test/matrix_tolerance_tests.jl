using HDF5, ACEhamiltonians, JuLIP, ACE
using ACEhamiltonians.Fitting: assemble_ls, solve_ls, basis2tikmat, evaluateval_real, params2wmodels
using ACEhamiltonians.Dictionary: recover_at
using ACEhamiltonians.Predict: predict_onsite_HS, predict_offsite_HS, wmodels2err
using ACE: evaluate
using Test
using Artifacts


# Helper Functions
"""
    load_block_from_file(path, block, n_orbs)

Load the specified atom-atom block of a Hamiltonian matrix stored in the
provided HDF5 database.

# Arguments
- `path::String`: path to the HDF5 database.
- `block::Vector{Int64}`: row and column of the target block.
- `n_orbs::Int64`: number of orbitals per atom.

# Returns
- `H_block::Matrix{Float64}` requested block of the Hamiltonian matrix.

# Warnings
The behaviour and call signature of this function will be changed.
"""
function load_block_from_file(path, block, n_orbs)
    H = h5open(path) do h5file
        read(h5file, "aitb/H")
    end
    i, j = ((block.-1).*n_orbs).+1
    return H[i:i+n_orbs-1, j:j+n_orbs-1]
end

"""
    flatten_reference(H_osb)

Flatten on-site reference data in manner consistent with the results produced
by `predict_on_sites`.

# Arguments
- `H_osb::Vector{Matrix{StaticArrays.SMatrix{Float64}}}`: the on-site sub-blocks
    of the Hamiltonian matrix. Note that all sub-blocks must be of the same type.
    
# Returns
- `H_osb_flat:Vector{Float64}`: the input data flattened into a single vector.
"""
function flatten_reference(H_osb)
    r_values = Vector{Float64}([])
    n_shells_i, n_shells_j = size(H_osb[1])
    for i=1:n_shells_i, j=1:n_shells_j
        append!(r_values, H_osb[1][i, j])
    end
    return r_values
end


"""
predict_values(H_osb, Rs, [on_site=true])

This function this takes on/off-site sub-blocks as its inputs. Creates and fits an
appropriate on/off-site model to this data. Then uses the resulting model to predict the
original on/off-site sub-blocks. The difference between the input and output data is the
training error.

# Arguments
- `H_osb::Vector{Matrix{StaticArrays.SMatrix{Float64}}}`: the on-site sub-blocks
    of the Hamiltonian matrix for which a model is to be created and used. Note
    that all sub-blocks must be of the same type, e.g. ss, sp, pp, etc. 
- `Rs::Vector{Vector{PositionState{Float64}}}`: the set of all atomic positions.
- `on_site::Bool`: indicates if the model is for on or off-site data.

# Returns
- `H_osb_predicted:Vector{Float64}` the on-site sub-blocks of the Hamiltonian matrix
    as predicted by the model as a single flattened vector.

# Warnings
This function is subject to change and will be non-trivially refactored at a
later date. 

"""
function predict_values(H_osb, Rs, on_site=True)
    # Identify the azimuthal numbers associated with this interaction
    ℓ₁, ℓ₂ = Int.((size(H_osb[1][1]) .- 1) ./ 2)

    # Create the initial basis entity.
    if on_site
        basis = OnsiteBasis(20.0, 5, 2, ℓ₁, ℓ₂; species = [:Al, ])
    else
        basis = OffsiteBasis(20.0, 5, 2, ℓ₁, ℓ₂; species = [:Al, ])
    end

    # Calculate the Tychonov regularisation term as per equation
    Γ = basis2tikmat(basis)

    values = Vector{Float64}([])

    # Loop over each possible pair of associate shells. For a valence-only minimal
    # basis set there would only be one i,j pair. 
    n_shells_i, n_shells_j = size(H_osb[1])
    for i=1:n_shells_i, j=1:n_shells_j
        # Setup and solve the initial linear system to get the coefficients.
        Φ, Y, Mean = ACEhamiltonians.Fitting.assemble_ls(basis, H_osb, Rs, i, j, "on")
        coefficients = solve_ls(Φ, Y, 1e-7, Γ, "LSQR")

        # Build and train the model
        model = TBModel(basis.basis, coefficients, Mean)

        for Rs_atom in Rs
            # Create a descriptor for the chemical environment around the atom.
            cfg = ACEConfig(Rs_atom)

            # Construct and evaluate the on-site basis
            A = evaluate(model.basis, cfg)
            B = evaluateval_real(A)
            append!(values, (model.coeffs' * B) + model.mean)
    
        end
    end
    
    return values
end

# Tests
@testset "Matrix value tolerance tests" begin
    path = joinpath(artifact"test_data", "Al3_1.h5")
    _, atoms = get_atoms(path)

    # Collect the onsite H & S data, and the atomic positions
    @testset "On-site tolerance tests" begin
        on_site_block_names = ["ss", "sp", "sd", "pp", "pd", "dd"]

        # A structure that holds the file names & indices (used during fitting).
        data_info = Data([path,], [1,])

        r_cut = repeat([20.0], 6)
        max_deg = repeat([8], 6)
        order = repeat([2], 6)
        λ = repeat([1e-7], 6)
        reg_type = 2
        parameters = Params(r_cut, max_deg, order, λ, reg_type, "LSQR")


        # Manually construct, fit & evaluate each on-site sub-block individually. This test
        # is intended evaluate the performance of the models directly without the need for
        # higher level helper functions.
        @testset "Isolated on-site block test" begin
            data = data_read([path,], [1,])
            for (block, name) in zip(data, on_site_block_names)
                # Collect the on-site sub-blocks of the Hamiltonian matrix & the
                # positions of the neighbouring atoms.
                H_osb, Rs = block[1], block[3]
                
                @show name

                # Hack to deal with type inconsistency issues
                if name == "dd" 
                    H_osb = reshape.(block[1], 1, 1)
                end

                # Construct the on-site model(s) for the current sub-block, fit it using
                # the 1'st the atomic block only then use it to re-predict said data.
                predicted = predict_values(H_osb, Rs, true)
                # Gather up the reference data
                reference = flatten_reference(H_osb)

                # Check that the predicted & reference values are within tolerance
                @testset "$name-block" begin
                    @test isapprox(predicted, reference, atol=1e-8, rtol=1e-5)
                end
            end
        end

        MWH, MWS, data_whole = params2wmodels(data_info, parameters, parameters)
        # Check that the off-site block values predicted by the predict_offsite_HS method are
        # within tolerance.
        @testset "Batch on-site block test" begin
            # MWH, MWS, data_whole = params2wmodels(data_info, parameters, parameters)
            @test isapprox(predict_onsite_HS(atoms, MWH, [1,]),
                            load_block_from_file(path, [1, 1], 14),
                            atol=1e-8, rtol=1e-5)
        end
 
        # Ensure the off-site values produced by the wmodels2err method are within tolerance.
        @testset "Full on-site block test" begin
            error = wmodels2err(MWH, MWS, data_info, data_whole)
            for (e, name) in zip(error[1][1], on_site_block_names)
                @testset "$name-block" begin
                    @test all(e .< 1E-6)
                end
            end
        end
    end


    @testset "Off-site tolerance tests" begin
        off_site_block_names = ["ss", "sp", "sd", "ps", "pp", "pd", "ds", "dp", "dd"]

        # A structure that holds the file names & indices (used during fitting).
        data_info = Data([path], [(1, 2)])

        r_cut = repeat([20.0], 9)
        max_deg = repeat([8], 9)
        order = repeat([2], 9)
        λ = repeat([1e-7], 9)
        reg_type = 2
        parameters = Params(r_cut, max_deg, order, λ, reg_type, "LSQR")

        # Manually construct, fit & evaluate each off-site sub-block individually. This test
        # is intended evaluate the performance of the models directly without the need for
        # higher level helper functions. 
        @testset "Isolated off-site block test" begin
            # Collect the offsite H & S data, and the atomic positions
            data = data_read([path,], [(1,2)])

            for (block, name) in zip(data, off_site_block_names)
                H_block, Rs = block[1], block[3]
                
                if name == "dd" # Hack to deal with type inconsistency issues
                    H_block = reshape.(H_block, 1, 1)
                end

                @testset "$name-block" begin
                    @test isapprox(predict_values(H_block, Rs, false),
                                flatten_reference(H_block),
                                atol=1e-8, rtol=1e-5)
                end
            end
        end

        MWH, MWS, data_whole = params2wmodels(data_info, parameters, parameters)
        # Check that the off-site block values predicted by the predict_offsite_HS method are
        # within tolerance.
        # @testset "batch_test_off_sites" begin  
        #     @test isapprox(predict_offsite_HS(atoms, MWH, [(1,2)])[:, :, 1],
        #                     load_block_from_file(path, [1, 2], 14),
        #                     atol=1e-8, rtol=1e-5)
        # end

        # Ensure the off-site values produced by the wmodels2err method are within tolerance.
        @testset "Full off-site block test" begin
            error = wmodels2err(MWH, MWS, data_info, data_whole)
            for (e, name) in zip(error[1][1], off_site_block_names)
                @testset "$name-block" begin
                    @test all(e .< 1E-7)
                end
            end
        end
    end
end

# Todo:
#   - Abstract the on and off-site code into a single unified function.
#   - Abstract IO operations.
#   - This takes far too long to run; ~7 min to compute a single atom-atom block of a
#     three atom system. The performance bottleneck must be identified and purged. 
