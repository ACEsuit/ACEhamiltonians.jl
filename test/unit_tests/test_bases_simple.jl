using ACEhamiltonians, HDF5, Test, Artifacts


_subscript(i::Integer) = i<0 ? error("$i is negative") : join('₀'+d for d in reverse(digits(i)))
_l2s(ℓ::Integer) = ['s', 'p', 'd', 'f', 'g'][ℓ+1]


function build_contrived_model()
    # Define the basis definition
    basis_definition = Dict(14=>[0, 0, 0, 1, 1, 1, 2])

    # On site parameter deceleration
    on_site_parameters = OnSiteParaSet(
        # Maximum correlation order
        GlobalParams(2),
        # Maximum polynomial degree
        GlobalParams(12),
        # Environmental cutoff radius
        GlobalParams(10.),
        # Scaling factor "r₀"
        GlobalParams(2.5)
    )

    # Off-site parameter deceleration
    off_site_parameters = OffSiteParaSet(
        # Maximum correlation order
        GlobalParams(1),
        # Maximum polynomial degree
        GlobalParams(10),
        # Bond cutoff radius
        GlobalParams(8.0),
        # Environmental cutoff radius
        GlobalParams(4.),
    )

    # Model initialisation
    return Model(basis_definition, on_site_parameters, off_site_parameters, "H")
end


function _get_off_site_dict(model)
    # This just returns the model.off_site_submodels filed with with the
    # id key swapped out for a more visually intuitive string
    bases = Dict()

    basis_definition = model.basis_definition

    bases = Dict()
    for (id, basis) in model.off_site_submodels
        i , j  = id[end-1:end]
        ℓ₁, ℓ₂ = basis_definition[id[1]][i], basis_definition[id[2]][j]
        name = "$(_l2s(ℓ₁))$(_subscript(i))$(_l2s(ℓ₂))$(_subscript(j))"
        bases[name] = basis
    end

    return bases
end


function _get_on_site_dict(model)
    # This just returns the model.off_site_submodels filed with with the
    # id key swapped out for a more visually intuitive string
    bases = Dict()

    basis_definition = model.basis_definition

    bases = Dict()
    for (id, basis) in model.on_site_submodels
        i , j  = id[end-1:end]
        ℓ₁, ℓ₂ = basis_definition[id[1]][i], basis_definition[id[1]][j]
        name = "$(_l2s(ℓ₁))$(_subscript(i))$(_l2s(ℓ₂))$(_subscript(j))"
        bases[name] = basis
    end

    return bases
end



"""
Check if the reference and predicted data agree upon whether the sub-blocks are fully
symmetric; i.e. AB = BAᵗ. 
"""
function fully_symmetric(ABᵣ, BAᵣ, ABₚ, BAₚ; tol=1E-6)

    # Only valid when looking at square sub-blocks
    if size(ABᵣ) == size(BAᵣ)
        return isapprox(ABᵣ, BAᵣ; atol=tol) == isapprox(ABₚ, BAₚ; atol=tol)
    else
        return true
    end
end


"""
Check if the reference and predicted data agree upon whether the sub-blocks are self
symmetric; i.e. AB = ABᵗ. 
"""
function self_symmetric(ABᵣ, BAᵣ, ABₚ, BAₚ; tol=1E-6)
    ℓ₁, ℓ₂ = (size(ABᵣ) .-1) .÷ 2

    # Only valid when looking at square sub-blocks
    if ℓ₁ == ℓ₂
        a, b = isapprox(ABᵣ, ABᵣ'; atol=tol), isapprox(BAᵣ, BAᵣ'; atol=tol)
        c, d = isapprox(ABₚ, ABₚ'; atol=tol), isapprox(BAₚ, BAₚ'; atol=tol)
        return a == b == c == d
    else
        return true
    end
end


"""
Check if the reference and predicted data agree upon whether the sub-blocks are transpose
symmetric; i.e. AB = ±BAᵗ. 
"""
function transpose_symmetric(ABᵣ, BAᵣ, ABₚ, BAₚ; tol=1E-6)
    parity = isodd(sum((size(ABᵣ) .-1) .÷ 2)) ? -1 : 1
    return isapprox(ABᵣ, parity * BAᵣ'; atol=tol) == isapprox(ABₚ, parity * BAₚ'; atol=tol)
end


function _check_contrived_system(bases, basis_definition, database_path, system; tol=1E-6)

    # Load Hamiltonian matrix, atoms object, and cell translation vectors of the
    # target group from the specified database.
    H, atoms, images = h5open(database_path) do database
        target = database[system]
        load_hamiltonian(target), load_atoms(target; recentre=true), load_cell_translations(target)
    end
    
    # Iterate over each of the supplied bases
    for (name, basis) in bases
        @testset "$name" begin
            
            # Gather the relevant dataset, but only get the AB and BA states where
            # A & B are the 1ˢᵗ & 2ⁿᵈ atoms respectively 
            dataset = get_dataset(
                H, atoms, basis, basis_definition, images; focus=[1 2; 2 1], no_reduce=true)

            # Extract the reference sub-blocks from the recently gathered dataset
            ABᵣ = dataset.values[:, :, 1]
            BAᵣ = dataset.values[:, :, 2]'
        
            # Fit the current basis, but only to the AB state
            fit!(basis, dataset[1:1])

            # Make predictions for the AB and BA sub-blocks
            ABₚ = predict(basis, dataset.states[1])
            BAₚ = predict(basis, dataset.states[2])'
        
            # Basis should accurately reproduce the one state it was fitted to
            @testset "Tolerance AB" begin
                @test ABᵣ ≈ ABₚ atol=tol
            end
            
            # It should also accurately reproduce the symmetrically equivalent state
            @testset "Tolerance BA" begin
                @test BAᵣ ≈ BAₚ atol=tol
            end

            # Check if the reference and predicted data agree on whether the two states are
            # fully symmetric (AB = BAᵗ)
            @testset "Fully Symmetric" begin
                @test fully_symmetric(ABᵣ, BAᵣ, ABₚ, BAₚ)
            end

            # Check if the reference and predicted data agree on whether the states are self
            # symmetric (AB = ABᵗ), if applicable
            @testset "Self Symmetric" begin
                @test self_symmetric(ABᵣ, BAᵣ, ABₚ, BAₚ)
            end

            # Check if the reference and predicted data agree on whether the data is transpose
            # symmetric (AB = ±BAᵗ)
            @testset "Transpose Symmetric" begin
                @test transpose_symmetric(ABᵣ, BAᵣ, ABₚ, BAₚ)
            end
        end
    end
end



function _check_contrived_system_1(bases, basis_definition, database_path; tol=1E-6)
    # Contrived System 1:
    #     This bond system contains no environmental atoms and should always
    #     be free from symmetry issues. This system is provided as an example
    #     of a known good true positive.
    #
    #
    #
    #   A ------- B
    #
    #
    #
    # FHI-aims geometry.in file contents:
    # lattice_vector 20.0000000000000000 0.0000000000000000 0.0000000000000000 
    # lattice_vector 0.0000000000000000 20.0000000000000000 0.0000000000000000 
    # lattice_vector 0.0000000000000000 0.0000000000000000 20.0000000000000000 
    # atom 2.5355339059327378 5.6568542494923797 4.5355339059327378 Si
    # atom 4.5355339059327378 8.4852813742385695 2.5355339059327378 Si
    @testset "Contrived System 1" begin
        _check_contrived_system(bases, basis_definition, database_path, "System_1"; tol=tol)
    end
end


function _check_contrived_system_2(bases, basis_definition, database_path; tol=1E-6)
    # Contrived System 2:
    #     This system contains two symmetrical equivalent bonding atoms and
    #     a pair of environmental atoms located in the plane passing through
    #     the bond's midpoint but is orthogonal to the bond vector. The highly
    #     symmetric nature of this system helps to highlight any serious bond
    #     symmetry related issues that may occur in the code.
    # 
    #     ---C---
    #    /       \\
    #   A ------- B 
    #    \\       /
    #     ---D---
    #
    # FHI-aims geometry.in file contents:
    # lattice_vector 20.0000000000000000 0.0000000000000000 0.0000000000000000 
    # lattice_vector 0.0000000000000000 20.0000000000000000 0.0000000000000000 
    # lattice_vector 0.0000000000000000 0.0000000000000000 20.0000000000000000 
    # atom 2.5355339059327378 5.6568542494923797 4.5355339059327378 Si
    # atom 4.5355339059327378 8.4852813742385695 2.5355339059327378 Si
    # atom 4.0355339059327378 6.3639610306789276 3.0355339059327378 Si
    # atom 3.0355339059327378 7.7781745930520234 4.0355339059327378 Si
    @testset "Contrived System 2" begin
        _check_contrived_system(bases, basis_definition, database_path, "System_2"; tol=tol)
    end
end


function _check_contrived_system_3(bases, basis_definition, database_path; tol=1E-6)
    # Contrived System 3:
    #     This system contains an atom which lies exactly at the midpoint between
    #     the two bonding atoms and is commonly subject to symmetry issues.
    #      
    #
    #
    #   A ---C--- B 
    #
    #
    #
    # FHI-aims geometry.in file contents:
    # lattice_vector 20.0000000000000000 0.0000000000000000 0.0000000000000000 
    # lattice_vector 0.0000000000000000 20.0000000000000000 0.0000000000000000 
    # lattice_vector 0.0000000000000000 0.0000000000000000 20.0000000000000000 
    # atom 2.5355339059327378 5.6568542494923797 4.5355339059327378 Si
    # atom 4.5355339059327378 8.4852813742385695 2.5355339059327378 Si
    # atom 3.5355339059327378 7.0710678118654755 3.5355339059327373 Si
    @testset "Contrived System 3" begin
        _check_contrived_system(bases, basis_definition, database_path, "System_3"; tol=tol)
    end
end


function _check_contrived_system_4(bases, basis_definition, database_path; tol=1E-6)
    # Contrived System 4:
    #     This asymmetric system is included to act as a true negative reference
    #     which helps to verify that the tests are working. 
    #  
    #
    #
    #   A ------- B 
    #    \\       /
    #     ---C---
    #
    # FHI-aims geometry.in file contents:
    # lattice_vector 20.0000000000000000 0.0000000000000000 0.0000000000000000 
    # lattice_vector 0.0000000000000000 20.0000000000000000 0.0000000000000000 
    # lattice_vector 0.0000000000000000 0.0000000000000000 20.0000000000000000 
    # atom 2.5355339059327378 5.6568542494923797 4.5355339059327378 Si
    # atom 4.5355339059327378 8.4852813742385695 2.5355339059327378 Si
    # atom 4.0355339059327378 6.3639610306789276 3.0355339059327378 Si
    @testset "Contrived System 4" begin
        _check_contrived_system(bases, basis_definition, database_path, "System_4"; tol=tol)
    end
end


function _check_on_site(bases, basis_definition, database_path; tol=1E-6)

    # Load Hamiltonian matrix, atoms object, and cell translation vectors of the
    # target group from the specified database.
    H, atoms, images = h5open(database_path) do database
        target = database["System_2"]
        load_hamiltonian(target), load_atoms(target; recentre=true), load_cell_translations(target)
    end
    
    # Iterate over each of the supplied bases
    for (name, basis) in bases
        @testset "$name" begin
            
            # Gather the relevant dataset, but only get the sub-block for the first atom
            dataset = get_dataset(
                H, atoms, basis, basis_definition, images; focus=[1])

  
            # Fit the current basis
            fit!(basis, dataset)

            # Make predictions for the on-site sub-block
            A = predict(basis, dataset.states[1])
            
            # Basis should accurately reproduce the one state it was fitted to
            @test A ≈ dataset.values[:, :, 1] atol=tol

        end
    end
end


@testset "Foundational Basis Checks" begin

    off_site_tol = 5E-6
    on_site_tol = 1E-5

    # This tests do not offer comprehensive coverage, but instead are intended to act
    # as simple tests to detect that the bases are able to perform the most basic of
    # prediction tasks under optimal conditions.

    database_path = joinpath(artifact"test_data", "test_data/contrived_Si_symmetry_testing_systems.h5")



    # Build the model used to run tests on the contrived systems
    model = build_contrived_model()

    basis_definition = model.basis_definition

    @testset "Off-site Symmetry and Contrived Tolerance Check" begin
        # This function will perform a series of tolerance and symmetry checks for each
        # off-site basis on a collection of small contrived systems. This is to ensure
        # that predicted data exhibits the same symmetry properties, if any, as the
        # reference data. Furthermore, this acts as a sanity check for the basis as
        # each is fitted to only a single data-point, and thus should be able to attain
        # a high degree of accuracy.
    
        # Extract the off site basis dictionary and swap out the key for a more
        # visually meaningful string to help with debugging.
        off_site_submodels = _get_off_site_dict(model)
    
        
        _check_contrived_system_1(off_site_submodels, basis_definition, database_path; tol=off_site_tol)
        _check_contrived_system_2(off_site_submodels, basis_definition, database_path; tol=off_site_tol)
        _check_contrived_system_3(off_site_submodels, basis_definition, database_path; tol=off_site_tol)
        _check_contrived_system_4(off_site_submodels, basis_definition, database_path; tol=off_site_tol)
    
    end

    @testset "On-site Contrived Tolerance Check" begin
        # No symmetry tests are required here thus only a tolerance check is
        # performed on one of the contrived systems.
        on_site_submodels = _get_on_site_dict(model)

        _check_on_site(on_site_submodels, basis_definition, database_path; tol=on_site_tol)
        
    end

end
