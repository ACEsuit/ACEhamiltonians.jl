using ACEhamiltonians, Test, LinearAlgebra
using StaticArrays: SVector
using JuLIP: Atoms
using ACE: CylindricalBondEnvelope

using ACEhamiltonians.States:
    _guard_position, _locate_minimum_image, _locate_target_image

function _error_free(expr::Expr)
    try
        eval(expr)
        return true
    catch
        return false
    end
end

@testset "States" begin
    t = SVector{3, Float64}

    # Just ensure that state entities can be instantiated manually without error
    @testset "Sanity Check" begin
        pos = t([0., 0., 0.])
        @test _error_free(:(AtomState($pos)))
        @test _error_free(:(BondState($pos, $pos, false)))
    end

    @testset "General Functionality" begin
        @testset "AtomStates" begin
            @test zero(AtomState{t}) == zero(AtomState{t}(rand(3))) == AtomState{t}(zero(t))

            @test AtomState(t([1, 1, 1])) == AtomState(t([1, 1, 1]))
            @test AtomState(t([1, 1, 1])) != AtomState(t([1, 1, 1.00000001]))

            @test AtomState(t([1, 1, 1])) ≈ AtomState(t([1, 1, 1.00000001]))

            @test AtomState(t([1, 1, 1])) == reflect(AtomState(t([1, 1, 1])))

            @test ison(AtomState(t([1, 1, 1])))

        end

        @testset "BondStates" begin
            a = BondState(ones(t), ones(t), false)
            b = BondState(t([1, 1, 1.00000001]), ones(t), false)
            c = BondState(ones(t), ones(t), true)
        
            @test zero(typeof(a)) == zero(a) == BondState{t, Bool}(zero(t), zero(t), false)
            
            @test a == BondState(ones(t), ones(t), false)
            @test a != b
            @test a != c

            @test a ≈ b
            @test !(a ≈ c)

            @test reflect(a) == BondState(ones(t), -ones(t), false)
            @test reflect(c) == BondState(-ones(t), -ones(t), true)

            @test !ison(a)
        end
    end

    @testset "Factory Helper Functions" begin
        # Note that no tests are run on the `_neighbours` method as is just calls functions
        # form JuLIP and NeighbourLists.

        @testset "_guard_position" begin
            rr0 = t([1., 1., 1.])
            @test _guard_position(ones(t), rr0, 1, 2) == ones(t)
            @test abs(norm(_guard_position(zero(t), rr0, 1, 2; cutoff=0.05)) - 0.05) < 1E-5
            @test abs(norm(_guard_position(t([0.001, 0.001, 0.001]), rr0, 1, 2; cutoff=0.05)) - 0.05) < 1E-5
        end

        @testset "_locate_minimum_image" begin
            idxs = [1, 2, 1, 2, 1, 2]
            vecs = [ones(3), ones(3), ones(3), ones(3) * 0.5, ones(3) * 0.4, ones(3)]
            @test _locate_minimum_image(2, idxs, vecs) == 4
            @test _locate_minimum_image(1, idxs, vecs) == 5
        end

        @testset "_locate_target_image" begin
            # _locate_target_image(j, idxs, images, image)
            idxs = [1, 2, 1, 2, 1, 2]
            images = [[0,0,0],[0,0,1],[0,0,2],[0,0,3],[0,0,4],[0,0,5]]

            @test _locate_target_image(1, idxs, images, [0,0,0]) == 1
            @test _locate_target_image(1, idxs, images, [0,0,2]) == 3
            @test _locate_target_image(2, idxs, images, [0,0,3]) == 4
            @test _locate_target_image(1, idxs, images, [0,0,100]) == 0

        end

    end

    # The following test-set is not fully comprehensive, however it should, together with
    # the previous tests, ensure that most serious errors are caught.
    @testset "get_state" begin
        cell = [10. 0 0; 0 10 0; 0 0 10]
        a = [1., 1, 1]
        b = normalize([1, 1, 1]) .+ 1
        @testset "atom states" begin
            atoms = Atoms(;Z=[1, 1], X=[a, b], cell=cell, pbc=true)
            @test length(get_state(1, atoms; r=10.)) == 4
            @test length(get_state(1, atoms; r=5.)) == 1
            @test length(get_state(1, atoms; r=.5)) == 0
            @test norm(get_state(1, atoms; r=5.)[1].rr) == 1.0
        end

        @testset "bond states" begin
            env = CylindricalBondEnvelope(10.0, 5.0, 5.0, floppy=false, λ=0.0)
            midpoint = 0.5(b - a) + a
            c = midpoint + normalize(rand(3)) * 0.3
            atoms = Atoms(;Z=[1, 1, 1], X=[a, b, c], cell=cell, pbc=true)
            bond_state, env_state = get_state(1, 2, atoms, env)

            # Ensure the rr0 and bond.rr values are correct
            @test bond_state.rr0 == env_state.rr0 == 2bond_state.rr

            # Check that environmental positions are relative to the bond's midpoint
            @test env_state.rr == c-midpoint

            # Ensure that the code can can locate a target cell when requested. 
            @test get_state(1, 2, atoms, env, [0, 0, 0]) == get_state(1, 2, atoms, env)
            
            # and can manually build it if it is not present in the neighbour list.
            @test length(get_state(1, 2, atoms, env, [0, 0, -100])) == 1
        
        end
    end
    
end
