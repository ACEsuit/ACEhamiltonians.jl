using ACEhamiltonians, Test
using JuLIP: Atoms
using StaticArrays: SVector

function _error_free(expr::Expr)
    try
        eval(expr)
        return true
    catch
        return false
    end
end

@testset "DataSets" begin
    t = SVector{3, Float64}
    # Sanity check
    @test _error_free(:(DataSet(rand(3,3,3), [1 2; 3 4], [[AtomState($t(rand(3)))]])))


    @testset "General Functionality" begin
        values = rand(ComplexF64, 5, 5, 8)
        blk_idxs = rand(1:10, 3, 8)
        states = [[AtomState(t(rand(3)))] for _=1:8]
        
        dataset_a = DataSet(values, blk_idxs, states)
        dataset_b = DataSet(values[:, :, 1:4], blk_idxs[:, 1:4], states[1:4])
        dataset_c = DataSet(values[:, :, 5:8], blk_idxs[:, 5:8], states[5:8])
        dataset_d = dataset_a'
        dataset_e = DataSet(values, blk_idxs, [[BondState(t(rand(3)), t(rand(3)), false)] for _=1:8])
        
        # Check the equality operator
        @test dataset_a == dataset_a
        @test dataset_a == DataSet(values, blk_idxs, states)
        @test dataset_b != dataset_c

        # Check that datasets can be combined
        @test dataset_b + dataset_c == dataset_a == sum([dataset_b, dataset_c])

        # Ensure they can be indexed
        @test dataset_a[1:4] == dataset_b
        @test dataset_a[5:8] == dataset_c
        @test dataset_a[1:end] == dataset_a

        # Test the adjoint operator
        @test dataset_d.values == conj(permutedims(values, (2, 1, 3)))
        @test (
            dataset_d.blk_idxs[1, :] == blk_idxs[2, :]
            && dataset_d.blk_idxs[2, :] == blk_idxs[1, :]
            && dataset_d.blk_idxs[3, :] == blk_idxs[3, :])
        @test dataset_d.states == [reflect.(i) for i in dataset_a.states]


        # Validate returns boolean returned by `ison`
        @test ison(dataset_a)
        @test !ison(dataset_e)
    end

    @testset "Filters" begin
        @testset "filter_sparse" begin

            values_a = reshape([1 1 1 1. 1 1 1 0. 0 0 0 1], 2,2,3)
            blk_idxs = rand(1:10, 2, 3)
            states = [[AtomState(t(rand(3)))] for _=1:3]
            
            dataset_a = DataSet(values_a, blk_idxs, states)
            dataset_b = DataSet(-values_a, blk_idxs, states)

            @test length(filter_sparse(dataset_a, 0.9)) == 3
            @test length(filter_sparse(dataset_b, 0.9)) == 3
            @test length(filter_sparse(dataset_a, 1.1)) == 0
        end

        @testset "filter_bond_distance" begin
            values = rand(2,2,4)
            blk_idxs = rand(1:10, 2, 4)
            v = t(rand(3))
            states = [
                [BondState(v, t([11.0, 0, 0]), true)], [BondState(v, t([22.0, 0, 0]), true)],
                [BondState(v, t([5.0, 0, 0]), true)], [BondState(v, t([1.0, 0, 0]), true)]]
            
            dataset_a = DataSet(values, blk_idxs, states)
            dataset_b = DataSet(values, blk_idxs, [[AtomState(v)] for _=1:4])

            @test filter_bond_distance(dataset_a, 10.0) == dataset_a[3:end]
            @test length(filter_bond_distance(dataset_a[1:0], 10.0)) == 0

            @test_throws AssertionError filter_bond_distance(dataset_b, 10.0)




        end
    end
end