using ACEhamiltonians
using Test

@testset "ACEhamiltonians.jl" begin
    @testset "Unit Tests" begin
        include("unit_tests/test_parameters.jl")
        include("unit_tests/test_data.jl")
        include("unit_tests/test_states.jl")
        include("unit_tests/test_datastructs.jl")
    end

    @testset "Regression Tests" begin
        # Test is skipped as this is all deprecated and must be fully rewritten
        @test_skip include("matrix_tolerance_tests.jl")
    end 
end
