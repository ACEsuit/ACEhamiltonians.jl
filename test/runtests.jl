using ACEhamiltonians
using Test

@testset "ACEhamiltonians.jl" begin
    @testset "Unit Tests" begin
        include("unit_tests/test_parameters.jl")
        include("unit_tests/test_data.jl")
        include("unit_tests/test_states.jl")
    end

    @testset "Regression Tests" begin
        include("matrix_tolerance_tests.jl")
    end 
end
