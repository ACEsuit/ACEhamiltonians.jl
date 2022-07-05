using ACEhamiltonians
using Test

@testset "ACEhamiltonians.jl" begin
    @testset "Unit Tests" begin
        include("unit_tests/test_parameters.jl")
    end

    @testset "Regression Tests" begin
        include("matrix_tolerance_tests.jl")
    end 
end
