using ACEhamiltonians
using Test

@testset "ACEhamiltonians.jl" begin
    @testset "Unit Tests" begin
        include("unit_tests/test_parameters.jl")
        include("unit_tests/test_data.jl")
        include("unit_tests/test_states.jl")
        include("unit_tests/test_datastructs.jl")
        include("unit_tests/test_bases_simple.jl")
    end
end
