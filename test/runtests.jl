using Test, Documenter, DiscreteEntropy

@testset "DiscreteEntropy.jl" begin
    doctest(DiscreteEntropy)

    @testset "countdata test" begin
        include("countdata_test.jl")
    end
    @testset "estimator types test" begin
        include("estimator_types.jl")
    end
    @testset "resampling tests" begin
        include("jackknife_test.jl")
    end
    @testset "util_tests" begin
        include("util_test.jl")
    end
    @testset "estimator_tests" begin
        include("estimator_test.jl")
    end
    @testset "divergence_tests" begin
        include("divergence_test.jl")
    end
end
