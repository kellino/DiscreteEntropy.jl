using Test

@testset "DiscreteEntropy.jl" begin
    # @testset "countdata test" begin
    #     include("countdata_test.jl")
    # end
    # @testset "estimator types test" begin
    #     include("estimator_types.jl")
    # end
    @testset "resampling tests" begin
        include("jackknife_test.jl")
    end
end
