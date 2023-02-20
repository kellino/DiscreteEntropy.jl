using Test

@testset "DiscreteEntropy.jl" begin
    @testset "countdata test" begin
        include("countdata_test.jl")
    end
end
