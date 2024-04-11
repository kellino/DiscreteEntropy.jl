using Test, DiscreteEntropy, Random

Random.seed!(1)

m = reshape(rand(1:10, 25), (5, 5))
