using DiscreteEntropy
using Test

@test NonParameterisedEstimator <: AbstractEstimator
@test ParameterisedEstimator <: AbstractEstimator

@test MaximumLikelihood <: AbstractEstimator
