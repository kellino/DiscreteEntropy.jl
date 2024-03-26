using DiscreteEntropy
using Test

@test NonParameterisedEstimator <: AbstractEstimator
@test ParameterisedEstimator <: AbstractEstimator

@test MaximumLikelihood <: NonParameterisedEstimator
@test JackknifeMLE <: NonParameterisedEstimator
@test MillerMadow <: NonParameterisedEstimator
@test Grassberger88 <: NonParameterisedEstimator
@test Grassberger03 <: NonParameterisedEstimator
@test ChaoShen <: NonParameterisedEstimator
@test Zhang <: NonParameterisedEstimator
@test Bonachela <: NonParameterisedEstimator
@test Shrink <: NonParameterisedEstimator
