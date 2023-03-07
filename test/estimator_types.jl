using DiscreteEntropy
using Test

@test NonParameterisedEstimator <: AbstractEstimator
@test ParameterisedEstimator <: AbstractEstimator

@test MaximumLikelihood <: NonParameterisedEstimator
@test JackknifeML <: NonParameterisedEstimator
@test MillerMadow <: NonParameterisedEstimator
@test Grassberger <: NonParameterisedEstimator
@test ChaoShen <: NonParameterisedEstimator
@test Zhang <: NonParameterisedEstimator
@test Bonachela <: NonParameterisedEstimator
@test Shrink <: NonParameterisedEstimator
