using DiscreteEntropy
using Test

@test NonParameterisedEstimator <: AbstractEstimator
@test ParameterisedEstimator <: AbstractEstimator

@test MaximumLikelihood <: NonParameterisedEstimator
@test JackknifeMLE <: NonParameterisedEstimator
@test MillerMadow <: NonParameterisedEstimator
@test Grassberger <: NonParameterisedEstimator
@test ChaoShen <: NonParameterisedEstimator
@test Zhang <: NonParameterisedEstimator
@test Bonachela <: NonParameterisedEstimator
@test Shrink <: NonParameterisedEstimator
@test ChaoWangJost <: NonParameterisedEstimator

# Frequentist with Parameter(s)
@test Schurmann <: ParameterisedEstimator
@test SchurmannGeneralised <: ParameterisedEstimator

# Bayesian with Parameter(s)
@test Bayes <: ParameterisedEstimator
@test NSB <: ParameterisedEstimator
@test PYM <: ParameterisedEstimator

@test AutoNSB <: NonParameterisedEstimator
@test ANSB <: NonParameterisedEstimator
@test Jeffrey <: NonParameterisedEstimator
@test LaPlace <: NonParameterisedEstimator
@test SchurmannGrassberger <: NonParameterisedEstimator
@test Minimax <: AbstractEstimator
