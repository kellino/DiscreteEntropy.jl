module DiscreteEntropy

import Base: length, size, sum, show
using Base: @propagate_inbounds

export Axis
export CountVector, SampleVector, XiVector, cvector, svector, xivector
export Histogram, Samples

export CountData, from_data, from_samples, from_counts, singletons, doubletons, marginal_counts, print_data

export AbstractEstimator, NonParameterisedEstimator, ParameterisedEstimator
export MaximumLikelihood, JackknifeMLE, MillerMadow, Grassberger88, Grassberger03,
    Schurmann, SchurmannGeneralised,
    ChaoShen, Zhang, Bonachela, Shrink, BUB, ChaoWangJost, PERT

# convenience function to create a unified interface
export estimate_h, estimate_h_and_var

# estimators
export maximum_likelihood, miller_madow, jackknife_mle, grassberger1988, schurmann, schurmann_generalised,
    chao_shen, zhang, bonachela, shrink, chao_wang_jost, bub

export Bayes, Jeffrey, LaPlace, SchurmannGrassberger, Minimax, NSB, AutoNSB, ANSB, PYM

# estimators
export bayes, jeffrey, laplace, schurmann_grassberger, minimax, nsb, ansb
# pym, cdm

# export conditional_entropy
# export kl_divergence, jeffreys_divergence, jensen_shannon_divergence, jensen_shannon_distance

# tools for changing the estimators
# export jackknife, bayesian_bootstrap, bootstrap

# utilities
export to_bits, to_bans, xlogx, logx

include("Core/countvectors.jl")
include("Core/countdata.jl")
include("Core/utils.jl")

include("Estimators/estimate.jl")
include("Estimators/Frequentist/frequentist.jl")
include("Estimators/Frequentist/bub.jl")
include("Estimators/Bayesian/bayesian.jl")
include("Estimators/Bayesian/nsb.jl")
include("Estimators/Bayesian/pym.jl")
include("Estimators/resample.jl")

# include("conditional_entropy.jl")
# include("divergence.jl")

end
