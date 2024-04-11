module DiscreteEntropy

import Base: length, size, sum, show
using Base: @propagate_inbounds

export Axis
export CountVector, SampleVector, XiVector, cvector, svector, xivector, EntropyData
export Histogram, Samples

export CountData, from_data, from_samples, from_counts, singletons, doubletons,
    marginal_counts, bins, multiplicities, from_csv

export AbstractEstimator, NonParameterisedEstimator, ParameterisedEstimator
export MaximumLikelihood, JackknifeMLE, MillerMadow, Grassberger,
    Schurmann, SchurmannGeneralised,
    ChaoShen, Zhang, Bonachela, Shrink, ChaoWangJost, PERT
export Bootstrap

# convenience function to create a unified interface
export estimate_h, estimate_h_and_var

# estimators
export maximum_likelihood, miller_madow, jackknife_mle, grassberger, schurmann, schurmann_generalised,
    chao_shen, zhang, bonachela, shrink, chao_wang_jost

export Bayes, Jeffrey, LaPlace, SchurmannGrassberger, Minimax, NSB, AutoNSB, ANSB, PYM

# estimators
export bayes, jeffrey, laplace, schurmann_grassberger, minimax, nsb, ansb, pym, pert, bayesian_bootstrap

# export mutual_info
export mutual_information

# export conditional_entropy
export kl_divergence, jeffreys_divergence, jensen_shannon_divergence, jensen_shannon_distance, cross_entropy,
    conditional_entropy

# tools for changing the estimators
export jackknife #, bayesian_bootstrap, bootstrap

# utilities
export to_bits, to_bans, xlogx, logx

include("Core/countvectors.jl")
include("Core/countdata.jl")
include("Core/utils.jl")

include("Estimators/estimate.jl")
include("Estimators/Frequentist/frequentist.jl")
include("Estimators/Bayesian/bayesian.jl")
include("Estimators/Bayesian/nsb.jl")
include("Estimators/Bayesian/pym.jl")
include("Estimators/resample.jl")

include("InfoTheory/mutual_info.jl")
include("InfoTheory/conditional_entropy.jl")
include("InfoTheory/divergence.jl")

end
