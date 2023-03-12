module DiscreteEntropy

import Base: length, size, sum, show
using Base: @propagate_inbounds

export CountVector, SampleVector, XiVector, cvector, svector, xivector
export Histogram, Samples

export CountData, from_data, from_samples, from_counts

export AbstractEstimator, NonParameterisedEstimator, ParameterisedEstimator
export MaximumLikelihood, JackknifeML, MillerMadow, Grassberger, Schurmann, SchurmannGeneralised,
    ChaoShen, Zhang, Bonachela, Shrink


# convenience function to create a unified interface
export estimate_h

# estimators
export maximum_likelihood, miller_madow, grassberger, schurmann, schurmann_generalised,
    chao_shen, zhang, bonachela, shrink
# , jackknife_ml, chao_wong_grjost, bub


export Bayes, Jeffrey, LaPlace, SchurmannGrassberger, Minimax, NSB, ANSB, PYM

# estimators
export bayes, jeffrey, laplace, schurmann_grassberger, minimax, nsb, ansb
# pym

export redundancy, uncertainty_coefficient

# Other Discrete Entropy measures, metrics and calculations
# export mutual_information
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
include("Estimators/Bayesian/bayesian.jl")
include("Estimators/Bayesian/nsb.jl")

include("InfoTheory/mutual_information.jl")
# include("conditional_entropy.jl")
# include("divergence.jl")
# include("resample.jl")
# include("Frequentist/bub.jl")
# include("Bayesian/bayesian.jl")
# include("Bayesian/nsb.jl")
# include("Bayesian/pym.jl")

end
