module DiscreteEntropy

export CountVector, SampleVector, cvector, svector
export Histogram, Samples

export CountData, from_data, from_samples, from_counts

export Maximum_Likelihood, MillerMadow, Grassberger, Schurmann
# , JackknifeML, SchurmannGeneralised, ChaoShen, Zhang, Bonachela


# convenience function to create a unified interface
export estimate_h

# estimators
export maximum_likelihood, miller_madow, grassberger, schurmann
# , jackknife_ml, miller_madohw, grassberger,
#     schurmann, schurmann_generalised, chao_shen, zhang, bonachela


# export Bayes, Jeffrey, LaPlace, SchurmannGrassberger, Minimax, NSB, ANSB, PYM

# estimators
# export bayes, jeffrey, laplace, schurmann_grassberger, minimax, nsb, ansb, pym



# Other Discrete Entropy measures, metrics and calculations
# export mutual_information
# export conditional_entropy
# export kl_divergence, jeffreys_divergence, jensen_shannon_divergence, jensen_shannon_distance

# tools for changing the estimators
# export jackknife, bayesian_bootstrap, bootstrap

# utilities
export to_bits, to_bans

include("Core/countvectors.jl")
include("Core/countdata.jl")
include("Core/utils.jl")

include("Estimators/estimate.jl")
include("Estimators/Frequentist/frequentist.jl")


# include("mutual_information.jl")
# include("conditional_entropy.jl")
# include("divergence.jl")
# include("resample.jl")
# include("Frequentist/bub.jl")
# include("Bayesian/bayesian.jl")
# include("Bayesian/nsb.jl")
# include("Bayesian/pym.jl")

end
