module DiscreteEntropy

export CountVector, SampleVector, cvector, svector
export SampleHistogram, Samples

export CountData, from_data, from_samples, from_counts

export Maximum_Likelihood
# , JackknifeML, MillerMadow, Grassberger, Schurmann, SchurmannGeneralised, ChaoShen, Zhang, Bonachela


export estimate_h

# estimators
export maximum_likelihood
# , jackknife_ml, miller_madohw, grassberger,
#     schurmann, schurmann_generalised, chao_shen, zhang, bonachela


# export Bayes, Jeffrey, LaPlace, SchurmannGrassberger, Minimax, NSB, ANSB, PYM

# estimators
# export bayes, jeffrey, laplace, schurmann_grassberger, minimax, nsb, ansb, pym

# convenience function to create a unified interface
# export estimate_h

# Other Discrete Entropy measures, metrics and calculations
# export mutual_information
# export conditional_entropy
# export kl_divergence, jeffreys_divergence, jensen_shannon_divergence, jensen_shannon_distance

# tools for changing the estimators
# export jackknife, bayesian_bootstrap, bootstrap

# utilities
# export to_bits, to_bans, to_pmf, gammalndiff, logx, xlogx, logspace, update_or_insert!
# export from_samples, from_counts, to_csv_string

include("countvectors.jl")
include("countdata.jl")
include("estimate.jl")
include("utils.jl")
# include("mutual_information.jl")
# include("conditional_entropy.jl")
# include("divergence.jl")
# include("resample.jl")
include("Frequentist/frequentist.jl")
# include("Frequentist/bub.jl")
# include("Bayesian/bayesian.jl")
# include("Bayesian/nsb.jl")
# include("Bayesian/pym.jl")

end
