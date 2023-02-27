module DiscreteEntropy

abstract type Estimator end
# Frequentist
struct Maximum_Likelihood <: Estimator end
struct JackknifeML <: Estimator end
struct MillerMadow <: Estimator end
struct Grassberger <: Estimator end
struct Schurmann <: Estimator end
struct SchurmannGeneralised <: Estimator end
struct ChaoShen <: Estimator end
struct Zhang <: Estimator end
struct Bonachela <: Estimator end

export Maximum_Likelihood, JackknifeML, MillerMadow, Grassberger, Schurmann,
    SchurmannGeneralised, ChaoShen, Zhang, Bonachela

# estimators
export maximum_likelihood, jackknife_ml, miller_madow, grassberger,
    schurmann, schurmann_generalised, chao_shen, zhang, bonachela

# Bayesian
struct Bayes <: Estimator end
struct Jeffrey <: Estimator end
struct LaPlace <: Estimator end
struct SchurmannGrassberger <: Estimator end
struct Minimax <: Estimator end
struct NSB <: Estimator end
struct ANSB <: Estimator end
struct PYM <: Estimator end

export Bayes, Jeffrey, LaPlace, SchurmannGrassberger, Minimax, NSB, ANSB, PYM

# estimators
export bayes, jeffrey, laplace, schurmann_grassberger, minimax, nsb, ansb, pym

# Other Discrete Entropy measures, metrics and calculations

export mutual_information
export kl_divergence, jeffreys_divergence, jensen_shannon_divergence, jensen_shannon_distance
export jackknife, bayesian_bootstrap, bootstrap

# utilities
export to_bits, to_bans, to_pmf, gammalndiff, logx, xlogx, logspace, update_or!
export from_samples, from_counts, to_csv_string, from_dict

include("utils.jl")
include("countdata.jl")
include("divergence.jl")
include("mi.jl")
include("resample.jl")
include("Frequentist/frequentist.jl")
include("Frequentist/bub.jl")
include("Bayesian/bayesian.jl")
include("Bayesian/nsb.jl")
include("Bayesian/pym.jl")

end
