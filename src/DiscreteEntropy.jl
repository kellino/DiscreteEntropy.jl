module DiscreteEntropy

const estimator = Set([
    :maximum_likelihood, :miller_madow, :grassberger, :schurmann, :chao_shen, :zhang
])

export to_bits, to_bans, to_pmf, gammalndiff, logx, xlogx, logspace, update_or!
export maximum_likelihood, miller_madow, grassberger,
    schurmann, chao_shen, zhang, bonachela, schurmann_generalised, jackknife_ml
export bayes, jeffrey, laplace, schurmann_grassberger, minimax, nsb, ansb
export from_samples, from_counts, to_csv_string, from_dict
export kl_divergence, jeffreys_divergence, jensen_shannon_divergence
export jackknife

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
