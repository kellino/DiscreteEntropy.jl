module DiscreteEntropy

const estimators = Set([
    :maximum_likelihood, :miller_madow, :grassberger, :schurmann, :chao_shen, :zhang
])

export to_bits, to_bans, basic_jack, to_pmf, gammalndiff, logx, xlogx
export maximum_likelihood, miller_madow, grassberger,
    schurmann, chao_shen, zhang, bonachela, schurmann_generalised
export bayes, jeffrey, laplace, schurmann_grassberger, minimax, nsb, ansb
export from_samples, from_counts, to_csv_string
export kl_divergence, mi

include("utils.jl")
include("countdata.jl")
include("divergence.jl")
include("frequentist.jl")
include("Bayesian/bayesian.jl")
include("Bayesian/nsb.jl")
include("Bayesian/pym.jl")

end
