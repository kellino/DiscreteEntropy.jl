include("DiscreteEntropy.jl")
include("Core/utils.jl")
using .DiscreteEntropy
using Random
using Distributions
using NLopt



#-----------------------------------------#
function H_estimation(data)
    # Frequentist
    println("FREQUENTIST ESTIMATORS")

    estimate = estimate_h(data, MaximumLikelihood)
    println("MaximumLikelihood " * string(estimate))

    #estimate = estimate_h(data, JackknifeML) ?

    estimate = estimate_h(data, MillerMadow)
    println("MillerMadow " * string(estimate))

    estimate = estimate_h(data, Grassberger88)
    println("Grassberger88 " * string(estimate))

    estimate = estimate_h(data, Grassberger03)
    println("Grassberger03 " * string(estimate))

    estimate = estimate_h(data, Schurmann)
    println("Schurmann " * string(estimate))

    #estimate = estimate_h(cvector(samples), SchurmannGeneralised, dist) ?

    estimate = estimate_h(data, ChaoShen)
    println("ChaoShen " * string(estimate))

    estimate = estimate_h(data, Zhang)
    println("Zhang " * string(estimate))

    estimate = estimate_h(data, Shrink)
    println("Shrink " * string(estimate))

    estimate = estimate_h(data, Bonachela)
    println("Bonachela " * string(estimate))

    estimate = estimate_h(data, ChaoWangJost)
    println("ChaoWangJost " * string(estimate))

    estimate = estimate_h(data, BUB)
    #println("BUB " * string(estimate))

    println()
    #------------------------------------------#

    # Bayesian
    println("BAYESIAN ESTIMATORS")

    estimate = estimate_h(data, PYM)
    println("PYM " * string(estimate))
    print()

    estimate = estimate_h(data, Bayes, 0.0)
    println("BAYES " * string(estimate))

    estimate = estimate_h(data, LaPlace)
    println("LaPlace " * string(estimate))

    estimate = estimate_h(data, Jeffrey)
    println("Jeffrey " * string(estimate))

    estimate = estimate_h(data, SchurmannGrassberger)
    println("SchurmannGrassberger " * string(estimate))

    estimate = estimate_h(data, Minimax)
    println("Minimax " * string(estimate))

    estimate = estimate_h(data, NSB, false)
    println("NSB " * string(estimate))

    estimate = estimate_h(data, ANSB)
    println("ANSB " * string(estimate))

    estimate = estimate_h(data, PERT)
    println("PERT " * string(estimate))

    print()
end


"""
# Random sample
# Samples from dist (n. of trials/observations)
l = 100
# Support set distribution (upper bound range of the observed variable's possible values, if.n = 100 -> [0-100])
global n = 100

Random.seed!(40)
dist = BetaBinomial(n, 1.0, 2.0)
samples = rand(dist, l)
data = from_samples(svector(samples),true)

gt = entropy(dist)
"""

#-----------------------------------------#

# Sample from PYM entropy estimator MATLAB reference implementation
# https://github.com/pillowlab/PYMentropy/tree/master

"""
samples = [1, 2, 2, 3, 3, 4, 4, 4, 4]
data = from_samples(svector(samples),true)

#println(sort(unique(svector(samples).values)))
println(data)

H_estimation(data)
"""









