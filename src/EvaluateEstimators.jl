include("DiscreteEntropy.jl")
using .DiscreteEntropy
using Random
using Distributions
using NLopt

#-----------------------------------------#
"""

#random sample
Random.seed!(42)
dist = BetaBinomial(100, 1.0, 2.0)
samples = rand(dist, 100)
data = from_samples(svector(samples),true)

gt = entropy(dist)
"""
#-----------------------------------------#

#sample from PYM entropy estimator MATLAB reference implementation
#https://github.com/pillowlab/PYMentropy/tree/master

samples = [1, 2, 2, 3, 3, 4, 4, 4, 4]
data = from_samples(svector(samples),true)

#println(sort(unique(svector(samples).values)))
println(data)
println()

#FREQUENTIST

"""
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

#estimate = estimate_h(data, SchurmannGeneralised) ?

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
#estimate = estimate_h(data, BUB) ?

println()
"""

#BAYESIAN

println("BAYESIAN ESTIMATORS")

estimate = estimate_h(data, PYM)
println("PYM " * string(estimate))
print()
"""
estimate = bayes(data, 0.0, data.K)
println("BAYES " * string(estimate))

#estimate = estimate_h(data, Bayes, 0.0)
estimate = estimate_h(data, LaPlace)
println("LaPlace " * string(estimate))

estimate = estimate_h(data, Jeffrey)
println("Jeffrey " * string(estimate))

estimate = estimate_h(data, SchurmannGrassberger)
println("SchurmannGrassberger " * string(estimate))

estimate = estimate_h(data, Minimax)
println("Minimax " * string(estimate))

estimate = estimate_h(data, NSB)
println("NSB " * string(estimate))

estimate = estimate_h(data, AutoNSB)
println("AutoNSB " * string(estimate))

estimate = estimate_h(data, ANSB)
println("ANSB " * string(estimate))

estimate = estimate_h(data, PERT)
println("PERT " * string(estimate))


print()
"""




