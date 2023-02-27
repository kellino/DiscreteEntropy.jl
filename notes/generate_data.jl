using StatsBase: Random
using Distributions
using Optim
using SpecialFunctions: digamma
using QuadGK
using StatsBase

function schurmann(samples::Vector, ξ)
    N = length(samples)
    map = countmap(samples)

    return digamma(N) - (1.0 / N) *
                        sum([(digamma(y) + (-1)^y * quadgk(t -> t^(y - 1) / (1 + t), 0, (1 / ξ) - 1.0)[1]) * y for (_, y) in map])
end

function ξ_limit()
    Random.seed!(42)
    dist = BetaBinomial(100, 1.0, 2.0)
    samples = rand(dist, 100)

    function loss(ξ)
        return abs(schurmann(samples, ξ) - entropy(dist))
    end

    xs = 0.3:0.01:10.0
    ys = [loss(ξ) for ξ in xs]
    return (xs, ys)
end

bern = Bernoulli()

samples = rand(bern, 2000)

xs = 10:10:2000

# standard = [schurmann(from_samples(samples[1:size])) for size in xs]

function f(x)
    s = samples[1:100]
    return abs(schurmann(s, x) - entropy(bern))
end

function get_best(dist, s)
    function loss(ξ)
        return abs(schurmann(s, ξ) - entropy(dist))
    end
    res = Optim.optimize(loss, 0.1, 10.0)
    return res
end

# function irf(samples)
#     return roots(ξ -> DiscreteEntropy.schurmann(DiscreteEntropy.from_samples(samples), ξ) - entropy(bern), 0 .. 10)
# end
