include("../src/DiscreteEntropy.jl")

using .DiscreteEntropy
using Optim, Distributions
using CSV

function generate(samplesize, dist)
    data = DiscreteEntropy.from_samples(rand(dist, samplesize))

    function loss(ξ)
        return abs(DiscreteEntropy.schurmann(data, ξ)) - entropy(dist)
    end

    res = Optim.optimize(loss, 0.1, 10.0)
    return (data, Optim.minimizer(res))
end
