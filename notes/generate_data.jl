include("../src/DiscreteEntropy.jl")

using .DiscreteEntropy
using Optim, Distributions
using Printf

function generate(samplesize, dist)
    data = DiscreteEntropy.from_samples(rand(dist, samplesize))

    function loss(ξ)
        return abs(DiscreteEntropy.schurmann(data, ξ)) - entropy(dist)
    end

    res = Optim.optimize(loss, 0.1, 20.0)
    return (data, Optim.minimizer(res), Optim.minimum(res))
end


function make_data(file::String, lines::Int, samplesize::Int, distribution)
    out = open(file, "a")

    for _ in 1:lines
        data, xi, min = generate(samplesize, distribution)
        write(out, @sprintf("%s,%.10f%.10f\n", DiscreteEntropy.to_csv_string(data), xi, min))
    end

    close(out)
end
