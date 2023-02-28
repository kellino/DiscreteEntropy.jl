using Distributions: Dirichlet;
using Random;
using StatsBase;

# TODO this need thorough testing!
function jk(data::CountData)
    res::Dict{CountData,Int64} = Dict()
    ks = collect(keys(data.histogram))
    vs = collect(values(data.histogram))

    res[data] = 1
    for i in 1:length(ks)
        (d, mm) = reduce(i, ks, vs)
        res[from_dict(d)] = mm
    end

    return res
end

@doc raw"""
    jackknife(data::CountData, statistic::Function; corrected=false)

Returns the jacknifed estimate of *statistic* on data.
"""
function jackknife(data::CountData, statistic::Function; corrected=false)
    entropies = [(statistic(c), mm) for (c, mm) in jk(data)]
    len = sum([mm for (_, mm) in entropies])

    μ = 1.0 / len * sum(h * mm for (h, mm) in entropies)

    denom = 0
    if corrected
        denom = 1.0 / (len - 1)
    else
        denom = 1.0 / len
    end

    v = denom * sum([(h - μ)^2.0 * mm for (h, mm) in entropies])

    return μ, v


end

function reduce(i, ks, vs)
    d::Dict{Int64,Int64} = Dict()
    mm = 1
    if vs[i] == 1
        n = ks[i] - 1
        ks[i] = n
        for j in 1:length(ks)
            update_or_insert!(d, ks[j], vs[j])
        end
        ks[i] += 1
    elseif vs[i] > 1
        mm = vs[i]
        vs[i] -= 1
        for j in 1:length(ks)
            update_or_insert!(d, ks[j], vs[j])
        end
        vs[i] = 1
    end
    return (d, mm)
end

# TODO add bootstrap resampling

function bootstrap(samples::AbstractVector, method, statistic; K=1000)
    # How do we do this directly over multiplicities. It should be much more efficient
    out = zeros(K)

    Threads.@threads for i = 1:K
        out[i] = method(samples, statistic)
    end

    return mean(out), var(out)

end

function bayesian_bootstrap(samples::AbstractVector, statistic::Function; seed=1, concentration=4)
    Random.seed!(seed)
    weights = Weights(rand(Dirichlet(ones(length(samples)) .* concentration)))
    boot = sample(samples, weights, length(samples))

    return statistic(boot)
end
