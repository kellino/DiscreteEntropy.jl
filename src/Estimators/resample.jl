using Distributions: Dirichlet;
using Random;
using StatsBase: weights, sample, Weights, mean, var

# https://towardsdatascience.com/the-bayesian-bootstrap-6ca4a1d45148

function reduce(i, mat::Matrix)
    d = Dict()

    if mat[:, i][2] > 1
        remainder = mat[:, i][2] - 1
        # 3
        for (j, x) in enumerate(eachcol(mat))
            if j == i
                update_or_insert!(d, x[1], x[2] - 1)
            else
                update_or_insert!(d, x[1], x[2])
            end
        end
        if remainder - mat[:, i][1] != 0
            update_or_insert!(d, 1, 1)
        end

    elseif mat[:, i][2] == 1
        for (j, x) in enumerate(eachcol(mat))
            if j != i
                update_or_insert!(d, x[1], x[2])
            end
        end

        update_or_insert!(d, mat[:, i][1] - 1, 1)
    end

    mm = prod(mat[:, i])
    ks = collect(keys(d))
    vs = collect(values(d))
    hist = [ks vs]'
    cd = CountData(hist, dot(ks, vs), sum(hist[2, :]))
    return (cd, mm)
end

function jk(data::CountData)
    res::Dict{CountData,Int64} = Dict()

    if data.N == data.K
        # if N == K then we only have unique values, ie something like [1,2,3,4] with no repetition
        # we don't need to do a clever reduce here, just knock off one
        mul = data.multiplicities
        mul[2] -= 1
        new = CountData(mul, data.N - 1, data.K - 1)
        res[new] = data.K
    else
        for (i, _) in enumerate(eachcol(data.multiplicities))
            (new, count) = reduce(i, data.multiplicities)
            res[new] = count
        end
    end

    res
end

@doc raw"""
    jackknife(data::CountData, statistic::Function; corrected=false)

Compute the jackknifed estimate of *statistic* on data.
"""
function jackknife(data::CountData, statistic::Function; corrected=false)
    entropies = ((statistic(c), mm) for (c, mm) in jk(data))
    len = sum(mm for (_, mm) in entropies)
    for e in entropies
        println(e)
    end

    μ = 1 / len * sum(h * mm for (h, mm) in entropies)

    denom = 0
    if corrected
        denom = 1 / (len - 1)
    else
        denom = 1 / len
    end

    v = denom * sum([(h - μ)^2 * mm for (h, mm) in entropies])

    return μ, v


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
