using Distributions: Dirichlet;
using Random;
using StatsBase: weights, sample, Weights, mean, var

# https://towardsdatascience.com/the-bayesian-bootstrap-6ca4a1d45148

function reduce(i, mat::Matrix)
    loc = deepcopy(mat)
    col = mat[:, i]

    loc[2, i] -= 1

    # reducing it deletes the column
    if loc[2, i] == 0
        new = loc[1, i] - 1.0
        loc = loc[:, setdiff(1:end, i)]
        ind = findall(x-> (x == new), loc[1, :])
        if length(ind) == 0
            loc = hcat(loc, [new, 1.0])
        else
            loc[2, ind[1]] += 1.0
        end
    # reducing it does *not* delete the column
    else
        new = loc[1, i] - 1.0
        ind = findall(x-> (x == loc[1, i] - 1.0), loc[1, :])
        if length(ind) == 0
            loc = hcat(loc, [new, 1.0])
        else
            loc[2, ind[1]] += 1.0
        end
    end

    inds = findall(x-> (x == 0.0), loc[1, :])
    loc = loc[:, setdiff(1:end, inds)]

    mm = prod(col)
    cd = CountData(sortslices(loc, dims=2), sum(prod.(eachcol(loc))), sum(loc[2, :]))
    return (cd, mm)
end

function jk(data::CountData)
    res = Dict{CountData, Int64}()

    if data.N == 0.0 || data.K == 0
        return res
    end

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
    reduced = jk(data)
    entropies = ((statistic(c), mm) for (c, mm) in reduced)
    len = sum(mm for (_, mm) in entropies)

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
