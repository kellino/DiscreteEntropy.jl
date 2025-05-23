using Distributions: Dirichlet;
using Random;
using StatsBase: weights, sample, Weights, mean, var


function reduce(i, mat::Matrix)
  loc = deepcopy(mat)
  col = mat[:, i]

  loc[2, i] -= 1

  # reducing it deletes the column
  if loc[2, i] == 0
    new = loc[1, i] - 1.0
    loc = loc[:, setdiff(1:end, i)]
    ind = findall(x -> (x == new), loc[1, :])
    if length(ind) == 0
      loc = hcat(loc, [new, 1.0])
    else
      loc[2, ind[1]] += 1.0
    end
    # reducing it does *not* delete the column
  else
    new = loc[1, i] - 1.0
    ind = findall(x -> (x == loc[1, i] - 1.0), loc[1, :])
    if length(ind) == 0
      loc = hcat(loc, [new, 1.0])
    else
      loc[2, ind[1]] += 1.0
    end
  end

  inds = findall(x -> (x == 0.0), loc[1, :])
  loc = loc[:, setdiff(1:end, inds)]

  mm = prod(col)
  cd = CountData(sortslices(loc, dims=2), sum(prod.(eachcol(loc))), sum(loc[2, :]))
  return (cd, mm)
end

function jk(data::CountData)
  res = Dict{CountData,Int64}()

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
     jackknife(data::CountData, estimator::Type{T}; corrected=false) where {T<:AbstractEstimator}

Compute the jackknifed estimate of `estimator` on `data`.

If `corrected` is true, then the variance is scaled with `data.N - 1`, else it is scaled with `data.N`. `corrected`
has no effect on the entropy estimation.
"""
function jackknife(data::CountData, estimator::Type{T}; corrected=false) where {T<:AbstractEstimator}
  reduced = jk(data)
  entropies = ((estimate_h(c, estimator), mm) for (c, mm) in reduced)
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

@doc raw"""
     bayesian_bootstrap(samples::SampleVector, estimator::Type{T}, reps, seed, concentration) where {T<:AbstractEstimator}

Compute a bayesian bootstrap resampling of `samples` for estimation with `estimator`, where
`reps` is number of resampling to perform, `seed` is the random seed and `concentration` is the
concentration parameter for a Dirichlet distribution.

# External Links
[The Bayesian Bootstrap](https://projecteuclid.org/journals/annals-of-statistics/volume-9/issue-1/The-Bayesian-Bootstrap/10.1214/aos/1176345338.full)
"""
function bayesian_bootstrap(samples::SampleVector, estimator::Type{T}, reps::Int, seed::Int64, concentration::Real) where {T<:AbstractEstimator}
  out = zeros(reps)

  Threads.@threads for i = 1:reps
    out[i] = _bbstrap(samples, estimator, seed, concentration)
  end

  return mean(out), var(out)

end

function _bbstrap(samples::SampleVector, estimator::Type{T}, seed, concentration) where {T<:AbstractEstimator}
  Random.seed!(seed)
  weights = Weights(rand(Dirichlet(ones(length(samples)) .* concentration)))
  boot = sample(samples, weights, length(samples))

  return estimate_h(from_data(boot, Samples), estimator)
end
