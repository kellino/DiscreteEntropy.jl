@doc raw"""
     cross_entropy(P::CountVector, Q::CountVector, ::Type{T}) where {T<:AbstractEstimator}

```math
H(P,Q) = - \sum_x(P(x) \log(Q(x)))
```

Compute the cross entropy of ``P`` and ``Q``, given an estimator of type ``T``.
``P`` and ``Q`` must be the same length. Both vectors are normalised. The cross entropy
of a probability distribution ``P`` with itself, is equal to its entropy,
ie ``H(P, P) = H(P)``.

# Example
```@jldoctest

julia> P = cvector([1,2,3,4,3,2])
julia> Q = cvector([2,5,5,4,3,4])

julia> ce = cross_entropy(P, Q, MaximumLikelihood)
1.778564897565542
```
Note: not every estimator is currently supported.
"""
function cross_entropy(P::CountVector, Q::CountVector, ::Type{MillerMadow})
  N = sum(P) + sum(Q)
  K = ((length(P) + length(Q)) / 2.0) - 1.0
  cross_entropy(P, Q, MaximumLikelihood) + K / N
end

function cross_entropy(P::AbstractVector{R}, Q::AbstractVector{R}, ::Type{MaximumLikelihood}) where {R<:Real}
  @assert length(P) == length(Q)
  freqs1 = P ./ sum(P)
  freqs2 = Q ./ sum(Q)
  if 0.0 in freqs2
    return Inf
  end
  c = -sum(freqs1 .* logx.(freqs2))
  abs(c)
end

function cross_entropy(P::CountVector, Q::CountVector, ::Type{Bayes}, α::Float64)
  p = pmf(cvector(P .+ α))
  q = pmf(cvector(Q .+ α))
  cross_entropy(p, q, MaximumLikelihood)
end

function cross_entropy(P::CountVector, Q::CountVector, ::Type{Laplace})
  cross_entropy(P, Q, Bayes, 1.0)
end
function cross_entropy(P::CountVector, Q::CountVector, ::Type{Jeffrey})
  cross_entropy(P, Q, Bayes, 0.5)
end
function cross_entropy(P::CountVector, Q::CountVector, ::Type{SchurmannGrassberger})
  cross_entropy(P, Q, Bayes, 1 / length(P))
end
function cross_entropy(P::CountVector, Q::CountVector, ::Type{Minimax})
  N = sqrt((sum(P) + sum(Q)) / 2) / length(P)
  cross_entropy(P, Q, Bayes, N)
end


@doc raw"""
    kl_divergence(P::CountVector, Q::CountVector, estimator::Type{T}; truncate::Union{Nothing, Int} = nothing) where {T<:AbstractEstimator}

```math
D_{KL}(P ‖ Q) = \sum_{x \in X} P(x) \log \left( \frac{P(x)}{Q(x)} \right)
```

Compute the [Kullback-Lebler Divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Interpretations)
between two discrete distributions. ``P`` and ``Q`` must be the same length.
If the distributions are not normalised, they will be.

If the distributions are not over the same space or the cross entropy is negative, then it returns ``Inf``.

If truncate is set to some integer value, ```x```, return kl_divergence rounded to ```x``` decimal places.
"""
function kl_divergence(P::CountVector, Q::CountVector, estimator::Type{T}; truncate::Union{Nothing,Int}=nothing, α=0.0) where {T<:AbstractEstimator}
  if estimator == Bayes
    cr = cross_entropy(P, Q, Bayes, α)
    c = cr - estimate_h(from_counts(P), Bayes, α)
  else
    cr = cross_entropy(P, Q, estimator)
    c = cr - estimate_h(from_counts(P), estimator)
  end
  if cr < 0.0
    return Inf
  end

  if truncate !== nothing
    c = round(c, digits=truncate)
  end
  c < 0.0 ? 0.0 : c
end
function kl_divergence(P::CountVector, Q::CountVector; truncate::Union{Nothing,Int}=nothing)
  kl_divergence(P, Q, MaximumLikelihood, truncate=truncate)
end
function kl_divergence(P::CountVector, Q::CountVector, ::Type{Bayes}, α::Float64; truncate::Union{Nothing,Int}=nothing)
  kl_divergence(P, Q, Bayes, truncate=truncate)
end


@doc raw"""
    jensen_shannon_divergence(countsP::CountVector, countsQ::CountVector)
    jensen_shannon_divergence(countsP::CountVector, countsQ::CountVector, estimator::Type{T}) where {T<:NonParamterisedEstimator}
    jensen_shannon_divergence(countsP::CountVector, countsQ::CountVector, estimator::Type{Bayes}, α)

Compute the Jensen Shannon Divergence between discrete distributions $P$ and $Q$, as represented by
their histograms. If no estimator is specified, it defaults to MaximumLikelihood.

```math
\widehat{JS}(P, Q) = \hat{H}\left(\frac{P + Q}{2} \right) - \left( \frac{H(P) + H(Q)}{2} \right)

```
"""
function jensen_shannon_divergence(P::CountVector, Q::CountVector, estimator::Type{T}) where {T<:AbstractEstimator}
  abs(0.5 * kl_divergence(P, Q, estimator) + 0.5 * kl_divergence(Q, P, estimator))
end
function jensen_shannon_divergence(P::CountVector, Q::CountVector)
  jensen_shannon_divergence(P, Q, MaximumLikelihood)
end

@doc raw"""
    jensen_shannon_distance(P::CountVector, Q::CountVector, estimator::Type{T}) where {T<:AbstractEstimator}

Compute the Jensen Shannon Distance

"""
function jensen_shannon_distance(P::CountVector, Q::CountVector, estimator::Type{T}) where {T<:AbstractEstimator}
  return sqrt(jensen_shannon_divergence(P, Q, estimator))
end

@doc raw"""
    jeffreys_divergence(P::CountVector, Q::CountVector)
    jeffreys_divergence(P::CountVector, Q::CountVector, estimator::Type{T}) where T<:AbstractEstimator

```math
J(p, q) = D_{KL}(p \Vert q) + D_{KL}(q \Vert p)
```

If no estimator is specified, then we calculate using maximum likelihood
# External Links

[Paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7516653/)
"""
function jeffreys_divergence(P::CountVector, Q::CountVector)
  return kl_divergence(P, Q, MaximumLikelihood) + kl_divergence(P, Q, MaximumLikelihood)
end

function jeffreys_divergence(P::CountVector, Q::CountVector, estimator::Type{T}) where {T<:AbstractEstimator}
  return kl_divergence(P, Q, estimator) + kl_divergence(P, Q, estimator)
end
