@doc raw"""
     cross_entropy(P::CountVector, Q::CountVector, ::Type{T}) where {T<:MaximumLikelihood}

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
function cross_entropy(P::Vector{Int}, Q::Vector{Int}, t::Type{T}) where {T<:AbstractEstimator}
    @warn("assuming P and Q are count vectors")
    cross_entropy(cvector(P), cvector(Q), t)
end

function cross_entropy(P::CountVector, Q::CountVector, ::Type{MaximumLikelihood})
    @assert length(P) == length(Q)
    freqs1 = pmf(P)
    freqs2 = pmf(Q)
    - sum(freqs1 .* logx.(freqs2))
end

function cross_entropy(P::CountVector, Q::CountVector, ::Type{MillerMadow})
    N = sum(P) + sum(Q)
    K = ((length(P) + length(Q)) / 2.0) - 1.0
    cross_entropy(P, Q, MaximumLikelihood) + K / N
end

function cross_entropy(P::CountVector, Q::CountVector, ::Type{Bayes}, α::Float64)
    p = pmf(cvector(P .+ α))
    q = pmf(cvector(Q .+ α))
    pa = p ./ sum(p)
    qa = q ./ sum(q)
    - sum(pa .* logx.(qa))
end

function cross_entropy(P::CountVector, Q::CountVector, ::Type{LaPlace}) cross_entropy(P, Q, Bayes, 1.0) end
function cross_entropy(P::CountVector, Q::CountVector, ::Type{Jeffrey}) cross_entropy(P, Q, Bayes, 0.5) end
function cross_entropy(P::CountVector, Q::CountVector, ::Type{SchurmannGrassberger}) cross_entropy(P, Q, Bayes, 1 / length(P)) end
function cross_entropy(P::CountVector, Q::CountVector, ::Type{Minimax})
    N = sqrt( (sum(P) + sum(Q)) / 2) / length(P)
    cross_entropy(P, Q, Bayes, N)
end


@doc raw"""
    kl_divergence(p::AbstractVector, q::AbstractVector)::Float64

```math
D_{KL}(P ‖ Q) = \sum_{x \in X} P(x) \log \left( \frac{P(x)}{Q(x)} \right)
```

Compute the [Kullback-Lebler Divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Interpretations)
between two discrete distributions. Both distributions needs to be defined over the same space,
so length(p) == length(q). If the distributions are not normalised, they will be.
"""
function kl_divergence(P::CountVector, Q::CountVector, estimator::Type{T}; truncate=true) where {T<:AbstractEstimator}
    c = cross_entropy(P, Q, estimator) - estimate_h(from_counts(P), estimator)
    if truncate
        round(c, digits=10)
    else
        c
    end
end

@doc raw"""
    jensen_shannon_divergence(countsP::AbstractVector, countsQ::AbstractVector)
    jensen_shannon_divergence(countsP::AbstractVector, countsQ::AbstractVector, estimator::Type{T}) where {T<:NonParamterisedEstimator}
    jensen_shannon_divergence(countsP::AbstractVector, countsQ::AbstractVector, estimator::Type{Bayes}, α)

Compute the Jensen Shannon Divergence between discrete distributions $P$ and $Q$, as represented by
their histograms. If no estimator is specified, it defaults to MaximumLikelihood.

```math
\widehat{JS}(P, Q) = \hat{H}\left(\frac{P + Q}{2} \right) - \left( \frac{H(P) + H(Q)}{2} \right)

```
"""
function jensen_shannon_divergence(P::CountVector, Q::CountVector)
    abs(0.5 * kl_divergence(P, Q, MaximumLikelihood) + 0.5 *
        kl_divergence(Q, P, MaximumLikelihood))
end

function jensen_shannon_divergence(P::CountVector, Q::CountVector, estimator::Type{T}) where {T<:AbstractEstimator}
    abs(0.5 * kl_divergence(P, Q, estimator) + 0.5 * kl_divergence(Q, P, estimator))
end

@doc raw"""
    jensen_shannon_distance(P::AbstractVector, Q::AbstractVector, estimator)

Compute the Jensen Shannon Distance

"""
function jensen_shannon_distance(P::AbstractVector, Q::AbstractVector, estimator::Type{T}) where {T<:AbstractEstimator}
    return sqrt(jensen_shannon_divergence(P, Q, estimator))
end

@doc raw"""
    jeffreys_divergence(p, q)

```math
J(p, q) = D_{KL}(p \Vert q) + D_{KL}(q \Vert p)
```

# External Links

[Paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7516653/)
"""
function jeffreys_divergence(P, Q)
    return kl_divergence(P, Q, MaximumLikelihood) + kl_divergence(P, Q, MaximumLikelihood)
end

function jeffreys_divergence(P, Q, estimator::Type{T}) where {T<:AbstractEstimator}
    return kl_divergence(P, Q, estimator) + kl_divergence(P, Q, estimator)
end
