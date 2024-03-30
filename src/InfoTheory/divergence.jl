import Base.Threads.@spawn

@doc raw"""
    kl_divergence(p::AbstractVector, q::AbstractVector)::Float64

```math
D_{KL}(P ‖ Q) = \sum_{x \in X} P(x) \log \left( \frac{P(x)}{Q(x)} \right)
```

Compute the [Kullback-Lebler Divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Interpretations)
between two discrete distributions. Both distributions needs to be defined over the same space,
so length(p) == length(q). If the distributions are not normalised, they will be.
"""
function kl_divergence(P::CountVector, Q::CountVector)
    @assert length(p) == length(q)

    # check that both distributions are normalised
    freqs1 = to_pmf(p)
    freqs2 = to_pmf(q)

    return sum(freqs1 .* logx.(freqs1 ./ freqs2))
end

@doc raw"""
    jenson_shannon_divergence(countsP::AbstractVector, countsQ::AbstractVector)
    jenson_shannon_divergence(countsP::AbstractVector, countsQ::AbstractVector, estimator::Type{T}) where {T<:NonParamterisedEstimator}

Compute the Jenson Shannon Divergence between discrete distributions $P$ and $q$, as represented by
their histograms.

```math
\widehat{JS}(p, q) = \hat{H}\left(\frac{p + q}{2} \right) - \left( \frac{H(p) + H(q)}{2} \right)

```
"""
function jensen_shannon_divergence(p::AbstractVector, q::AbstractVector)
    p = p ./ sum(p)
    q = q ./ sum(q)
    m = 0.5 .* (p .+ q)

    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
end

function jensen_shannon_divergence(countsP::AbstractVector, countsQ::AbstractVector, estimator::Type{T}) where {T<:NonParameterisedEstimator}
    p = @spawn from_counts(countsP)
    q = @spawn from_counts(countsQ)
    pq = @spawn from_counts(0.5 .* (countsP .+ countsQ))

    estimate_h(fetch(pq), estimator) - 0.5 * estimate_h(fetch(p), estimator) + estimate_h(fetch(q), estimator)
end

@doc raw"""
    jensen_shannon_distance(P::AbstractVector, Q::AbstractVector, estimator)

Compute the Jensen Shannon Distance

"""
function jensen_shannon_distance(P::AbstractVector, Q::AbstractVector, estimator::Function)
    return sqrt(jensen_shannon_divergence(P, Q, estimator))
end

@doc raw"""
    jeffreys_divergence(p, q)
    (link)[https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7516653/]

```math
J(p, q) = D_{KL}(p \Vert q) + D_{KL}(q \Vert p)
```
"""
function jeffreys_divergence(p, q)
    return kl_divergence(p, q) + kl_divergence(q, p)
end
