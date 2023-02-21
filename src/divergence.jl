@doc raw"""
    kl_divergence(p::AbstractVector, q::AbstractVector)::Float64

```math
D_{KL}(P ‖ Q) = \sum_{x \in X} P(x) \log \left( \frac{P(x)}{Q(x)} \right)
```

Returns the [Kullback-Lebler Divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Interpretations)
between two discrete distributions. Both distributions needs to be defined over the same space,
so length(p) == length(q). If the distributions are not normalised, they will be.
"""
function kl_divergence(p::AbstractVector, q::AbstractVector)::Float64
    @assert length(p) == length(q)

    # check that both distributions are normalised
    freqs1 = to_pmf(p)
    freqs2 = to_pmf(q)

    return sum(freqs1 .* logx.(freqs1 ./ freqs2))

end

@doc raw"""
    jenson_shannon_divergence(p::AbstractVector, q::AbstractVector)::Float64

Returns the Jenson Shannon Divergence between discrete distributions $p$ and $q$.
"""
function jensen_shannon_divergence(p::AbstractVector, q::AbstractVector)::Float64
    p = p ./ sum(p)
    q = q ./ sum(q)
    m = 0.5 .* (p .+ q)

    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
end

@doc raw"""

```math
\hat{JS}(p, q) = \hat{H}\frac{p + q}{2} - \frac{H(p) + H(q)}{2}

```
"""
function jensen_shannon_divergence(p::AbstractVector, q::AbstractVector, estimator::Function)
    datap = from_counts(p)
    dataq = from_counts(q)
    datapq = from_counts((p .+ q) ./ 2.0)
    # TODO this does not seem to produce the desired output
    # set_N!(datapq, 0.5 * datap.N + dataq.N)

    return estimator(datapq) - 0.5 * (estimator(datap) + estimator(dataq))

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
