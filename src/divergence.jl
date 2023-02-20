function kl_divergence(counts1::AbstractVector{Int64}, counts2::AbstractVector{Int64})::Float64
    @assert length(counts1) == length(counts2)

    freqs1 = to_pmf(counts1)
    freqs2 = to_pmf(counts2)

    return sum(freqs1 .* logx.(freqs1 ./ freqs2))

end

function kl_divergence(pmf1, pmf2)::Float64
    @assert length(pmf1) == length(pmf2)

    return sum(pmf1 .* logx.(pmf1 ./ pmf2))
end

function jensen_shannon_divergence(p::AbstractVector, q::AbstractVector)
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
J(p, q) = KL(p, q) + KL(q, p)
```
"""
function jeffreys_divergence(p, q)
    return kl_divergence(p, q) + kl_divergence(q, p)
end
