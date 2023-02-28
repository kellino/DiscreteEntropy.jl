@doc raw"""
     conditional_entropy(pmfX::AbstractVector{AbstractFloat}, pmfXY::AbstractVector{AbstractFloat})

 Compute the conditional entropy of Y conditioned on X

 ```math
 H(Y \mid X) = - \sum_{x \in X, y \in Y} p(x, y) \ln \frac{p(x, y)}{p(x)}
 ```

 """
function conditional_entropy(pmfX::AbstractVector{AbstractFloat}, pmfXY::AbstractVector{AbstractFloat})
    @assert length(pmfX) == length(pmfXY)

    if sum(pmfX) != 0.0
        @warn("Normalising X")
        pmfX = to_pmf(pmfX)
    end
    if sum(pmfXY) != 0.0
        @warn("Normalising the joint probability distribution P(X, Y)")
        pmfXY = to_pmf(pmfXY)
    end

    -sum([pxy * logx(pxy / px) for (pxy, px) in collect(zip(pmfXY, pmfX))])
end
