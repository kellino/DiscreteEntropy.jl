@doc raw"""
    conditional_entropy(pmfX::AbstractVector{AbstractFloat}, pmfXY::AbstractVector{AbstractFloat})
    conditional_entropy(X::CountData, XY::CountData, estimator::Type{T}) where {T<:NonParameterisedEstimator}
    conditional_entropy(X::CountData, XY::CountData, estimator::Type{T}, param) where {T<:ParameterisedEstimator}

Compute the conditional entropy of Y conditioned on X

```math
H(Y \mid X) = - \sum_{x \in X, y \in Y} p(x, y) \ln \frac{p(x, y)}{p(x)}
```

Compute the estimated conditional entropy of Y given X, from counts of X, and (X,Y) and
estimator *estimator*

```math
\hat{H}(Y \mid X) = \hat{H}(X, Y) - \hat{H}(X)
```
 """
function conditional_entropy(pmfX::AbstractVector{AbstractFloat}, pmfXY::AbstractVector{AbstractFloat})
    @assert length(pmfX) == length(pmfXY)

    if sum(pmfX) != 1.0
        @warn("Normalising X")
        pmfX = to_pmf(pmfX)
    end
    if sum(pmfXY) != 1.0
        @warn("Normalising the joint probability distribution P(X, Y)")
        pmfXY = to_pmf(pmfXY)
    end

    -sum([pxy * logx(pxy / px) for (pxy, px) in collect(zip(pmfXY, pmfX))])
end

function conditional_entropy(X::CountData, XY::CountData, estimator::Type{T}) where {T<:NonParameterisedEstimator}
    estimate_h(XY, estimator) - estimate_h(X, estimator)
end
