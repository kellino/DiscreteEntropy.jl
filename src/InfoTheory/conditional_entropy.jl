@doc raw"""
    conditional_entropy(X::CountData, XY::CountData, estimator::Type{T}) where {T<:NonParameterisedEstimator}
    conditional_entropy(joint::Matrix{R}, estimator::Type{NSB}; dim=1, guess=false, KJ=nothing, KX=nothing) where {R<:Real}
    conditional_entropy(joint::Matrix{R}, estimator::Type{Bayes}, α; dim=1, KJ=nothing, KX=nothing) where {R<:Real}

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
function conditional_entropy(X::CountData, XY::CountData, estimator::Type{T}) where {T<:NonParameterisedEstimator}
    estimate_h(XY, estimator) - estimate_h(X, estimator)
end

function conditional_entropy(joint::Matrix{R}, estimator::Type{T}; dim=1) where {T<:NonParameterisedEstimator, R<:Real}
    X = from_counts(marginal_counts(joint, dim))
    estimate_h(from_counts(cvector(joint)), estimator) - estimate_h(X, estimator)
end

function conditional_entropy(joint::Matrix{R}, estimator::Type{NSB}; dim=1, guess=false, KJ=nothing, KX=nothing) where {R<:Real}
    X = from_counts(marginal_counts(joint, dim))
    estimate_h(from_counts(cvector(joint)), estimator, guess=guess, K=KJ) - estimate_h(X, estimator, KX)
end

function conditional_entropy(joint::Matrix{R}, estimator::Type{Bayes}, α; dim=1, KJ=nothing, KX=nothing) where {R<:Real}
    X = from_counts(marginal_counts(joint, dim))
    estimate_h(from_counts(cvector(joint)), Bayes, α, K=KJ) - estimate_h(X, estimator, α, K=KX)
end
