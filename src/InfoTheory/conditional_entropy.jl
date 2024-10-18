@doc raw"""
    conditional_entropy(X::CountData, XY::CountData, estimator::Type{T}) where {T<:NonParameterisedEstimator}
    conditional_entropy(joint::Matrix{R}, estimator::Type{NSB}; dim=1, guess=false, KJ=nothing, KX=nothing) where {R<:Real}
    conditional_entropy(joint::Matrix{R}, estimator::Type{Bayes}, α; dim=1, KJ=nothing, KX=nothing) where {R<:Real}

Compute the conditional entropy of Y conditioned on X

```math
H(Y \mid X) = - \sum_{x \in X, y \in Y} p(x, y) \ln \frac{p(x, y)}{p(x)}
```

Compute the estimated conditional entropy of Y given X, from counts of X, and (X,Y) and
`estimator`

```math
\hat{H}(Y \mid X) = \hat{H}(X, Y) - \hat{H}(X)
```

# Example
```@jldoctest
julia> m = [4 9 4 5 8; 10 2 7 9 6; 3 5 6 9 6; 4 2 1 5 8; 4 5 43 8 3]
julia> X = DiscreteEntropy.marginal_counts(m, 1)

julia> conditional_entropy(from_data(X, Histogram), from_data,(m, Histogram), Zhang)
1.395955392163378

julia> conditional_entropy(m, Zhang)
1.395955392163378
```

 """
function conditional_entropy(X::CountData, XY::CountData, estimator::Type{T}) where {T<:NonParameterisedEstimator}
  estimate_h(XY, estimator) - estimate_h(X, estimator)
end

function conditional_entropy(joint::Matrix{R}, estimator::Type{T}; dim=1) where {T<:NonParameterisedEstimator,R<:Real}
  X = from_counts(marginal_counts(joint, dim))
  estimate_h(from_counts(cvector(joint)), estimator) - estimate_h(X, estimator)
end

function conditional_entropy(joint::Matrix{R}, ::Type{NSB}; dim=1, guess=false, KJ=nothing, KX=nothing) where {R<:Real}
  X = from_counts(marginal_counts(joint, dim))
  estimate_h(from_data(joint, Histogram), NSB, guess=guess, K=KJ) - estimate_h(X, NSB, K=KX)
end

function conditional_entropy(joint::Matrix{R}, estimator::Type{Bayes}, α; dim=1, KJ=nothing, KX=nothing) where {R<:Real}
  X = from_counts(marginal_counts(joint, dim))
  estimate_h(from_counts(cvector(joint)), Bayes, α, K=KJ) - estimate_h(X, estimator, α, K=KX)
end
