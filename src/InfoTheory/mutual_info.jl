@doc raw"""
     mutual_information(X::CountData, Y::CountData, XY::CountData, estimator::Type{T}) where {T<:AbstractEstimator}
     mutual_information(joint::Matrix{I}, estimator::Type{T}) where {T<:AbstractEstimator, I<:Real}

```math
I(X;Y) = \sum_{y \in Y} \sum_{x \in X} p(x, y) \log \left(\frac{p_{X,Y}(x,y)}{p_X(x) p_Y(y)}\right)
```

But we use the identity

```math
I(X;Y) = H(X) + H(Y) - H(X,Y)
```
where ``H(X,Y)`` is the entropy of the joint distribution

"""
function mutual_information(X::CountData, Y::CountData, XY::CountData, estimator::Type{T}) where {T<:AbstractEstimator}
    mi = estimate_h(X, estimator) + estimate_h(Y, estimator) - estimate_h(XY, estimator)
    if mi < 0.0
        return 0.0
    end
    mi
end

function mutual_information(joint::Matrix{I}, estimator::Type{T}) where {T<:AbstractEstimator, I<:Real}
    X = from_counts(marginal_counts(joint, 1))
    Y = from_counts(marginal_counts(joint, 2))
    XY = from_counts(vec(joint))

    mutual_information(X, Y, XY, estimator)
end
