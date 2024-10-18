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

The joint distribution, `XY` needs to be supplied by the user in the form of a `CountData` struct.

# Example
Calculating mutual information directly over a matrix
is the easiest way to use this function:

```@jldoctest
julia> m = Matrix([1 0; 0 1])
julia> i = mutual_information(m, MaximumLikelihood)
0.6931471805599453

julia> to_bits(i)
1.0
```


"""
function mutual_information(X::CountData, Y::CountData, XY::CountData, estimator::Type{T}) where {T<:AbstractEstimator}
  mi = estimate_h(X, estimator) + estimate_h(Y, estimator) - estimate_h(XY, estimator)
  if mi < 0.0
    return 0.0
  end
  mi
end

function mutual_information(joint::Matrix{I}, estimator::Type{T}) where {T<:AbstractEstimator,I<:Real}
  X = from_counts(marginal_counts(joint, 1))
  Y = from_counts(marginal_counts(joint, 2))
  XY = from_counts(vec(joint))

  mutual_information(X, Y, XY, estimator)
end

@doc raw"""
     uncertainty_coefficient(joint::Matrix{I}, estimator::Type{T}; symmetric=false) where {T<:AbstractEstimator, I<:Real}

Compute Thiel's uncertainty coefficient on 2 dimensional matrix `joint`, with  `estimator`, where `joint` is
the histogram of the joint distribution of two random variables ``(X;Y)``, and ``I(X;Y)`` is the (estimated)
mutual information.

```math
U(X \mid Y) =  \frac{I(X;Y)}{H(X)}
```

If `symmetric` is `true` then compute the weighted average between `X` and `Y`

```math
U(X, Y) = 2 \left[ \frac{H(X) + H(Y) - H(X, Y)} {H(X) + H(Y)} \right]
```

"""
function uncertainty_coefficient(joint::Matrix{I}, estimator::Type{T}; symmetric=false) where {T<:AbstractEstimator,I<:Real}
  if symmetric
    X = marginal_counts(joint, 1)
    Y = marginal_counts(joint, 2)
    hx = estimate_h(from_data(X, Histogram), estimator)
    hy = estimate_h(from_data(Y, Histogram), estimator)
    hxy = estimate_h(from_data(joint, Histogram), estimator)
    2 * ((hx + hy - hxy) / (hx + hy))
  else
    mutual_information(joint, estimator) / estimate_h(from_data(marginal_counts(joint, 1), Histogram), estimator)
  end
end
