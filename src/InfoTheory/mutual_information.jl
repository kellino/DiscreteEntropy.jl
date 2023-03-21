@doc raw"""
    uncertainty_coefficient(counts::Matrix, estimator::Type{T}; dim=Axis, symmetric=false) where {T<:AbstractEstimator}

Return the estimated uncertainty coefficient, or *Thiel's U* of `counts`, using estimator `estimator` along axis `dim`.

```math
C_X = \frac{I(X;Y)}{H(X)}
```

Return the symmetrical uncertainty coefficient if `symmetric` is `true`.
Do not set a value for `dim` is using `symmetric`: it will be ignored.

```math
2 \left(\frac{H(X) + H(Y) - I(X;Y)}{H(X) + H(Y)}\right)
```

# External Links
[Theil's U](https://en.wikipedia.org/wiki/Uncertainty_coefficient)
"""
function uncertainty_coefficient(contingency_matrix::Matrix, estimator::Type{T}; dim=Axis, symmetric=false) where {T<:AbstractEstimator}
    hx = estimate_h(from_data(marginal_counts(contingency_matrix, 2), Histogram), estimator)
    hy = estimate_h(from_data(marginal_counts(contingency_matrix, 1), Histogram), estimator)
    ixy = mutual_information(contingency_matrix, estimator)

    if symmetric
        return 2.0 * (hx + hy - ixy) / (hx + hy)
    else
        if dim == X
            return ixy / hx
        else
            return ixy / hy
        end
    end
end

function uncertainty_coefficient(X::CountData, Y::CountData, XY::CountData, estimator::Type{T}) where {T<:AbstractEstimator}
    hp = estimate_h(P, estimator)
    hxy = mutual_information(X, Y, XY, estimator)
end

@doc raw"""
    redundancy(data::CountData, estimator::Type{T}) where {T<:AbstractEstimator}

Return the estimated information redundancy of `data`, with `K` the sampled support size.

```math
R = \log(K) - \hat{H}(data)
```

    redundancy(data::CountData, estimator::Type{T}, K::Int64) where {T<:AbstractEstimator}

Return the estimated information redundancy of `data`, with `K` set by the user.

# External Links
[Redundancy (wikipedia)](https://en.wikipedia.org/wiki/Redundancy_(information_theory))

"""
function redundancy(data::CountData, estimator::Type{T}) where {T<:AbstractEstimator}
    log(data.K) - estimate_h(data, estimator)
end

function redundancy(data::CountData, estimator::Type{T}, K::Int64) where {T<:AbstractEstimator}
    log(K) - estimate_h(data, estimator)
end

@doc raw"""
    information_variation(X::CountData, Y::CountData, XY::CountData, H::Function)::Float64

Return the [Variation of Information](https://en.wikipedia.org/wiki/Variation_of_information). This
satisfies the properties of a metric (triangle inequality, non-negativity, indiscernability and symmetry).

```math
VI(X, Y) = 2 H(X, Y) - H(X) - H(Y)
"""
function information_variation(contingency_matrix::Matrix{T}, estimator::Type{E}) where {T<:Real,E<:AbstractEstimator}
    2.0 *
    # estimate_h(from_data(contingency_matrix), Histogram), estimator) -
    estimate_h(from_data(marginal_counts(contingency_matrix, 1), Histogram), estimator) -
    estimate_h(from_data(marginal_counts(contingency_matrix, 2), Histogram), estimator)
end

function information_variation(X::CountData, Y::CountData, XY::CountData, H::Function)
    return 2.0 * H(XY) - H(X) - H(Y)
end

@doc raw"""
    mutual_information(X::CountData, Y::CountData, XY::CountData, estimator::Function)::Float64
    mutual_information(X::CountData, Y::CountData, XY::CountData, estimator::Type{T}) where {T<:NonParameterisedEstimator}
    mutual_information(X::CountData, Y::CountData, XY::CountData, estimator::Type{T}, args...) where {T<:ParameterisedEstimator}

Compute the mutual information between X and Y, given their joint countdata $XY$, using
the specified estimator. Due to bias in the estimators, this might return a negative number.

```math
I(X;Y) = H(X) + H(Y) - H(X,Y)

```
"""
function mutual_information(contingency_matrix::Matrix, estimator::Type{T}) where {T<:AbstractEstimator}
    hx = estimate_h(from_data(marginal_counts(contingency_matrix, 2), Histogram), estimator)
    hy = estimate_h(from_data(marginal_counts(contingency_matrix, 1), Histogram), estimator)
    hxy = estimate_h(from_data(contingency_matrix, Histogram), estimator)

    hx + hy - hxy
end

function mutual_information(X::CountData, Y::CountData, XY::CountData, estimator::Type{T}) where {T<:NonParameterisedEstimator}
    estimate_h(X, estimator) + estimate_h(Y, estimator) - estimate_h(XY, estimator)
end

function mutual_information(X::CountData, Y::CountData, XY::CountData, estimator::Type{T}, args...) where {T<:ParameterisedEstimator}
    # The parameterised estimators so far
    # Schurmann, NSB, Bayes, PYM and SchurmmanGeneralised
    # SchurmannGeneralised is the odd one out, as the others just take a scalar
    # args... is unchecked at the moment. The user must ensure the correct types are passed in
    estimate_h(X, args[1], estimator) + estimate_h(Y, args[2], estimator) - estimate_h(XY, args[3], estimator)
end

function mutual_information(X::CountData, Y::CountData, XY::CountData,
    e1::Type{A}, e2::Type{B}, e3::Type{C}) where {A,B,C<:NonParameterisedEstimator}
    estimate_h(X, e1) + estimate_h(Y, e2) - estimate_h(XY, e3)
end
