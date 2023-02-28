"""
    uncertainty_coefficient(counts::Matrix)

``math
C_{XY} = \frac{I(X;Y)}{H(Y)}
``
"""
function uncertainty_coefficient(counts::Matrix)
    0.0
end

"""
    redundancy(counts::Matrix)

``math
R = \frac{I(X;Y)}{H(X) + H(Y)}
``
"""
function redundancy(data::CountData, estimator::Function; k=data.K)
    0.0
end


@doc raw"""
    information_variation(X::CountData, Y::CountData, XY::CountData, H::Function)::Float64

Returns the [Variation of Information](https://en.wikipedia.org/wiki/Variation_of_information). This
satisfies the properties of a metric (triangle inequality, non-negativity, indiscernability and symmetry).

```math
VI(X, Y) = 2 H(X, Y) - H(X) - H(Y)
"""
function information_variation(X::CountData, Y::CountData, XY::CountData, H::Function)
    return 2.0 * H(XY) - H(X) - H(Y)
end

@doc raw"""
    mutual_information(X::CountData, Y::CountData, XY::CountData, estimator::Function)::Float64

Returns the mutual information between X and Y, given their joint countdata $XY$, using
the specified estimator. Due to bias in the estimators, this might return a negative number.

```math
I(X;Y) = H(X) + H(Y) - H(X,Y)

```
"""
function mutual_information(X::CountData, Y::CountData, XY::CountData, estimator::Function)
    return estimator(X) + estimator(Y) - estimator(XY)
end

function mutual_information(X::CountData, Y::CountData, XY::CountData, estimator::Type{T}) where {T<:NonParameterisedEstimator}
    entropy(X, estimator) + entropy(Y, estimator) - entropy(XY, estimator)
end

function mutual_information(X::CountData, Y::CountData, XY::CountData, estimator::Type{T}, args...) where {T<:ParameterisedEstimator}
    # The parameterised estimators so far
    # Schurmann, NSB, Bayes, PYM and SchurmmanGeneralised
    # SchurmannGeneralised is the odd one out, as the others just take a scalar
    # args... is unchecked at the moment. The user must ensure the correct types are passed in
    entropy(X, args[1], estimator) + entropy(Y, args[2], estimator) - entropy(XY, args[3], estimator)
    # println(param)
    # 0.0

end

# function mutual_information(X::CountData, Y::CountData, XY::CountData,
#     e1::Type{T}, e2::Type{T2}, e3::Type{T3}) where {T<:Estimator,T2<:Estimator,T3<:Estimator}
#     # TODO figure out how to get this to work
#     return e1(X) + e2(Y) - e3(XY)
# end

# function mutual_information(counts::Matrix, args...)::Float64
#     X = from_counts(marginal_counts(counts, 1))
#     Y = from_counts(marginal_counts(counts, 2))
#     XY = from_counts([1, 2, 3]) # TODO need to implement this

#     return mutual_information(X, Y, XY, args...)
# end
