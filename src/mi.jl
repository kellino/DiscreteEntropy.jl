"""
    uncertainty_coefficient(counts::Matrix)

``math
C_{XY} = \frac{I(X;Y)}{H(Y)}
``
"""
function uncertainty_coefficient(counts::Matrix)::Float64
    0.0
end

"""
    redundancy(counts::Matrix)

``math
R = \frac{I(X;Y)}{H(X) + H(Y)}
``
"""
function redundancy(data::CountData, estimator::Function; k=data.K)::Float64
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
