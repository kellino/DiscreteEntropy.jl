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

#
# TODO it might be possible to rewrite all of this using macros. cleaner, easier to maintain code
#

function mutual_information(X::CountData, Y::CountData, XY::CountData, ::Type{Maximum_Likelihood})
    return maximum_likelihood(X) + maximum_likelihood(Y) - maximum_likelihood(XY)
end

function mutual_information(X::CountData, Y::CountData, XY::CountData, ::Type{JackknifeML})
    return jackknife_ml(X) + jackknife_ml(Y) - jackknife_ml(XY)
end

function mutual_information(X::CountData, Y::CountData, XY::CountData, ::Type{MillerMadow})
    return miller_madow(X) + miller_madow(Y) - miller_madow(XY)
end

function mutual_information(X::CountData, Y::CountData, XY::CountData, ::Type{Grassberger})
    return grassberger(X) + grassberger(Y) - grassberger(XY)
end

function mutual_information(X::CountData, Y::CountData, XY::CountData, ::Type{Schurmann}; ξ::Float64=exp(-1 / 2))
    return schurmann(X, ξ) + schurmann(Y, ξ) - schurmann(XY, ξ)
end

function mutual_information(X::CountData, Y::CountData, XY::CountData, ::Type{SchurmannGeneralised}, xis::AbstractVector{Float64})
    return schurmann_generalised(X, xis) + schurmann_generalised(Y, xis) - schurmann_generalised(XY, xis)
end

function mutual_information(X::CountData, Y::CountData, XY::CountData, ::Type{ChaoShen})
    return chao_shen(X) + chao_shen(Y) - chao_shen(XY)
end

function mutual_information(X::CountData, Y::CountData, XY::CountData, ::Type{Zhang})
    return zhang(X) + zhang(Y) - zhang(XY)
end

function mutual_information(X::CountData, Y::CountData, XY::CountData, ::Type{Bonachela})
    return bonachela(X) + bonachela(Y) - bonachela(XY)
end

function mutual_information(X::CountData, Y::CountData, XY::CountData, ::Type{Bayes}, α::Float64)
    return bayes(α, X) + bayes(α, Y) - bayes(α, XY)
end

function mutual_information(X::CountData, Y::CountData, XY::CountData, ::Type{Jeffrey})
    return jeffrey(X) + jeffrey(Y) - jeffrey(XY)
end

function mutual_information(X::CountData, Y::CountData, XY::CountData, ::Type{LaPlace})
    return laplace(X) + laplace(Y) - laplace(XY)
end

function mutual_information(X::CountData, Y::CountData, XY::CountData, ::Type{SchurmannGrassberger})
    return schurmann_grassberger(X) + schurmann_grassberger(Y) - schurmann_grassberger(XY)
end

function mutual_information(X::CountData, Y::CountData, XY::CountData, ::Type{Minimax})
    return minimax(X) + minimax(Y) - minimax(XY)
end

function mutual_information(X::CountData, Y::CountData, XY::CountData, ::Type{NSB})
    return nsb(X) + nsb(Y) - nsb(XY)
end

function mutual_information(X::CountData, Y::CountData, XY::CountData, ::Type{NSB}, ks::AbstractVector{Float64})
    @assert length(ks) == 3
    return nsb(X, k=ks[1]) + nsb(Y, k=ks[2]) - nsb(XY, k=ks[3])
end

function mutual_information(X::CountData, Y::CountData, XY::CountData, ::Type{ANSB})
    return ansb(X) + ansb(Y) - ansb(XY)
end

function mutual_information(X::CountData, Y::CountData, XY::CountData, ::Type{PYM})
    # TODO
    0.0
end

function mutual_information(X::CountData, Y::CountData, XY::CountData,
    e1::Type{T}, e2::Type{T2}, e3::Type{T3}) where {T<:Estimator,T2<:Estimator,T3<:Estimator}
    # TODO figure out how to get this to work
    return e1(X) + e2(Y) - e3(XY)
end

function mutual_information(counts::Matrix, args...)::Float64
    X = from_counts(marginal_counts(counts, 1))
    Y = from_counts(marginal_counts(counts, 2))
    XY = from_counts([1, 2, 3]) # TODO need to implement this

    return mutual_information(X, Y, XY, args...)
end
