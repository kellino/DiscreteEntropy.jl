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

@doc raw"""
    mutual_information(X::CountData, Y::CountData, XY::CountData, estimator::Function)::Float64

Returns the mutual information between X and Y, given their joint countdata $XY$, using
the specified estimator. Due to bias in the estimators, this might return a negative number.

```math
I(X;Y) = H(X) + H(Y) - H(X,Y)

```
"""
function mutual_information(X::CountData, Y::CountData, XY::CountData, estimator::Function)::Float64
    return estimator(X) + estimator(Y) - estimator(XY)
end

function mutual_information(X::CountData, Y::CountData, XY::CountData, ::Type{Maximum_Likelihood})::Float64
    return maximum_likelihood(X) + maximum_likelihood(Y) - maximum_likelihood(XY)
end

function mutual_information(X::CountData, Y::CountData, XY::CountData, ::Type{JackknifeML})::Float64
    return jackknife_ml(X) + jackknife_ml(Y) - jackknife_ml(XY)
end

function mutual_information(X::CountData, Y::CountData, XY::CountData, ::Type{MillerMadow})::Float64
    return miller_madow(X) + miller_madow(Y) - miller_madow(XY)
end

function mutual_information(X::CountData, Y::CountData, XY::CountData, ::Type{Grassberger})::Float64
    return grassberger(X) + grassberger(Y) - grassberger(XY)
end

function mutual_information(X::CountData, Y::CountData, XY::CountData, ::Type{Schurmann}; ξ::Float64=exp(-1 / 2))::Float64
    return schurmann(X, ξ) + schurmann(Y, ξ) - schurmann(XY, ξ)
end

function mutual_information(X::CountData, Y::CountData, XY::CountData, ::Type{SchurmannGeneralised}, xis::AbstractVector{Float64})::Float64
    return schurmann_generalised(X, xis) + schurmann_generalised(Y, xis) - schurmann_generalised(XY, xis)
end

function mutual_information(X::CountData, Y::CountData, XY::CountData, ::Type{ChaoShen})::Float64
    return chao_shen(X) + chao_shen(Y) - chao_shen(XY)
end

function mutual_information(X::CountData, Y::CountData, XY::CountData, ::Type{Zhang})::Float64
    return zhang(X) + zhang(Y) - zhang(XY)
end

function mutual_information(X::CountData, Y::CountData, XY::CountData, ::Type{Bonachela})::Float64
    return bonachela(X) + bonachela(Y) - bonachela(XY)
end

function mutual_information(X::CountData, Y::CountData, XY::CountData, ::Type{Bayes}, α::Float64)::Float64
    return bayes(α, X) + bayes(α, Y) - bayes(α, XY)
end

function mutual_information(X::CountData, Y::CountData, XY::CountData, ::Type{Jeffrey})::Float64
    return jeffrey(X) + jeffrey(Y) - jeffrey(XY)
end

function mutual_information(X::CountData, Y::CountData, XY::CountData, ::Type{LaPlace})::Float64
    return laplace(X) + laplace(Y) - laplace(XY)
end

function mutual_information(X::CountData, Y::CountData, XY::CountData, ::Type{SchurmannGrassberger})::Float64
    return schurmann_grassberger(X) + schurmann_grassberger(Y) - schurmann_grassberger(XY)
end

function mutual_information(X::CountData, Y::CountData, XY::CountData, ::Type{Minimax})::Float64
    return minimax(X) + minimax(Y) - minimax(XY)
end

function mutual_information(X::CountData, Y::CountData, XY::CountData, ::Type{NSB})::Float64
    return nsb(X) + nsb(Y) - nsb(XY)
end

function mutual_information(X::CountData, Y::CountData, XY::CountData, ::Type{NSB}; ks=ks::AbstractVector{Float64})::Float64
    @assert length(ks) == 3
    return nsb(X, k=ks[1]) + nsb(Y, k=ks[2]) - nsb(XY, k=ks[3])
end

function mutual_information(X::CountData, Y::CountData, XY::CountData, ::Type{ANSB})::Float64
    return ansb(X) + ansb(Y) - ansb(XY)
end

function mutual_information(X::CountData, Y::CountData, XY::CountData, ::Type{PYM})::Float64
    # TODO
    0.0
end
