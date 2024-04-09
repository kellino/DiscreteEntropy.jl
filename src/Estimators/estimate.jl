using Distributions: Distribution

"""
    AbstractEstimator

Supertype for NonParameterised and Parameterised entropy estimators.
"""
abstract type AbstractEstimator end

abstract type NonParameterisedEstimator <: AbstractEstimator end
abstract type ParameterisedEstimator <: AbstractEstimator end

# Frequentist
struct MaximumLikelihood <: NonParameterisedEstimator end
struct JackknifeMLE <: NonParameterisedEstimator end
struct MillerMadow <: NonParameterisedEstimator end
struct Grassberger <: NonParameterisedEstimator end
struct ChaoShen <: NonParameterisedEstimator end
struct Zhang <: NonParameterisedEstimator end
struct Bonachela <: NonParameterisedEstimator end
struct Shrink <: NonParameterisedEstimator end
struct ChaoWangJost <: NonParameterisedEstimator end

# Frequentist with Parameter(s)
struct Schurmann <: ParameterisedEstimator end
struct BUB <: ParameterisedEstimator end
struct SchurmannGeneralised <: ParameterisedEstimator end

# Bayesian with Parameter(s)
struct Bayes <: ParameterisedEstimator end

struct NSB <: ParameterisedEstimator end

struct PYM <: ParameterisedEstimator end

struct AutoNSB <: NonParameterisedEstimator end
struct ANSB <: NonParameterisedEstimator end
struct Jeffrey <: NonParameterisedEstimator end
struct LaPlace <: NonParameterisedEstimator end
struct SchurmannGrassberger <: NonParameterisedEstimator end
struct Minimax <: AbstractEstimator end


# Other
struct PERT <: AbstractEstimator end


@doc raw"""
    estimate_h(data::CountData, estimator::Type{T}) where {T<:AbstractEstimator}

Return the estimate in nats of Shannon entropy of `data` using `estimator`.

Wrapper function indended to make using the libary easier.
"""
function estimate_h(data::CountData, ::Type{MaximumLikelihood})
    maximum_likelihood(data)
end

function estimate_h(data::CountData, ::Type{JackknifeMLE}; corrected=false)
    jackknife_mle(data; corrected)[1]
end

function estimate_h_and_var(data::CountData, ::Type{JackknifeMLE}; corrected=false)
    jackknife_mle(data; corrected)
end

function estimate_h(data::CountData, ::Type{MillerMadow})
    miller_madow(data)
end

function estimate_h(data::CountData, ::Type{Grassberger})
    grassberger(data)
end

function estimate_h(data::CountData, ::Type{Schurmann}, xi=nothing)
    if xi === nothing
        schurmann(data)
    else
        schurmann(data, xi)
    end
end

function estimate_h(data::CountVector, ::Type{SchurmannGeneralised}, xis::XiVector)
    schurmann_generalised(data, xis)
end

function estimate_h(data::CountVector, ::Type{SchurmannGeneralised}, xis::Distribution)
    schurmann_generalised(data, xis, true)
end

function estimate_h(data::CountData, ::Type{ChaoShen})
    chao_shen(data)
end

function estimate_h(data::CountData, ::Type{Zhang})
    zhang(data)
end

function estimate_h(data::CountData, ::Type{Shrink})
    shrink(data)
end

function estimate_h(data::CountData, ::Type{Bonachela})
    bonachela(data)
end

function estimate_h(data::CountData, ::Type{ChaoWangJost})
    chao_wang_jost(data)
end

function estimate_h(data::CountData, ::Type{BUB})
    bub(data)
end

function estimate_h(data::CountData, ::Type{Bayes}, α::AbstractFloat; K=nothing)
    if K === nothing
        K = data.K
    end
    bayes(data, α, K)
end

function estimate_h(data::CountData, ::Type{LaPlace}; K=data.K)
    laplace(data, K)
end

function estimate_h(data::CountData, ::Type{Jeffrey}; K=data.K)
    jeffrey(data, K)
end

function estimate_h(data::CountData, ::Type{SchurmannGrassberger}; K=data.K)
    schurmann_grassberger(data, K)
end

function estimate_h(data::CountData, ::Type{Minimax}; K=nothing)
    if K === nothing
        minimax(data, data.K)
    else
        minimax(data, K)
    end
end

function estimate_h(data::CountData, ::Type{PYM}; param=nothing)
    pym(data; param=param)
end

function estimate_h(data::CountData, ::Type{NSB}; guess=false, K=nothing)
    if guess
        K = guess_k(data)
    else
        if K === nothing
            K = data.K
        end
    end
    nsb(data, K)
end

function estimate_h(data::CountData, ::Type{ANSB})
    ansb(data)
end

function estimate_h(data::CountData, ::Type{PERT})
    # TODO no reason to prefer ChaoShen here
    pert(data, ChaoShen)
end

@enum Pert_Type begin
    Pert
    Triangular
end

# E = (a + m + b) / 3.
@doc raw"""
    pert(data::CountData, estimator)
    pert(data::CountData, e1::Type{T}, e2::Type{T}) where {T<:AbstractEstimator}

A Pert estimate of entropy, where

```
a = best estimate
b = most likely estimate
c = worst case estimate
```

```
H = \frac{a + 4b + c}{6}
```

where a is the minimum (maximum_likelihood), c is the maximum (log(k)) and $b$ is the most likely value, but default ChaoShen
"""
function pert(data::CountData, estimator::Type{T}) where {T<:AbstractEstimator}
    return (estimate_h(data, MaximumLikelihood) + 4 * estimate_h(data, estimator) + estimate_h(data, ANSB)) / 6.0
end

function pert(data::CountData, e1::Type{T1}, e2::Type{T2}) where {T1, T2 <:AbstractEstimator}
    return (estimate_h(data, MaximumLikelihood) + 4 * estimate_h(data, e1) + estimate_h(data, e2)) / 6.0
end

function estimate_h(data::CountData, ::Type{PERT}, e::Type{T}) where {T<:AbstractEstimator}
    pert(data, e)
end

function estimate_h(data::CountData, ::Type{PERT}, e1::Type{T1}, e2::Type{T2}) where {T1, T2<:AbstractEstimator}
    if typeof(e1) == PERT || typeof(e2) == PERT
        @warn("argument error")
        return
    end
    pert(data, e1, e2)
end
