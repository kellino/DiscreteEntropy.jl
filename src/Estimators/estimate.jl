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
struct Grassberger88 <: NonParameterisedEstimator end
struct Grassberger03 <: NonParameterisedEstimator end
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

"""
    estimate_h(data::CountData, estimator::Type{T}) where {T<:AbstractEstimator}

Return the estimate in nats of Shannon entropy of `data` using `estimator`.

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

function estimate_h(data::CountData, ::Type{Grassberger88})
    grassberger1988(data)
end

function estimate_h(data::CountData, ::Type{Grassberger03})
    grassberger2003(data)
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
    schurmann_generalised(data, xis)
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

function estimate_h(data::CountData, ::Type{Bayes}, α::AbstractFloat; K=data.K)
    bayes(α, data, K)
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

function estimate_h(data::CountData, ::Type{Minimax}; K=data.K)
    minimax(data, K)
end

function estimate_h(data::CountData, ::Type{NSB}, k)
    nsb(data, k)
end

function estimate_h(data::CountData, ::Type{AutoNSB})
    nsb(data, guess_k(data))
end

function estimate_h(data::CountData, ::Type{PYM}, param=nothing)
    #@warn("not yet finished")
    #0.0
    mm = data.multiplicities[2:2, :]
    icts = data.multiplicities[1:1, :]
    mm = vec(round.(Int, mm))
    icts = vec(round.(Int, icts))
    icts = sort(icts)

    (Hbls, Hvar) = pym(mm, icts)

    return Hbls
end

function estimate_h(data::CountData, ::Type{ANSB})
    ansb(data)
end

function estimate_h(data::CountData, ::Type{PERT})
    # TODO no reason to prefer ChaoShen here
    pert(data, ChaoShen)
end

@doc raw"""
    pert(data::CountData, estimator)

A Pert estimate of entropy, where

```
H = \frac{a + 4b + c}{6}
```

where a is the minimum (maximum_likelihood), c is the maximum (log(k)) and $b$ is the most likely value (ChaoShen)
"""
function pert(data::CountData, estimator::Type{T}) where {T<:AbstractEstimator}
    return (estimate_h(data, MaximumLikelihood) + 4 * estimate_h(data, estimator) + estimate_h(data, ANSB)) / 6.0
end
