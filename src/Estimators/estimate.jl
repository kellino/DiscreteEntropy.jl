using Distributions: Distribution

"""
    AbstractEstimator

Supertype for NonParameterised and Parameterised entropy estimators.
"""
abstract type AbstractEstimator end

"""
    NonParameterisedEstimator

Type for NonParameterised  entropy estimators.
"""
abstract type NonParameterisedEstimator <: AbstractEstimator end

"""
    ParameterisedEstimator

Type for Parameterised  entropy estimators.
"""
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
struct SchurmannGeneralised <: ParameterisedEstimator end
struct BUB <: ParameterisedEstimator end

# Bayesian with Parameter(s)
struct Bayes <: ParameterisedEstimator end

struct NSB <: ParameterisedEstimator end

struct PYM <: ParameterisedEstimator end

struct AutoNSB <: NonParameterisedEstimator end
struct ANSB <: NonParameterisedEstimator end
struct Jeffrey <: NonParameterisedEstimator end
struct LaPlace <: NonParameterisedEstimator end
struct SchurmannGrassberger <: NonParameterisedEstimator end
struct Minimax <: NonParameterisedEstimator end


# Other
struct PERT <: AbstractEstimator end
struct Bootstrap end


@doc raw"""
    estimate_h(data::CountData, estimator::Type{T}) where {T<:AbstractEstimator}
    estimate_h(data::CountData, ::Type{JackknifeMLE}; corrected=false)
    estimate_h(data::CountData, ::Type{Schurmann}, xi=nothing)

    estimate_h(data::CountVector, ::Type{SchurmannGeneralised}, xis::XiVector)
    estimate_h(data::CountData, ::Type{Bayes}, α::AbstractFloat; K=nothing)
    estimate_h(data::CountData, ::Type{NSB}; guess=false, K=nothing)

Return the estimate in nats of Shannon entropy of `data` using `estimator`.


# Example

```@jldoctest
julia> import Random; Random.seed!(1);

julia> X = rand(1:10, 1000)
julia> estimate_h(from_data(X, Samples), Schurmann)
2.3039615201251173
```

## Note
While most calls to estimate_h take a CountData struct, this is not true for every estimator, especially those that
work directly over samples, or need the original structure of the histogram.

For a complete list of methods of the function, try

```@jldoctest
julia> methods(estimate_h)
```

This function is a wrapper indended to make using the libary easier. For finer control over some of the estimators,
it is advisable to call them directly, rather than through this function.

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

function estimate_h(data::CountData, ::Type{BUB}; k_max=11, truncate=false, lambda=0.0)
    (h, _ ) = bub(data, k_max=k_max, truncate=truncate, lambda=lambda)
    return h
end

function estimate_h(data::CountData, ::Type{Bayes}, α::AbstractFloat; K=nothing)
    bayes(data, α, K=K)
end

function estimate_h(data::CountData, ::Type{LaPlace}; K=nothing)
    laplace(data, K=K)
end

function estimate_h(data::CountData, ::Type{Jeffrey}; K=nothing)
    jeffrey(data, K=K)
end

function estimate_h(data::CountData, ::Type{SchurmannGrassberger}; K=nothing)
    schurmann_grassberger(data, K=K)
end

function estimate_h(data::CountData, ::Type{Minimax}; K=nothing)
    minimax(data, K=K)
end

function estimate_h(data::CountData, ::Type{PYM}; param=nothing)
    pym(data; param=param)
end

function estimate_h(data::CountData, ::Type{NSB}; guess=false, K=nothing)
    loc_K = K
    if guess
        loc_K = guess_k(data)
    else
        if loc_K === nothing
            loc_K = data.K
        end
    end
    nsb(data, loc_K)
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

@doc raw"""
    pert(data::CountData, estimator::Type{T}) where {T<:AbstractEstimator}
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

where the default estimators are: a = maximum_likelihood, c = ANSB and $b$ is the most likely value = ChaoShen
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

function estimate_h(data::SampleVector, estimator::Type{T}, ::Type{Bootstrap}; seed=1, reps=1000, concentration=4) where {T<:AbstractEstimator}
    bayesian_bootstrap(data, estimator, reps, seed, concentration)[1]
end
