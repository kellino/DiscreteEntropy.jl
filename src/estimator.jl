
abstract type AbstractEstimator end

abstract type NonParameterisedEstimator <: AbstractEstimator end
abstract type ParameterisedEstimator <: AbstractEstimator end

# Frequentist
struct Maximum_Likelihood <: NonParameterisedEstimator end
struct JackknifeML <: NonParameterisedEstimator end
struct MillerMadow <: NonParameterisedEstimator end
struct Grassberger <: NonParameterisedEstimator end
struct ChaoShen <: NonParameterisedEstimator end
struct Zhang <: NonParameterisedEstimator end
struct Bonachela <: NonParameterisedEstimator end

struct Schurmann <: ParameterisedEstimator end
struct SchurmannGeneralised <: ParameterisedEstimator end

# Bayesian
struct Bayes <: ParameterisedEstimator end
struct NSB <: ParameterisedEstimator end
struct PYM <: ParameterisedEstimator end

struct ANSB <: NonParameterisedEstimator end
struct Jeffrey <: NonParameterisedEstimator end
struct LaPlace <: NonParameterisedEstimator end
struct SchurmannGrassberger <: NonParameterisedEstimator end
struct Minimax <: AbstractEstimator end


function entropy(data::CountData, ::Type{Maximum_Likelihood})
    maximum_likelihood(data)
end

function entropy(data::CountData, ::Type{JackknifeML})
    jackknife_ml(data)[1]
end

function entropy(data::CountData, ::Type{MillerMadow})
    miller_madow(data)
end

function entropy(data::CountData, ::Type{Grassberger})
    grassberger(data)
end

function entropy(data::CountData, ::Type{SchurmannGeneralised}; xis::AbstractVector{AbstractFloat})
    schurmann(data, xis)
end

function entropy(data::CountData, ::Type{ChaoShen})
    chao_shen(data)
end

function entropy(data::CountData, ::Type{Zhang})
    zhang(data)
end

function entropy(data::CountData, ::Type{Bonachela})
    bonachela(data)
end

function entropy(data::CountData, ::Type{Bayes}, α::AbstractFloat)
    bayes(α, data)
end

function entropy(data::CountData, ::Type{LaPlace})
    laplace(data)
end

function entropy(data::CountData, ::Type{Jeffrey})
    jeffrey(data)
end

function entropy(data::CountData, ::Type{SchurmannGrassberger})
    schurmann_grassberger(data)
end

function entropy(data::CountData, ::Type{Minimax})
    minimax(data)
end

function entropy(data::CountData, ::Type{Schurmann}, ξ=nothing)
    if ξ === nothing
        schurmann(data)
    else
        schurmann(data, ξ)
    end
end

function entropy(data::CountData, ::Type{SchurmannGeneralised}, xis::AbstractVector)
    schurmann_generalised(data, xis)
end

function entropy(data::CountData, ::Type{NSB}, K=nothing)
    nsb(data, K)
end

function entropy(data::CountData, ::Type{PYM}, param=nothing)
    @warn("not yet finished")
    0.0
end
