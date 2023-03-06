using Distributions: Distribution

"""
    AbstractEstimator

Supertype for NonParameterised and Parameterised entropy estimators.
"""
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
struct Shrink <: NonParameterisedEstimator end

# Frequentist with Parameter(s)
struct Schurmann <: ParameterisedEstimator end
struct SchurmannGeneralised <: ParameterisedEstimator end

# Bayesian with Parameter(s)
struct Bayes <: ParameterisedEstimator end
struct NSB <: ParameterisedEstimator end
struct PYM <: ParameterisedEstimator end

struct ANSB <: NonParameterisedEstimator end
struct Jeffrey <: NonParameterisedEstimator end
struct LaPlace <: NonParameterisedEstimator end
struct SchurmannGrassberger <: NonParameterisedEstimator end
struct Minimax <: AbstractEstimator end



"""
    estimate_h(data::CountData, ::Type{T}) where {T<:AbstractEstimator}
"""
function estimate_h(data::CountData, ::Type{Maximum_Likelihood})
    maximum_likelihood(data)
end

# function estimate_h(data::CountData, ::Type{JackknifeML})
#     jackknife_ml(data)[1]
# end

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
    schurmann_generalised(data, xis)
end

# function estimate_h(data::CountData, ::Type{ChaoShen})
#     chao_shen(data)
# end

# function estimate_h(data::CountData, ::Type{Zhang})
#     zhang(data)
# end

# function estimate_h(data::CountData, ::Type{Shrink})
#     shrink(data)
# end
#
# function estimate_h(data::CountData, ::Type{Bonachela})
#     bonachela(data)
# end

# function estimate_h(data::CountData, ::Type{Bayes}, α::AbstractFloat)
#     bayes(α, data)
# end

# function estimate_h(data::CountData, ::Type{LaPlace})
#     laplace(data)
# end

# function estimate_h(data::CountData, ::Type{Jeffrey})
#     jeffrey(data)
# end

# function estimate_h(data::CountData, ::Type{SchurmannGrassberger})
#     schurmann_grassberger(data)
# end

# function estimate_h(data::CountData, ::Type{Minimax})
#     minimax(data)
# end

# function estimate_h(data::CountData, ::Type{Schurmann}, ξ=nothing)
#     if ξ === nothing
#         schurmann(data)
#     else
#         schurmann(data, ξ)
#     end
# end

# function estimate_h(data::CountData, ::Type{SchurmannGeneralised}, xis::AbstractVector)
#     schurmann_generalised(data, xis)
# end

# function estimate_h(data::CountData, ::Type{NSB}, K=nothing)
#     nsb(data, K)
# end

# # function estimate_h(data::CountData, ::Type{PYM}, param=nothing)
# #     @warn("not yet finished")
# #     0.0
# # end
