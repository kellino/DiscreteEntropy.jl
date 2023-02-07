using SpecialFunctions
using QuadGK: quadgk

@inline function ξ(β::Float64, k::Int64)::BigFloat
    return digamma(k * β + 1) - digamma(β + 1)
end

function ρ(β::Float64, k::Int64, data::CountData)
    return gamma(k * ξ(β, k)) / gamma(data.N + k * ξ(β, k)) *
           prod([gamma(y + BigFloat(β)) / gamma(β) * c for (y, c) in data.histogram])

end

@doc raw"""
    nsb(k, data::CountData)
    nsb(data::CountData)
    nsb(samples::AbstractVector)

    Bayesian estimator
"""
function nsb(k::Int64, data::CountData)::Float64

    if data.N <= 1
        error("Too few samples")
    end

    return quadgk(β -> ρ(β, k, data) * bayes(β, data), 1e-10, log(k))[1] /
           quadgk(β -> ρ(β, k, data), 1e-10, log(k))[1]
end

function nsb(data::CountData)::Float64
    nsb(data.K, data)
end

function nsb(samples::AbstractVector)::Float64
    nsb(from_samples(samples))
end
