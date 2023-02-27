using SpecialFunctions: loggamma

@doc raw"""
    logx(x)::Float64
Returns natural logarithm of x, or 0.0 if x is zero
"""
function logx(x)::Float64
    if iszero(x)
        return zero(x)
    end
    return log(x)
end

@doc raw"""
    xlogx(x::Float64)
Returns `x * log(x)` for `x ≥ 0`, or 0.0 if x is zero
"""
function xlogx(x)::Float64
    return x * logx(x)
end

@doc raw"""
    to_bits(x::Float64)
Return ``\frac{h}{\log(2)}`` where h is in nats
"""
function to_bits(h::Float64)::Float64
    return h / log(2)
end

@doc raw"""
    to_bans(x::Float64)
Return ``\frac{h}{log(10)}`` where `h` is in nats
"""
function to_bans(h::Float64)::Float64
    return h / log(10)
end

function basic_jack(xs::Vector{Int64})
    res = []
    push!(res, from_samples(xs))
    for i in 1:length(xs)
        out = vcat(xs[1:i-1], xs[i+1:length(xs)])
        push!(res, from_samples(out))
    end

    return res
end

function gammalndiff(x::Float64, dx::Float64)::Float64
    return loggamma(x + dx) - loggamma(x)
end


function logspace(start::Float64, stop::Float64, steps::Int64)::Vector{Float64}
    return 10 .^ range(start, stop, length=steps)
end

function update_or!(d::Dict, k, v)
    if k == 0
        return d
    end
    if haskey(d, k)
        d[k] += v
    else
        d[k] = v
    end
end

# @enum Axis x = 1 y = 2

function marginal_counts(joint::Matrix, dim)
    return vec(sum(joint, dims=dim))
end
