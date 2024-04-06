using SpecialFunctions: loggamma
using StatsBase: countmap;

# @enum Axis X = 2 Y = 1

@doc raw"""
    logx(x)::Float64
Returns natural logarithm of x, or 0.0 if x is zero
"""
function logx(x)
    if iszero(x)
        return zero(x)
    end
    return log(x)
end

@doc raw"""
    xlogx(x::Float64)
Returns `x * log(x)` for `x ≥ 0`, or 0.0 if x is zero
"""
function xlogx(x)
    return x * logx(x)
end

@doc raw"""
     xFx(f::Function, x)

Returns `x * f(x)` for `x ≥ 0`, or 0.0 if x is zero
"""
function xFx(f::Function, x)
    if iszero(x)
        return zero(x)
    else
        return x * f(x)
    end
end

@doc raw"""
    to_bits(x::Float64)
Return ``\frac{h}{\log(2)}`` where h is in nats
"""
function to_bits(h::Float64)
    return h / log(2)
end

@doc raw"""
    to_bans(x::Float64)
Return ``\frac{h}{log(10)}`` where `h` is in nats
"""
function to_bans(h::Float64)
    return h / log(10)
end

@doc raw"""
     gammalndiff(x::Float64, dx::Float64)

Return ``\log \Gamma(x + δx) - \log Γ(x)``

"""
function gammalndiff(x::Float64, dx::Float64)
    return loggamma(x + dx) - loggamma(x)
end

@doc raw"""
    marginal_counts(contingency_matrix::Matrix, dim; normalise=false)

Return the marginal counts of `contingency_matrix` along dimension `dim`.

If normalised = true, return as probability distribution.

"""
function marginal_counts(joint::Matrix, dim; normalise=false)
    p = nothing
    if dim == 1
        p = [sum(x) for x in eachrow(joint)]
    end
    if dim == 2
        p = [sum(x) for x in eachcol(joint)]
    end

    if p === nothing
        return p
    else
        if normalise
            return p ./ sum(p)
        end
    end

    return p

end

function logspace(start, stop, steps::Integer)
    return 10 .^ range(start, stop, length=steps)
end

function update_dict!(d, k; v=1)
    if haskey(d, k)
        d[k] += v
    else
        d[k] = v
    end
end
