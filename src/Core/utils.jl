using SpecialFunctions: loggamma

@enum Axis X = 2 Y = 1

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

function xFx(f::Function, x)
    if iszero(x)
        return zero(x)
    else
        return f(x)
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

function gammalndiff(x::Float64, dx::Float64)
    return loggamma(x + dx) - loggamma(x)
end

function update_or_insert!(d::Dict, k, v)
    if k == 0
        return d
    end
    if haskey(d, k)
        d[k] += v
    else
        d[k] = v
    end
end

@doc raw"""
    marginal_counts(contingency_matrix::Matrix, dim)

Return the *unnormalised* marginal counts of `contingency_matrix` along dimension `dim`.

"""
function marginal_counts(joint::Matrix, dim)
    if dim == 1
        return [sum(x) for x in eachrow(joint)]
    end

    if dim != 2
        @warn("unexpected dimension, returning dim=2")
    end

    return [sum(x) for x in eachcol(joint)]

end

function logspace(start, stop, steps::Integer)
    return 10 .^ range(start, stop, length=steps)
end

function round_data(data::Float64)
    return round(data; digits=4)
end 

function print_data(arg1::String, arg2::Float64)
    println(arg1 * " " * string(round_data(arg2)))
end

function print_data(arg1, arg2::Float64)
    println(string(arg1) * " " * string(round_data(arg2)))
end

function print_data(data::Float64)
    println(round_data(data))
end

function print_data(data::Int)
    println(data)
end
