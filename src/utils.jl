@doc raw"""
    xlogx(x::Float64)
Return `x * log(x)` for `x ≥ 0`, handling `x == 0` by return 0.
"""
function xlogx(x)
    result = x * log(x)
    iszero(result) || isnan(result) ? zero(result) : result
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
