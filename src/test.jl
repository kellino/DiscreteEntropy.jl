function r(v, f, t)::Float64
    prod([1.0 + ((1.0 - f) / (t - 1.0 - j)) for j in 0:v-1])
end

function q(f::Int64, t::Int64)::Float64
    return sum([r(v, f, t) / v for v in 1:t-f])
end


# this is very slow
# function zhang(data::CountData)::Float64
#     return sum([b * f * q(f, data.N) for (f, b) in data.histogram]) / data.N
# end
