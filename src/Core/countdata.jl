using CSV;
using StatsBase: countmap;
using Printf;
using LinearAlgebra: dot;

@doc """
    abstract type EntropyData
"""
abstract type EntropyData end
struct Histogram <: EntropyData end
struct Samples <: EntropyData end

@doc """
    CountData
    an 2 x m matrix where m[1, :] is counts and m[2, :] the number of bins with those counts
    [[2 3 1] => counts / icts
    [2 1 2]] => bins / mm
    so we have two bins with two, 1 bin with 3, and 2 bins with 1
"""
mutable struct CountData
    multiplicities::Matrix{Float64}
    N::Float64
    K::Int64
end

Base.:(==)(x::CountData, y::CountData) = Base.:(==)(x.multiplicities, y.multiplicities) && x.N == y.N && x.K == y.K

Base.copy(x::CountData) = CountData(x.multiplicities, x.N, x.K)

function Base.hash(g::CountData, h::UInt)
    hash(g.multiplicities, hash(g.K, hash(g.N, h)))
end

# function hash(obj::CountData)
#     return hash((obj.multiplicities, obj.N, obj.K))
# end

function find_col(data::CountData, target)
    for x in eachcol(data.multiplicities)
        if x[1] == target
            return x
        end
    end
end

function singletons(data::CountData)
    find_col(data, 1)
end

function doubletons(data::CountData)
    find_col(data, 2)
end

function coincidences(data::CountData)
    data.N - data.K
end

function ratio(data::CountData)::Float64
    coincidences(data) / data.N
end

function _from_counts(counts::AbstractVector{T}, zeros) where {T<:Real}
    map = countmap(counts)
    if zeros
        delete!(map, 0)
    end
    x1 = collect(keys(map))
    x2 = collect(values(map))
    mm = [x1 x2]

    return CountData(mm', dot(x1, x2), sum(mm[:, 2]))
end

function from_counts(counts::CountVector, remove_zeros::Bool)
    _from_counts(counts.values, remove_zeros)
end

function from_counts(counts::AbstractVector; remove_zeros::Bool=true)
    from_counts(CountVector(counts), remove_zeros)
end

function from_samples(samples::SampleVector, remove_zeros::Bool)
    K = length(unique(samples.values))

    if K == 1
        N::Float64 = length(samples.values)
        return CountData([1.0 N]', N, K)
    end

# <<<<<<< HEAD
    counts::Dict{Int64,Int64} = Dict()
    for x in filter(!iszero, samples.values)
        if haskey(counts, x)
            counts[x] += 1
        else
            counts[x] = 1
        end
    end
    v = collect(values(counts))

    _from_counts(v, remove_zeros)
end

@doc """
    from_data(data::AbstractVector, ::Type{Samples})
    from_data(data::AbstractVector, ::Type{Histogram})
"""
function from_data(data::AbstractVector, ::Type{Samples}; remove_zeros=true)
    from_samples(svector(data), remove_zeros)
end

function from_data(data::AbstractVector, ::Type{Histogram}, remove_zeros=true)
    from_counts(cvector(data), remove_zeros)
end

function from_data(count_matrix::Matrix, ::Type{Histogram}; remove_zeros=true)
    from_counts(cvector(vec(count_matrix)), remove_zeros)
end

@doc raw"""
    from_samples
"""
function from_samples(file::String, field; remove_zeros=true)
    csv = CSV.File(file)
    from_samples(csv[field], remove_zeros)
end


function pmf(histogram::CountVector, x)
    # TODO bounds checking required
    return histogram[x]
end

function to_csv_string(data::CountData)::String
    # TODO check this still works
    dict = []
    for x in eachcol(data.multiplicities)
        push!(dict, x[1], x[2])
    end

    return @sprintf("%s,%d,%d", join(dict, ','), data.N, data.K)
end

function set_K(data::CountData, K::Integer)
    ret = copy(data)
    ret.K = K
    ret
end

function set_K!(data::CountData, K::Int64)
    data.K = K
    data
end

function set_N!(data::CountData, N::Float64)
    data.N = N
    data
end
