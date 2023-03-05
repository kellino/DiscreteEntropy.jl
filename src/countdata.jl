using CSV;
using StatsBase: countmap, Histogram, fit;
using Printf;
using LinearAlgebra: dot;

@doc """
     abstract type EntropyData
 """
abstract type EntropyData end
struct SampleHistogram <: EntropyData end
struct Samples <: EntropyData end

mutable struct CountData
    multiplicities::Matrix{Float64}
    N::Float64
    K::Int64
end

Base.:(==)(x::CountData, y::CountData) = Base.:(==)(x.multiplicities, y.multiplicities) && x.N == y.N && x.K == y.K

Base.copy(x::CountData) = CountData(x.multiplicities, x.N, x.K)

function coincidences(data::CountData)
    0.0
    # return data.N - sum([kᵢ for (_, kᵢ) in data.histogram])
end

function ratio(data::CountData)::Float64
    0.0
    # return coincidences(data) / data.N
end

function _from_counts(counts::AbstractVector{T}) where {T<:Real}
    map = countmap(counts)
    x1 = collect(keys(map))
    x2 = collect(values(map))
    mm = [x1 x2]

    return CountData(mm', dot(x1, x2), length(mm[:, 1]))
end

function from_counts(counts::CountVector)
    _from_counts(counts.values)
end

function from_samples(samples::SampleVector)
    K = length(unique(samples.values))

    if K == 1
        N::Float64 = length(samples.values)
        return CountData([1.0 N]', N, K)
    end

    counts = filter(!iszero, fit(Histogram, samples.values, nbins=K).weights)

    _from_counts(counts)
end

@doc """
     from_data(data::AbstractVector, ::Type{Samples})
     from_data(data::AbstractVector, ::Type{SampleHistogram})
 """
function from_data(data::AbstractVector, ::Type{Samples})
    from_samples(svector(data))
end

function from_data(data::AbstractVector, ::Type{SampleHistogram})
    from_counts(cvector(data))
end

@doc raw"""
    from_samples
"""
function from_samples(file::String, field)
    csv = CSV.File(file)
    from_samples(csv[field])
end

function to_pmf(data::CountData)
    # TODO not correct
    norm = 1.0 / data.N
    (x[1] * x[2] * norm for x in eachcol(data.multiplicities))
end

function pmf(histogram::CountVector, x)
    # TODO bounds checking required
    return histogram[x]
end

function to_csv_string(data::CountData)::String
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
