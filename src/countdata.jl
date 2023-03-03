using CSV;
using StatsBase;
using Printf;
using OrderedCollections;
using LinearAlgebra;

mutable struct CountData
    histogram::OrderedDict{Float64,Int64}
    N::Float64
    K::Int64
end

Base.:(==)(x::CountData, y::CountData) = Base.:(==)(x.histogram, y.histogram) && x.N == y.N && x.K == y.K

Base.copy(x::CountData) = CountData(x.histogram, x.N, x.K)

function coincidences(data::CountData)::Int64
    return data.N - sum([kᵢ for (_, kᵢ) in data.histogram])
end

function ratio(data::CountData)::Float64
    return coincidences(data) / data.N
end

function from_counts(counts::AbstractVector)::CountData
    map = countmap(counts)
    # TODO isn't this wrong?
    return CountData(map, sum(counts), length(counts))
end

function from_dict(d::Dict)::CountData
    return CountData(d, sum(x * y for (x, y) in d), sum(y for (_, y) in d))
end

function from_samples(samples::AbstractVector)::CountData
    if isempty(samples)
        return error("no samples provided")
    end
    if typeof(samples[1]) == String
        println("not yet implemented")
        return
    end

    K = length(unique(samples))

    if K == 1
        return CountData(Dict(1.0 => length(samples)), length(samples), K)
    end

    nn = filter(!iszero, fit(Histogram, samples, nbins=K).weights)

    map = countmap(nn)
    return CountData(map, length(samples), K)
end

function from_samples(file::String, field)
    csv = CSV.File(file)
    from_samples(csv[field])
end

function to_pmf(data::CountData)::Vector{Float64}
    return [y * mm / data.N for (y, mm) in data.histogram]
end

function to_pmf(counts::AbstractVector{Int64})::Vector{Float64}
    norm = 1.0 / sum(counts)
    return map(x -> x * norm, counts)
end

function to_csv_string(data::CountData)::String
    dict = []
    for (y, mm) in data.histogram
        push!(dict, y, mm)
    end

    return @sprintf("%s,%d,%d", join(dict, ','), data.N, data.K)
end

function set_K(data::CountData, K::Integer)
    ret = copy(data)
    ret.K = K
    ret
end

function set_k!(data::CountData, K::Int64)
    data.K = K
    data
end

function set_N!(data::CountData, N::Float64)
    data.N = N
    data
end

function from_pmf(p::AbstractVector, N::Int64)
    return from_counts(p .* N)
end
