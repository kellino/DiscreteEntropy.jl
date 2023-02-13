using CSV;
using StatsBase;
using Printf;
using OrderedCollections;

mutable struct CountData
    histogram::OrderedDict{Int64,Int64}
    N::Int64
    K::Int64
    f1::Int64
    f2::Int64
end

function coincidences(data::CountData)::Int64
    return data.N - sum([nᵢ > 1 ? kᵢ : 0 for (nᵢ, kᵢ) in data.histogram])
end

function ratio(data::CountData)::Float64
    return coincidences(data) / data.N
end

function singles_and_doubles(histogram::Dict{Int64,Int64})::Tuple{Int64,Int64}
    singletons = 0
    doubletons = 0

    for (v, mm) in histogram
        if v == 1
            singletons += mm
        end
        if v == 2
            doubletons += mm
        end
    end

    return (singletons, doubletons)
end

function from_counts(counts::AbstractVector{Int64})::CountData
    map = countmap(counts)
    s, d = singles_and_doubles(map)
    return CountData(map, sum(counts), length(counts), s, d)
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
        return CountData(Dict(1 => length(samples)), length(samples), K, 1, 0)
    end

    nn = filter(!iszero, fit(Histogram, samples, nbins=K).weights)

    # from_counts(nn)
    map = countmap(nn)
    s, d = singles_and_doubles(map)
    return CountData(map, length(samples), K, s, d)
end

function from_samples(file::String, field)
    csv = CSV.File(file)
    from_samples(csv[field])
end

function to_pmf(data::CountData)::Vector{Float64}
    return [y * mm / data.N for (y, mm) in data.histogram]
end

function to_csv_string(data::CountData)::String
    dict = []
    for (y, mm) in data.histogram
        push!(dict, y, mm)
    end

    return @sprintf("%s,%d,%d,%d,%d", join(dict, ','), data.N, data.K, data.f1, data.f2)
end
