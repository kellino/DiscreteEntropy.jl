using CSV;
using StatsBase;
using Printf;
using LinearAlgebra: dot;

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

function fc(counts::AbstractVector{T}) where {T<:Real}
    map = countmap(counts)
    x1 = collect(keys(map))
    x2 = collect(values(map))
    mm = [x1 x2]

    return CountData(mm', dot(x1, x2), length(mm[:, 1]))
end

function from_counts(counts::CountVector)
    fc(counts.values)
end

function from_samples(samples::SampleVector)
    K = length(unique(samples.values))

    if K == 1
        N::Float64 = length(samples.values)
        return CountData([1.0 N]', N, K)
    end

    counts = filter(!iszero, fit(Histogram, samples.values, nbins=K).weights)

    fc(counts)
end

function from_samples(file::String, field)
    csv = CSV.File(file)
    from_samples(csv[field])
end

function to_pmf(data::CountData)
    # TODO not correct
    norm = 1.0 / data.N
    (x[1] * x[2] * norm for x in eachcol(data.multiplicities))
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
