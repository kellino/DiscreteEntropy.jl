using CSV;
using StatsBase: countmap, fit, Histogram as Hgm;
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
"""
mutable struct CountData
    multiplicities::Matrix{Float64}
    N::Float64
    K::Int64
end

Base.:(==)(x::CountData, y::CountData) = Base.:(==)(x.multiplicities, y.multiplicities) && x.N == y.N && x.K == y.K

Base.copy(x::CountData) = CountData(x.multiplicities, x.N, x.K)

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

    counts = filter(!iszero, fit(Hgm, samples.values, nbins=K).weights)

    _from_counts(counts, remove_zeros)
end

@doc """
     from_data(data::AbstractVector, ::Type{Samples})
     from_data(data::AbstractVector, ::Type{SampleHistogram})
 """
function from_data(data::AbstractVector, ::Type{Samples}; remove_zeros=true)
    from_samples(svector(data), remove_zeros)
end

function from_data(data::AbstractVector, ::Type{Histogram}; remove_zeros=true)
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
