using CSV;
using StatsBase: countmap;
using Printf;
using LinearAlgebra: dot;

@doc raw"""
    abstract type EntropyData
    Histogram <: EntropyData
    Samples <: EntropyData

It is very easy, when confronted with a vector such as ``[1,2,3,4,5,4]`` to forget
whether it represents samples from a distribution, or a histogram of a (discrete) distribution. *DiscreteEntropy.jl*
attempts to make this a difficult mistake to make by enforcing a type difference between a vector of
samples and a vector of counts.

See [`svector`](@ref) and [`cvector`](@ref).

"""
abstract type EntropyData end
struct Histogram <: EntropyData end
struct Samples <: EntropyData end

@doc raw"""
    CountData

# Fields
- `multiplicities::Matrix{Float64}`: multiplicity representation of data
- `N::Float64`: total number of samples
- `K::Int64`: observed support size


# Multiplicities

All of the estimators operate over a multiplicity representation of raw data. Raw data
takes the form either of a vector of samples, or a vector of counts (ie a histogram).

Given histogram `= [1,2,3,2,1,4]`, the multiplicity representation is

```math
\begin{pmatrix}
4 & 2 & 3 & 1 \\
1 & 2 & 1 & 2
\end{pmatrix}
```

The top row represents bin contents, and the bottom row the number of bins.
We have 1 bin with 4 elements, 2 bins with 2 elements, 1 bin with 3 elements
and 2 bins with only 1 element.

The advantages of the multiplicity representation are compactness and efficiency.
Instead of calculating the surprisal of a bin of 2 twice, we can calculate it once and
multiply by the multiplicity. The downside of the representation may be floating point creep due
to multiplication.

# Constructor

CountData is not expected to be called directly, nor is it advised to directly manipulate the fields. Use either [`from_data`](@ref), [`from_counts`](@ref) or
[`from_samples`](@ref) instead.
"""
mutable struct CountData
  multiplicities::Matrix{Float64}
  N::Float64
  K::Int64
end

Base.:(==)(x::CountData, y::CountData) = Base.:(==)(x.multiplicities, y.multiplicities) && x.N == y.N && x.K == y.K
Base.copy(x::CountData) = CountData(x.multiplicities, x.N, x.K)
Base.Broadcast.broadcastable(q::CountData) = Ref(q)

function empty_countdata()
  x::Matrix{Float64} = [;;]
  CountData(x, 0.0, 0)
end

function Base.hash(g::CountData, h::UInt)
  hash(g.multiplicities, hash(g.K, hash(g.N, h)))
end

@doc raw"""
    bins(x::CountData)
Return the bins (top row) of `x.multiplicities`
"""
function bins(x::CountData)
  return x.multiplicities[1, :]
end

@doc raw"""
    bins(x::CountData)
Return the bin multiplicities (bottom row) of `x.multiplicities`
"""
function multiplicities(x::CountData)
  return x.multiplicities[2, :]
end

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

@doc raw"""
     from_counts(counts::AbstractVector; remove_zeros::Bool=true)
     from_counts(counts::CountVector, remove_zeros::Bool)

Return a [`CountData`](@ref) object from a vector or CountVector. Many estimators
cannot handle a histogram with a 0 value bin, so there are filtered out unless remove_zeros is set to false.
"""
function from_counts(counts::CountVector, remove_zeros::Bool)
  if isempty(counts)
    @warn("returning empty CountData object")
    return empty_countdata()
  end
  _from_counts(counts.values, remove_zeros)
end

function from_counts(counts::AbstractVector; remove_zeros::Bool=true)
  from_counts(CountVector(counts), remove_zeros)
end

@doc raw"""
     from_samples(sample::SampleVector)

Return a [`CountData`](@ref) object from a vector of samples.
"""
function from_samples(samples::SampleVector)
  if isempty(samples)
    return empty_countdata()
  end

  K = length(unique(samples.values))

  if K == 1
    N::Float64 = length(samples.values)
    return CountData([1.0 N]', N, K)
  end

  counts::Dict{Int64,Int64} = Dict()

  for x in samples.values
    update_dict!(counts, x)
  end
  v = collect(values(counts))

  from_counts(v)
end

@doc raw"""
    from_data(data::AbstractVector, ::Type{T}; remove_zeros=true) where {T<:EntropyData}

Create a CountData object from a vector or matrix. The function is parameterised on whether
the vector contains samples or the histogram.

0 is automatically removed from `data` when `data` is treated as a count vector, but not when
`data` is a vector of samples.
"""
function from_data(data::AbstractVector, t::Type{T}; remove_zeros=true) where {T<:EntropyData}
  if t == Samples
    from_samples(svector(data))
  else
    from_counts(cvector(data), remove_zeros)
  end
end

function from_data(count_matrix::Matrix, ::Type{Histogram}; remove_zeros=true)
  from_counts(cvector(vec(count_matrix)), remove_zeros)
end

@doc raw"""
    from_csv(file::String, field, ::Type{T}; remove_zeros=false, header=nothing, kw...) where {T<:EntropyData}
Simple wrapper around `CSV.File()` which returns a [`CountData`](@ref) object. For more complex
requirements, it is best to call CSV directly.
"""
function from_csv(file::Union{String,IOBuffer}, field, t::Type{T}; remove_zeros=false, header=false, kw...) where {T<:EntropyData}
  data::Vector{Int64} = []
  for row in CSV.File(file; header=header, kw...)
    push!(data, row[field])
  end

  if t == Samples
    from_samples(svector(data))
  else
    from_counts(cvector(data), remove_zeros=remove_zeros)
  end
end

function pmf(histogram::CountVector)
  histogram ./ sum(histogram)
end

function pmf(histogram::CountVector, x)
  if x > length(histogram)
    @warn("out of bounds")
    return nothing
  end
  normed = histogram ./ sum(histogram)
  return normed[x]
end

function to_csv_string(data::CountData; sep=',')::String
  dict = []
  for x in eachcol(data.multiplicities)
    push!(dict, x[1], x[2])
  end

  return @sprintf("[%s],%d,%d", join(dict, sep), data.N, data.K)
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
