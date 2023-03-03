using StatsBase: countmap
using SpecialFunctions: digamma
using LinearAlgebra
using QuadGK: quadgk
using Base: @propagate_inbounds
# using BenchmarkTools

# abstract type AbstractData{V::AbstractVector{AbstractFloat}} <: AbstractVector{AbstractFloat} end

abstract type AbstractCounts{T<:Real,V<:AbstractVector{T}} <: AbstractVector{T} end

mutable struct CountVector{T<:Real,V<:AbstractVector{T}} <: AbstractCounts{T,V}
    values::V
    # CountVector{T,V}(vs) where {T<:Real,V<:AbstractVector{T}} = isinf(sum(vs)) ? error("bad") : new{T,V}(vs)
    CountVector(vs::AbstractVector{T}) where {T<:Real} =
        isinf(Base.sum(vs)) ? error("bad") : new{T,typeof(vs)}(vs)
end

length(wv::AbstractCounts) = Base.length(wv.values)
sum(wv::AbstractCounts) = sum(wv.values)
Base.isempty(wv::AbstractCounts) = Base.isempty(wv.values)
size(wv::AbstractCounts) = Base.size(wv.values)
Base.axes(wv::AbstractCounts) = Base.axes(wv.values)

Base.IndexStyle(::Type{<:AbstractCounts{T,V}}) where {T,V} = IndexStyle(V)

Base.dataids(wv::AbstractCounts) = Base.dataids(wv.values)

Base.convert(::Type{Vector}, wv::AbstractCounts) = convert(Vector, wv.values)

@propagate_inbounds function Base.getindex(wv::AbstractCounts, i::Integer)
    @boundscheck checkbounds(wv, i)
    @inbounds wv.values[i]
end

@propagate_inbounds function Base.getindex(wv::W, i::AbstractArray) where {W<:AbstractCounts}
    @boundscheck checkbounds(wv, i)
    @inbounds v = wv.values[i]
    W(v)
end

Base.getindex(wv::W, ::Colon) where {W<:AbstractCounts} = W(copy(wv.values), sum(wv))

# @propagate_inbounds function Base.setindex!(wv::AbstractCounts, v::Real, i::Int)
#     s = v - wv[i]
#     sum = wv.sum + s
#     isfinite(sum) || throw(ArgumentError("weights cannot contain Inf or NaN values"))
#     wv.values[i] = v
#     wv.sum = sum
#     v
# end



mutable struct CountData
    histogram::Dict{Float64,Int64}
    N::Float64
    K::Int64
end

mutable struct MatrixData
    multiplicities::Matrix{Float64}
    N::Float64
    K::Int64
end

function from_counts(counts::AbstractVector)
    map = countmap(counts)
    return CountData(map, sum(counts), length(unique(counts)))
end


function from_counts_matrix(counts::AbstractVector)
    map = countmap(counts)
    x1 = collect(keys(map))
    x2 = collect(values(map))
    mm = [x1 x2]
    return MatrixData(mm', dot(x1, x2), length(mm[:, 1]))
end

function schurmann(data::CountData, ξ::Float64=exp(-1 / 2))
    @assert ξ > 0.0
    lim = (1.0 / ξ) - 1.0
    return digamma(data.N) -
           (1.0 / data.N) *
           sum([(digamma(yₓ) + (-1.0)^yₓ * quadgk(t -> t^(yₓ - 1.0) / (1.0 + t), 0, lim)[1]) * yₓ * mm for (yₓ, mm) in data.histogram])

end

function out(y, m, ξ=exp(-1 / 2))
    lim = (1.0 / ξ) - 1.0
    return (digamma(y) + (-1.0)^y * quadgk(t -> t^(y - 1.0) / (1.0 + t), 0, lim)[1]) * y * m
end

function schurmann(data::MatrixData, ξ::Float64=exp(-1 / 2))
    @assert ξ > 0.0
    return digamma(data.N) -
           (1.0 / data.N) *
           sum((out(x[1], x[2], ξ) for x in eachcol(data.multiplicities)))
end

@inline function out_inlined(y, m, ξ=exp(-1 / 2))
    lim = (1.0 / ξ) - 1.0
    return (digamma(y) + (-1.0)^y * quadgk(t -> t^(y - 1.0) / (1.0 + t), 0, lim)[1]) * y * m
end

function schurmann_inlined(data::MatrixData, ξ::Float64=exp(-1 / 2))
    @assert ξ > 0.0
    return digamma(data.N) -
           (1.0 / data.N) *
           sum((out_inlined(x[1], x[2], ξ) for x in eachcol(data.multiplicities)))
end

function schurmann_inlined_div(data::MatrixData, ξ::Float64=exp(-1 / 2))
    @assert ξ > 0.0
    return digamma(data.N) -
           # (1.0 / data.N) *
           sum((out_inlined(x[1], x[2], ξ) for x in eachcol(data.multiplicities))) / data.N
end

# function bench()
# b1 = @benchmark schurmann(from_counts(x)) setup = (x = rand(1:100, 100_000))
# b2 = @benchmark schurmann(from_counts_matrix(x)) setup = (x = rand(1:100, 100_000))
# @benchmark schurmann_inlined(from_counts_matrix(x)) setup = (x = rand(1:100, 100_000))
# @benchmark schurmann_inlined_div(from_counts_matrix(x)) setup = (x = rand(1:100, 100_000))

# end
