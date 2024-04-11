using Base: @propagate_inbounds

# using Core: throw_inexacterror
@doc raw"""
AbstractCounts{T<:Real,V<:AbstractVector{T}} <: AbstractVector{T}

Enforced type incompatibility between vectors of samples, vectors of counts, and
vectors of xi.

# CountVector
A vector representing a histogram

# SampleVector
A vector of samples

# XiVector
A vector of xi values for use with the [`schurmann_generalised`](@ref) estimator.

"""
abstract type AbstractCounts{T<:Real,V<:AbstractVector{T}} <: AbstractVector{T} end

macro counts(name)
    return quote
        mutable struct $name{T<:Real,V<:AbstractVector{T}} <: AbstractCounts{T,V}
            values::V
            function $(esc(name)){T,V}(values) where {T<:Real,V<:AbstractVector{T}}
                isinf(Base.sum(values)) ? throw(ArgumentError("this vector cannot contain Inf or NaN")) : new{T,V}(values)
            end
        end
        $(esc(name))(values::AbstractVector{T}) where {T<:Real} = $(esc(name)){T,typeof(values)}(values)
    end
end

length(wv::AbstractCounts) = length(wv.values)
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


@counts CountVector
@doc raw"""
     cvector(vs::AbstractVector{<:Integer})
     cvector(vs::AbstractVector{<:Real}) = CountVector(vs)
     cvector(vs::AbstractArray{<:Real}) = CountVector(vec(vs))

Convert an AbstractVector into a CountVector. A CountVector represents the frequency of sampled values.
"""
cvector(vs::AbstractVector{<:Integer}) = CountVector(convert(Vector{Float64}, vs))
cvector(vs::AbstractVector{<:Real}) = CountVector(vs)
cvector(vs::AbstractArray{<:Real}) = CountVector(vec(vs))

@counts SampleVector
@doc raw"""
    svector(vs::AbstractVector{<:Integer})
    svector(vs::AbstractVector{<:Real})
    svector(vs::AbstractArray{<:Real})

Convert an AbstractVector into a SampleVector. A SampleVector represents a sequence of sampled values.
"""
svector(vs::AbstractVector{<:Integer}) = SampleVector(convert(Vector{Float64}, vs))
svector(vs::AbstractVector{<:Real}) = SampleVector(vs)
svector(vs::AbstractArray{<:Real}) = SampleVector(vec(vs))

@counts XiVector
@doc raw"""
     xivector(vs::AbstractVector{<:Real})
     xivector(vs::AbstractArray{<:Real})

Convert an AbstractVector{Real} into a XiVector. Exclusively for use with [`schurmann_generalised`](@ref).
"""
xivector(vs::AbstractVector{<:Real}) = XiVector(vs)
xivector(vs::AbstractArray{<:Real}) = XiVector(vec(vs))
