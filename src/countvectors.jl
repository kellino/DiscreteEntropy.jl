using Base: @propagate_inbounds

abstract type AbstractCounts{T<:Real,V<:AbstractVector{T}} <: AbstractVector{T} end

macro counts(name)
    return quote
        mutable struct $name{T<:Real,V<:AbstractVector{T}} <: AbstractCounts{T,V}
            values::V
            function $(esc(name)){T,V}(values) where {T<:Real,V<:AbstractVector{T}}
                isinf(Base.sum(values)) ? throw(ArgumentError("counts cannot contain Inf or NaN values")) : new{T,V}(values)
            end
        end
        $(esc(name))(values::AbstractVector{T}) where {T<:Real} = $(esc(name)){T,typeof(values)}(values)
    end
end

# length(wv::AbstractCounts) = length(wv.values)
# sum(wv::AbstractCounts) = sum(wv.values)
# Base.isempty(wv::AbstractCounts) = Base.isempty(wv.values)
# size(wv::AbstractCounts) = Base.size(wv.values)
# Base.axes(wv::AbstractCounts) = Base.axes(wv.values)

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
cvector(vs::AbstractVector{<:Integer}) = CountVector(convert(Vector{Float64}, vs))
cvector(vs::AbstractVector{<:Real}) = CountVector(vs)
cvector(vs::AbstractArray{<:Real}) = CountVector(vec(vs))

@counts SampleVector
svector(vs::AbstractVector{<:Integer}) = SampleVector(convert(Vector{Float64}, vs))
svector(vs::AbstractVector{<:Real}) = SampleVector(vs)
svector(vs::AbstractArray{<:Real}) = SampleVector(vec(vs))
