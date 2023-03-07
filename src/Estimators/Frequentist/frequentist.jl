using SpecialFunctions: digamma
using QuadGK
using Optim: maximizer
using Distributions: Distribution

# Maximum Likelihood

@doc raw"""
    maximum_likelihood(data::CountData)::Float64

Compute the maximum likelihood estimation of Shannon entropy of `data` in nats.

```math
\hat{H}_{ML} = \log(N) - \frac{1}{N} \sum_{i=1}^{K}h_i \log(h_i)
```

# Examples
```jldoctest

julia> data = from_data([1,2,3,2,1], Histogram)
CountData([2.0 3.0 1.0; 2.0 1.0 2.0], 9.0, 6)

julia> maximum_likelihood(data)
1.522955067
```
"""
function maximum_likelihood(data::CountData)
    log(data.N) -
    (1.0 / data.N) *
    sum(xlogx(x[1]) * x[2] for x in eachcol(data.multiplicities))
end

function maximum_likelihood(counts::AbstractVector{AbstractFloat})
    maximum_likelihood(from_counts(counts))
end


# Jackknife MLE

@doc raw"""
    jackknife_ml(data::CountData; corrected=false)::Tuple{AbstractFloat, AbstractFloat}

Returns the *jackknifed* estimate of data and the variance of the jackknifing (not the variance of the estimator itself).

If corrected in true, then the variance is scaled with n-1, else it is scaled with n

As found in the [paper](https://academic.oup.com/biomet/article/65/3/625/234287)
"""
function jackknife_ml(data::CountData; corrected=false)
    # TODO
    0.0
    # return jackknife(data, maximum_likelihood, corrected=corrected)
end


# Miller Madow Corretion Estimator

@doc raw"""
    miller_madow(data::CountData)

Returns the maximum likelihood estimation of Shannon entropy, with a positive offset based
on the total number of samples seen (N) and the support size (K).

```math
\hat{H}_{MM} = \hat{H}_{ML} + \frac{K - 1}{2N}
```
"""
function miller_madow(data::CountData)
    return maximum_likelihood(data) + ((data.K - 1.0) / (2.0 * data.N))
end


# Grassberger Estimator

@doc raw"""
    grassberger(data::CountData)

Returns the Grassberger estimation of Shannon entropy.

```math
\hat{H}_G = log(N) - \frac{1}{N} \sum_{i=1}^{K} h_i \; G(h_i)
```
This is essentially the same as ``\hat{H}_{ML}``, but with the logarithm swapped for the scalar function ``G``

where
```math
G(h) = \psi(h) + \frac{1}{2}(-1)^h \big( \psi(\frac{h+1}{2} - \psi(\frac{h}{2}))
```

This is the solution to ``G(h) = \psi(h) + (-1)^h \int_0^1 \frac{x^h - 1}{x+1} dx``
as given in the [paper](https://arxiv.org/pdf/physics/0307138v2.pdf)

"""
function grassberger(data::CountData)
    log(data.N) -
    (1.0 / data.N) *
    sum(xFx(g, x[1]) * x[2] for x in eachcol(data.multiplicities))
end

@inline function g(h::T) where {T<:Real}
    return digamma(h) + 0.5 * -1.0^h * (digamma(h + 1.0 / 2.0) - digamma(h / 2.0))
end

function grassberger(counts::AbstractVector{Int64})
    grassberger(from_counts(counts))
end


@doc raw"""
    schurmann(data::CountData, ξ::Float64 = ℯ^(-1/2))

[schurmann](https://arxiv.org/pdf/cond-mat/0403192.pdf)

```math
\hat{H}_{SHU} = \psi(N) - \frac{1}{N} \sum_{i=1}^{K} \, h_i \big( \psi(h_i) + (-1)^{h_i} ∫_0^{\frac{1}{\xi} - 1} \frac{t^{h_i}-1}{1+t}dt \big)

```
This is no one ideal value for ``\xi``, however the paper suggests ``e^{(-1/2)} \approx 0.6``

"""
function schurmann(data::CountData, ξ::Float64=exp(-1 / 2))
    @assert ξ > 0.0
    return digamma(data.N) -
           (1.0 / data.N) *
           sum((_schurmann(x[1], x[2], ξ) for x in eachcol(data.multiplicities)))

end

function _schurmann(y, m, ξ=exp(-1 / 2))
    lim = (1.0 / ξ) - 1.0
    return (digamma(y) + (-1.0)^y * quadgk(t -> t^(y - 1.0) / (1.0 + t), 0, lim)[1]) * y * m
end

@doc raw"""
    schurmann_generalised(data::CountVector, xis::XiVector{T}) where {T<:Real}

[schurmann_generalised](https://arxiv.org/pdf/2111.11175.pdf)

```math
\hat{H}_{SHU} = \psi(N) - \frac{1}{N} \sum_{i=1}^{K} \, h_i \big( \psi(h_i) + (-1)^{h_i} ∫_0^{\frac{1}{\xi_i} - 1} \frac{t^{h_i}-1}{1+t}dt \big)

```

Computes the generalised Schurmann entropy estimation, given a countvector *data* and a xivector *xis*, which must both
be the same length.


    schurmann_generalised(data::CountVector, xis::Distribution, scalar=false)

Computes the generalised Schurmann entropy estimation, given a countvector *data* and a distribution *xis*.
"""
function schurmann_generalised(data::CountVector, xis::XiVector{T}) where {T<:Real}
    @assert Base.length(data) == Base.length(xis)
    N = sum(data)

    digamma(N) -
    (1.0 / N) *
    sum(_schurmann(x[2], 1, xis[x[1]]) for x in enumerate(data))
end

function schurmann_generalised(data::CountVector, xis::T, scalar::Bool=false) where {T<:Distribution}
    if scalar
        # TODO this doesn't work for some reason.
        # if rand(xis) is a scalar, then we sample from it length(data) times
        xi_vec = rand(xis, length(data))
    else
        # some distributions, such as Dirichlet, return a vector when sampled, we
        # take this as the default case is it seems more likely to occur
        xi_vec = xivector(rand(xis))
    end

    schurmann_generalised(data, xi_vec)
end

@doc raw"""
    chao_shen(data::CountData)

Return the Chao-Shen estimate of the Shannon entropy of `data` in nats.

```math
\hat{H}_{CS} = - \sum_{i=i}^{K} \frac{\hat{p}_i^{CS} \log \hat{p}_i^{CS}}{1 - (1 - \hat{p}_i^{CS})}
```
where

```math
\hat{p}_i^{CS} = (1 - \frac{1 - \hat{p}_i^{ML}}{N}) \hat{p}_i^{ML}
```
"""
function chao_shen(data::CountData)
    f1 = 0.0 # number of singletons
    for x in eachcol(data.multiplicities)
        if x[1] == 1.0
            f1 = x[2]
            break
        end
    end

    if f1 == data.N
        f1 = data.N - 1 # avoid C=0
    end

    C = 1 - f1 / data.N # estimated coverage

    -sum(xlogx(C * x[1] / data.N) / (1 - (1 - (x[1] / data.N) * C)^data.N) * x[2] for x in eachcol(data.multiplicities))
end


@doc raw"""
    zhang(data::CountData)

Return the Zhang estimate of the Shannon entropy of `data` in nats.

The recommended definition of Zhang's estimator is from [Grabchak *et al.*](https://www.tandfonline.com/doi/full/10.1080/09296174.2013.830551)
```math
\hat{H}_Z = \sum_{i=1}^K \hat{p}_i \sum_{v=1}^{N - h_i} \frac{1}{v} ∏_{j=0}^{v-1} \left( 1 + \frac{1 - h_i}{N - 1 - j} \right)
```

The actual algorithm comes from [Fast Calculation of entropy with Zhang's estimator](https://arxiv.org/abs/1707.08290) by Lozano *et al.*.

# Links
[Entropy estimation in turing's perspective](https://dl.acm.org/doi/10.1162/NECO_a_00266)
"""
function zhang(data::CountData)
    ent = 0.0
    for c in eachcol(data.multiplicities)
        t1 = 1
        t2 = 0
        for k in 1:data.N-c[1]
            t1 *= 1 - ((c[1] - 1.0) / (data.N - k))
            t2 += t1 / k
        end
        ent += t2 * (c[1] / data.N) * c[2]
    end
    ent
end

@doc raw"""
    bonachela(data::CountData)
"""
function bonachela(data::CountData)::Float64
    return 1 / data.N * sum([(y + 1) * sum([1 / j for j in y+2:data.N+2]) * mm for (y, mm) in data.histogram])
end

function shrink(data::CountData)
    # TODO also known as the James-Stein
    0.0
end
