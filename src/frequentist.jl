using SpecialFunctions: digamma
using QuadGK


@doc raw"""
    maximum_likelihood(data::CountData)::Float64
Returns the maximum likelihood estimation of Shannon entropy.

```math
\hat{H}_{ML} = \log(n) - \frac{1}{n} \sum_{k=1}^{K}h_k \log(h_k)
```

where n is the number of samples
"""
function maximum_likelihood(data::CountData)::Float64
    return log(data.N) -
           (1.0 / data.N) *
           sum([xlogx(x) * v for (x, v) in data.histogram])
end

function maximum_likelihood(counts::AbstractVector{Int64})
    maximum_likelihood(from_counts(counts))
end

@doc raw"""
    miller_madow(data::CountData)::Float64

Returns the maximum likelihood estimation of Shannon entropy, with a positive offset based
on the total number of samples seen (n) and the support size (K).

```math
\hat{H}_{MM} = \hat{H}_{ML} + \frac{K - 1}{2n}
```
"""
function miller_madow(data::CountData)::Float64
    return maximum_likelihood(data) + ((data.K - 1.0) / (2.0 * data.N))
end

@inline function g(h::Int64)::Float64
    return digamma(h) + 0.5 * -1.0^h * (digamma(h + 1.0 / 2.0) - digamma(h / 2.0))
end


@doc raw"""
    grassberger(data::CountData)::Float64

Returns the Grassberger estimation of Shannon entropy.

```math
\hat{H}_G = log(n) - \frac{1}{n} \sum_{k=1}^{K} h_k \; G(h_k)
```
This is essentially the same as ``\hat{H}_{ML}``, but with the logarithm swapped for the scalar function ``G``

where
```math
G(h) = \psi(h) + \frac{1}{2}(-1)^h \big( \psi(\frac{h+1}{2} - \psi(\frac{h}{2}))
```

This is the solution to ``G(h) = \psi(h) + (-1)^h \int_0^1 \frac{x^h - 1}{x+1} dx``
as given in the [paper](https://arxiv.org/pdf/physics/0307138v2.pdf)

"""
function grassberger(data::CountData)::Float64
    return log(data.N) - (
        1.0 / data.N * sum([k * g(k) * c for (k, c) in data.histogram])
    )
end

function grassberger(counts::AbstractVector{Int64})::Float64
    grassberger(from_counts(counts))
end


@doc raw"""
    schurmann(data::CountData, ξ::Float64 = ℯ^(-1/2))::Float64

[schurmann](https://arxiv.org/pdf/cond-mat/0403192.pdf)

```math
\hat{H}_{SHU} = \psi(n) - \frac{1}{n} \sum_{k=1}^{K} \, y_x \big( \psi(y_x) + (-1)^{y_x} ∫_0^{\frac{1}{\xi} - 1} \frac{t^{y_x}-1}{1+t}dt \big)

```
This is no one ideal value for ``\xi``, however the paper suggests ``e^{(-1/2)} \approx 0.6``

"""
function schurmann(data::CountData, ξ::Float64=exp(-1 / 2))::Float64
    @assert ξ > 0.0
    return digamma(data.N) -
           (1.0 / data.N) *
           sum([(digamma(yₓ) + (-1.0)^yₓ * quadgk(t -> t^(yₓ - 1) / (1 + t), 0, (1 / ξ) - 1.0)[1]) * yₓ * mm for (yₓ, mm) in data.histogram])

end

@doc raw"""
    schurmann_generalised(data::CountData, xis::Vector{Float64})::Float64

[schurmann_generalised](https://arxiv.org/pdf/2111.11175.pdf)

```math
\hat{H}_{SHU} = \psi(n) - \frac{1}{n} \sum_{k=1}^{K} \, y_x \big( \psi(y_x) + (-1)^{y_x} ∫_0^{\frac{1}{\xi_x} - 1} \frac{t^{y_x}-1}{1+t}dt \big)

```
Accepts a vector is $ξ$ values, rather than just one.

"""
function schurmann_generalised(data::CountData, xis::Vector{Float64})::Float64
    @assert length(data.histogram) == length(xis)

    return digamma(data.N) -
           (1.0 / data.N) *
           sum([(digamma(yₓ) + (-1.0)^yₓ * quadgk(t -> t^(yₓ - 1) / (1 + t), 0, (1 / ξ) - 1.0)[1]) * yₓ * mm
                for ((yₓ, mm), ξ) in collect(zip(data.histogram, xis))])
end


# TODO this is not yet correct, suffers from overflow on big samples
function chao_shen(data::CountData)::Float64
    p = [k / data.N for (k, _) in data.histogram]
    f1 = sum([v for (k, v) in data.histogram if k == 1])

    C = 1 - f1 / data.N
    pa = C .* p
    n = BigFloat(data.N)
    la = (1 .- (1 .- pa) .^ n)

    s = 0
    for i in 1:length(p)
        s += pa[i] * log(pa[i]) / la[i] * get!(data.histogram, i, 0.0)
    end

    return -s
end


function helper(v::Int64, data::CountData)::Float64
    sum([p_k * prod([1 - p_k - (j / data.N) for j in 0:v-1]) for p_k in to_pmf(data)])
end

function Z(v::Int64, data::CountData)::Float64
    # TODO check when factorial gets too big, for the moment we just make everything big
    return data.N^(1 + v) * factorial(big(data.N - (1 + v))) / factorial(big(data.N)) * helper(v::Int64, data)
end

@doc raw"""
    zhang(data::CountData)
"""
function zhang(data::CountData)::Float64
    # TODO check N, if it's too big, this is ruinously expensive to compute
    return sum(1.0 / v * Z(v, data) for v in 1:data.N-1)
end

@doc raw"""
    bonachela(data::CountData)
"""
function bonachela(data::CountData)::Float64
    return 1 / data.N * sum([(y + 1) * sum([1 / j for j in y+2:data.N+2]) * mm for (y, mm) in data.histogram])
end
