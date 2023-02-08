using SpecialFunctions
using QuadGK: quadgk

γ = Base.MathConstants.eulergamma

@doc raw"""
    ansb(data::CountData; undersampled::Float64=0.1)::Float64

```math
\hat{H}_{ANSB} = \frac{C_\gamma}{\ln(2)} - 1 + 2 \ln(N) - \psi_0(\Delta)
```
where $C_\gamma$ is Euler's Gamma Constant $\approx 0.57721...$, $\psi_0$ is the digamma function and
$\Delta$ the number of coincidences in the data.

Returns the [Asymptotic NSB estimator](https://arxiv.org/pdf/physics/0306063.pdf) (equations 11 and 12)

This is designed for the extremely undersampled regime (K ~ N) and diverges with N when well-sampled. ANSB requires
that ``N/K → 0``, which we set to be ``N/K < 0.1`` by default

"""
function ansb(data::CountData; undersampled::Float64=0.1)::Tuple{Float64,Float64}
    rd = ratio(data)
    if rd > undersampled
        error("data is not sufficiently undersampled $rd")
    end

    Δ = coincidences(data)

    return ((γ / logx(2)) - 1 + 2 * logx(data.N) - digamma(Δ), sqrt(Δ))
end

@inline function ξ(β::Float64, k::Int64)::BigFloat
    return digamma(k * β + 1) - digamma(β + 1)
end

function ρ(β::Float64, k::Int64, data::CountData)
    return gamma(k * ξ(β, k)) / gamma(data.N + k * ξ(β, k)) *
           prod([gamma(y + BigFloat(β)) / gamma(β) * c for (y, c) in data.histogram])

end

@doc raw"""
    nsb(data::CountData; k=data.K)
    nsb(samples::AbstractVector; k=length(unique(samples)))

    Bayesian estimator
"""
function nsb(data::CountData; k=data.K)::Float64

    if data.N <= 1
        @warn("Too few samples")
        return NaN
    end

    return quadgk(β -> ρ(β, k, data) * bayes(β, data), 1e-10, log(k))[1] /
           quadgk(β -> ρ(β, k, data), 1e-10, log(k))[1]
end

function nsb(samples::AbstractVector; k=length(unique(samples)))::Float64
    nsb(from_samples(samples), k=k)
end
