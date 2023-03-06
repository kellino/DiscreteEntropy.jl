using SpecialFunctions
using QuadGK: quadgk
using Roots: find_zero

γ = Base.MathConstants.eulergamma
Γ = gamma

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
        @warn("data is not sufficiently undersampled $rd, so calculation may diverge...")
    end

    Δ = coincidences(data)

    return ((γ / logx(2)) - 1 + 2 * logx(data.N) - digamma(Δ), sqrt(Δ))
end

function ik(nk, β, Nhat)
    return (digamma(nk + β + 1) - digamma(Nhat + 2)) * (digamma(nk + β + 1.0) - digamma(Nhat + 2)) - trigamma(Nhat + 2)
end

function ji(nk, β, Nhat)
    return (digamma(nk + β + 2) - digamma(Nhat + 2))^2 + trigamma(nk + β + 2) - trigamma(Nhat + 2)
end

function s2(β, data::CountData)
    # calculate the variance or σ
    ν = data.N + data.K * β
    nx = collect(keys(data.histogram))
    kx = collect(values(data.histogram))
    norm = 1.0 / log(2)^2

    left = sum([(((nᵢ + β + 1) * (nᵢ + β) / (ν * (ν + 1))) *
                 (digamma(nᵢ + β + 2) - digamma(ν + 2))^2 +
                 trigamma(nᵢ + β + 2) -
                 trigamma(ν + 2)) * kᵢ
                for (nᵢ, kᵢ) in collect(zip(nx, kx))])


    right = 0.0
    for i in 1:length(nx)
        ni = nx[i] + β
        for k in 1:length(nx)
            if i != k
                nk = nx[k] + β
                right += ((ni * nk) / ν * (ν + 1) *
                          (digamma(nk + 1) - digamma(ν + 2)) *
                          (digamma(ni + 1) - digamma(ν + 2)) -
                          trigamma(ν + 2)) * kx[i] * kx[k]
            end
        end
    end


    return left + right
end


function dlogrho(K0, K1, N)
    # equation 15 from Inference of Entropies of Discrete Random Variables with Unknown Cardinalities,
    # rearranged to make solving for 0 easier

    return K1 / K0 - digamma(K0 + N) + digamma(K0)
end

function find_extremum_log_rho(K::Int64, N::Float64)
    func(x) = dlogrho(x, K, N)

    return find_zero(func, 1)
end

function neg_log_rho(β, data::CountData)::BigFloat
    # equation 8 from Inference of Entropies of Discrete Random Variables with Unknown Cardinalities,
    # rearranged to take logarithm (to avoid overflow), and with P(β(ξ)) = 1 (and therefore ignored)
    κ = data.K * β

    return -(
        (loggamma(κ) - loggamma(data.N + κ)) +
        (sum([kᵢ * (loggamma(nᵢ + β) - loggamma(β)) for (nᵢ, kᵢ) in data.histogram])))

end

function find_l0(data::CountData)
    return neg_log_rho(find_extremum_log_rho(data.K, data.N) / data.K, data)
end

function dxi(β, k)::Float64
    # The derivative of ξ = ψ(kappa + 1) - ψ(β + 1)
    return k * polygamma(1, 1 + k * β) - polygamma(1, 1 + β)
end

@doc raw"""
    nsb(data, K=data.K)

Returns the Bayesian estimate of Shannon entropy of data, using the Nemenman, Shafee, Bialek algorithm

```math
\hat{H}^{\text{NSB}} = \frac{ \int_0^{\ln(K)} d\xi \, \rho(\xi, \textbf{n}) \langle H^m \rangle_{\beta (\xi)}  }
                            { \int_0^{\ln(K)} d\xi \, \rho(\xi\mid n)}
```
where

```math
\rho(\xi \mid \textbf{n}) =
    \mathcal{P}(\beta (\xi)) \frac{ \Gamma(\kappa(\xi))}{\Gamma(N + \kappa(\xi))}
    \prod_{i=1}^K \frac{\Gamma(n_i + \beta(\xi))}{\Gamma(\beta(\xi))}
```

"""
function nsb(data::CountData, K=nothing)

    if K === nothing
        find_nsb(data)
    else
        new_data = set_K(data, K)
        find_nsb(new_data)
    end
end

function nsb(samples::AbstractVector)
    return nsb(from_samples(samples))
end

function find_nsb(data::CountData)
    l0 = find_l0(data)

    # in addition to rearranging equation 8 to avoid over/underflow, it's also
    # helpful to wrap \beta in a "big" to handle the exponential correctly
    top = quadgk(x -> exp(-neg_log_rho(big(x), data) + l0) * dxi(x, data.K) * bayes(data, x), 0, log(data.K))[1]

    evidence = quadgk(x -> exp(-neg_log_rho(big(x), data) + l0) * dxi(x, data.K), 0, log(data.K))[1]

    convert(Float64, top / evidence)

end
