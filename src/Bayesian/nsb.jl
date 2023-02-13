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
        error("data is not sufficiently undersampled $rd")
    end

    Δ = coincidences(data)

    return ((γ / logx(2)) - 1 + 2 * logx(data.N) - digamma(Δ), sqrt(Δ))
end


function dlogrho(K0, K1, N)
    # equation 15 from Inference of Entropies of Discrete Random Variables with Unknown Cardinalities,
    # rearranged to make solving for 0 easier

    return K1 / K0 - digamma(K0 + N) + digamma(K0)
end

function find_extremum_log_rho(K::Int64, N::Int64)::Float64
    func(x) = dlogrho(x, K, N)

    return find_zero(func, 1)
end

function neg_log_rho(β::BigFloat, data::CountData)::BigFloat
    # equation 8 from Inference of Entropies of Discrete Random Variables with Unknown Cardinalities,
    # rearranged to take logarithm (to avoid overflow), and with P(β(ξ)) = 1 (and therefore ignored)
    κ = data.K * β

    return -(
        (loggamma(κ) - loggamma(data.N + κ)) +
        (sum([kᵢ * (loggamma(nᵢ + β) - loggamma(β)) for (nᵢ, kᵢ) in data.histogram])))

end

function find_l0(data::CountData)::Float64
    return neg_log_rho(find_extremum_log_rho(data.K, data.N) / data.K, data)
end

function dxi(β, k)::Float64
    # The derivative of ξ = ψ(kappa + 1) - ψ(β + 1)
    return k * polygamma(1, 1 + k * β) - polygamma(1, 1 + β)
end

@doc raw"""
    nsb(data; k=data.K)

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
function nsb(data::CountData; k=data.K)

    l0 = find_l0(data)

    # in addition to rearranging equation 8 to avoid over/underflow, it's also
    # helpful to wrap \beta in a "big" to handle the exponential correctly
    top = quadgk(x -> exp(-neg_log_rho(big(x), data) + l0) * dxi(x, data.K) * bayes(x, data), 0, log(k))[1]
    bot = quadgk(x -> exp(-neg_log_rho(big(x), data) + l0) * dxi(x, data.K), 0, log(k))[1]

    return convert(Float64, top / bot)
end
