using SpecialFunctions
using QuadGK: quadgk
using Roots: find_zero

γ = Base.MathConstants.eulergamma
Γ = gamma

@doc raw"""
    ansb(data::CountData; undersampled::Float64=0.1)::Float64

Return the Asymptotic NSB estimation of the Shannon entropy of `data` in nats.

See [Asymptotic NSB estimator](https://arxiv.org/pdf/physics/0306063.pdf) (equations 11 and 12)

```math
\hat{H}_{\tiny{ANSB}} = (C_\gamma - \log(2)) + 2 \log(N) - \psi(\Delta)
```

where $C_\gamma$ is Euler's Gamma ($\approx 0.57721...$), $\psi_0$ is the digamma function and
$\Delta$ the number of coincidences in the data.

This is designed for the extremely undersampled regime (K ~ N) and diverges with N when well-sampled. ANSB requires
that ``N/K → 0``, which we set to be ``N/K < 0.1`` by default

# External Links
[Asymptotic NSB estimator](https://arxiv.org/pdf/physics/0306063.pdf) (equations 11 and 12)
"""
function ansb(data::CountData; undersampled::Float64=0.1, std_dev=false, verbose=false)
    if verbose
        rd = ratio(data)
        if rd > undersampled
            @warn("data is not sufficiently undersampled $rd, so calculation may diverge...")
        end
    end

    Δ = coincidences(data)
    if iszero(Δ)
        if verbose
            @warn("no coincidences")
        end
        return NaN
    end

    if std_dev
        return (γ - log(2)) + 2 * log(data.N) - digamma(Δ), sqrt(trigamma(Δ))
    else
        return (γ - log(2)) + 2 * log(data.N) - digamma(Δ)
    end
end

function var1(data::CountData, α, ν)
    digamma(ν + 1) - sum(digamma(x[1] + α + 1) * ((x[1] + α + 1) / ν) * x[2] for x in eachcol(data.multiplicities))
end

# TODO doesn't yet take into account the multiplicies, need to figure that one out
function var2(data::CountData, α, ν)
    phi(n) = digamma(n + α + 1) - digamma(ν + 2)
    jf(n) = (digamma(n + 2) - digamma(ν + 2))^2 + trigamma(n + 2) - trigamma(ν + 2)
    c = trigamma(ν + 2)
    norm = (ν + 1) * ν

    var = 0.0
    for i in 1:length(data.multiplicities[1, :])
        ni = data.multiplicities[1, i]
        for k in 1:length(data.multiplicities[1, :])
            nk = data.multiplicities[1, k]
            temp = 0.0
            if i != k
                temp += (ni * nk) / norm * (phi(nk) * phi(ni) - c)
                temp += (ni + 1) * ni / norm * jf(ni)
                temp *= data.multiplicities[2, k]
            end
            var += temp
        end
        var *= data.multiplicities[2, i]
    end
    return var
end

function dlogrho(K0, K1, N)
    # equation 15 from Inference of Entropies of Discrete Random Variables with Unknown Cardinalities,
    # rearranged to make solving for 0 easier
    K1 / K0 - digamma(K0 + N) + digamma(K0)
end

function find_extremum_log_rho(K::Int64, N::Float64)
    func(K0) = dlogrho(K0, K, N)

    return find_zero(func, 1)
end

function neg_log_rho(data::CountData, β, K::Int64)
    # equation 8 from Inference of Entropies of Discrete Random Variables with Unknown Cardinalities,
    # rearranged to take logarithm (to avoid overflow)
    κ = K * β

    return -(
        (loggamma(κ) - loggamma(data.N + κ)) +
        (sum(x[2] * (loggamma(x[1] + β) - loggamma(β)) for x in eachcol(data.multiplicities))))
end

function find_l0(K, data::CountData)
    neg_log_rho(data, find_extremum_log_rho(K, data.N) / K, K)
end

function dxi(β, K)::Float64
    # The derivative of ξ = ψ(kappa + 1) - ψ(β + 1)
    return K * polygamma(1, 1 + K * β) - polygamma(1, 1 + β)
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
function nsb(data::CountData, K)
    l0 = find_l0(K, data)

    numerator = quadgk(β -> exp(-neg_log_rho(data, big(β), K) + l0) * dxi(β, K) * bayes(data, β, K), 0, log(K))[1]

    denominator = quadgk(β -> exp(-neg_log_rho(data, big(β), K) + l0) * dxi(β, K), 0, log(K))[1]

    h = numerator / denominator

    convert(Float64, h)
end

function guess_k(data::CountData, eps=1.e-5)
    # adapted from guess_alphabet_size()
    # https://github.com/simomarsili/ndd/blob/master/ndd/estimators.py
    multiplier = 10
    dk = log(multiplier)
    k1 = convert(Integer, sum(data.multiplicities[2, :]))
    h0 = nsb(data, k1)
    hasym = ansb(data)[1]

    for _ in 1:40
        k1 = round(k1 * multiplier)
        h1 = nsb(data, k1)
        dh = (h1 - h0) / dk
        if dh < eps
            break
        end
        if !(isnan(hasym)) && h1 >= hasym
            return hasym
        end
        h0 = h1
    end

    return convert(Integer, round(k1 / sqrt(multiplier)))
end
