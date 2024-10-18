using SpecialFunctions: digamma
using QuadGK

# Maximum Likelihood

@doc raw"""
    maximum_likelihood(data::CountData)::Float64

Compute the maximum likelihood estimation of Shannon entropy of `data` in nats.

```math
\hat{H}_{\tiny{ML}} = - \sum_{i=1}^K p_i \log(p_i)
```

or equivalently
```math
\hat{H}_{\tiny{ML}} = \log(N) - \frac{1}{N} \sum_{i=1}^{K}h_i \log(h_i)
```
"""
function maximum_likelihood(data::CountData)
  log(data.N) -
  (1.0 / data.N) *
  sum(xlogx(x[1]) * x[2] for x in eachcol(data.multiplicities))
end


# Jackknife MLE

@doc raw"""
    jackknife_mle(data::CountData; corrected=false)::Tuple{AbstractFloat, AbstractFloat}

Compute the *jackknifed* [`maximum_likelihood`](@ref) estimate of data and the variance of the
jackknifing (not the variance of the estimator itself).

If `corrected` is true, then the variance is scaled with $data.N-1$, else it is scaled with $data.N$. `corrected`
has no effect on the entropy estimation.

# External Links
[Estimation of the size of a closed population when capture probabilities vary among animals](https://academic.oup.com/biomet/article/65/3/625/234287)
"""
function jackknife_mle(data::CountData; corrected=false)
  return jackknife(data, MaximumLikelihood, corrected=corrected)
end


# Miller Madow Corretion Estimator

@doc raw"""
    miller_madow(data::CountData)

Compute the Miller Madow estimation of Shannon entropy, with a positive bias based
on the total number of samples seen (N) and the support size (K).

```math
\hat{H}_{\tiny{MM}} = \hat{H}_{\tiny{ML}} + \frac{K - 1}{2N}
```
"""
function miller_madow(data::CountData)
  return maximum_likelihood(data) + ((data.K - 1.0) / (2.0 * data.N))
end


# Grassberger Estimator

@doc raw"""
    grassberger(data::CountData)

Compute the Grassberger (1988) estimation of Shannon entropy of `data` in nats

```math
\hat{H}_{\tiny{Gr88}} = \sum_i \frac{h_i}{H} \left(\log(N) - \psi(h_i) - \frac{(-1)^{h_i}}{n_i + 1}  \right)
```
Equation 13 from
[Finite sample corrections to entropy and dimension estimate](https://www.academia.edu/10831948/Finite_sample_corrections_to_entropy_and_dimension_estimates)
"""
function grassberger(data::CountData)
  log_n = log(data.N)
  sum((x[1] / data.N * (log_n - digamma(x[1]) - (-1)^x[1] / (x[1] + 1))) * x[2] for x in eachcol(data.multiplicities))
end

@doc raw"""
     schurmann(data::CountData, ξ::Float64 = ℯ^(-1/2))

Compute the Schurmann estimate of Shannon entropy of `data` in nats.

```math
\hat{H}_{SHU} = \psi(N) - \frac{1}{N} \sum_{i=1}^{K} \, h_i \left( \psi(h_i) + (-1)^{h_i} ∫_0^{\frac{1}{\xi} - 1} \frac{t^{h_i}-1}{1+t}dt \right)
```
There is no ideal value for ``\xi``, however the paper suggests ``e^{(-1/2)} \approx 0.6``

# External Links
[schurmann](https://arxiv.org/pdf/cond-mat/0403192.pdf)
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

# Schurmann Generalised Estimator

@doc raw"""
    schurmann_generalised(data::CountVector, xis::XiVector{T}) where {T<:Real}


```math
\hat{H}_{\tiny{SHU}} = \psi(N) - \frac{1}{N} \sum_{i=1}^{K} \, h_i \left( \psi(h_i) + (-1)^{h_i} ∫_0^{\frac{1}{\xi_i} - 1} \frac{t^{h_i}-1}{1+t}dt \right)

```

Compute the generalised Schurmann entropy estimation, given a countvector `data` and a xivector `xis`, which must both
be the same length.


    schurmann_generalised(data::CountVector, xis::Distribution, scalar=false)

Computes the generalised Schurmann entropy estimation, given a countvector `data` and a vector of `xi` values.

## External Links
[schurmann_generalised](https://arxiv.org/pdf/2111.11175.pdf)
"""
function schurmann_generalised(data::CountVector, xis::XiVector{T}) where {T<:Real}
  @assert Base.length(data) == Base.length(xis)
  N = sum(data)


  r = 0.0
  for x in enumerate(data)
    r += sum(_schurmann(x[2], 1, xis[x[1]]))
  end

  digamma(N) - (1.0 / N) * r
end


# Chao Shen Estimator

@doc raw"""
    chao_shen(data::CountData)

Compute the Chao-Shen estimate of the Shannon entropy of `data` in nats.

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

  # TODO this might suffers from under/overflow when data.N is large
  -sum(xlogx(C * x[1] / data.N) / (1 - (1 - (x[1] / data.N) * C)^data.N) * x[2] for x in eachcol(data.multiplicities))
end


# Zhang Estimator

@doc raw"""
    zhang(data::CountData)

Compute the Zhang estimate of the Shannon entropy of `data` in nats.

The recommended definition of Zhang's estimator is from [Grabchak *et al.*](https://www.tandfonline.com/doi/full/10.1080/09296174.2013.830551)
```math
\hat{H}_Z = \sum_{i=1}^K \hat{p}_i \sum_{v=1}^{N - h_i} \frac{1}{v} ∏_{j=0}^{v-1} \left( 1 + \frac{1 - h_i}{N - 1 - j} \right)
```

The actual algorithm comes from [Fast Calculation of entropy with Zhang's estimator](https://arxiv.org/abs/1707.08290) by Lozano *et al.*.

# Exernal Links
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


# Bonachela Estimator

@doc raw"""
    bonachela(data::CountData)

Compute the Bonachela estimator of the Shannon entropy of `data` in nats.

```math
\hat{H}_{B} = \frac{1}{N+2} \sum_{i=1}^{K} \left( (h_i + 1) \sum_{j=n_i + 2}^{N+2} \frac{1}{j} \right)
```

# External Links
[Entropy estimates of small data sets](https://arxiv.org/pdf/0804.4561.pdf)
"""
function bonachela(data::CountData)
  acc = 0.0
  for x in eachcol(data.multiplicities)
    t = 0.0
    ni = x[1] + 1
    for j in ni+1:data.N+2
      t += 1 / j
    end
    acc += ni * t * x[2]
  end
  return 1.0 / (data.N + 2) * acc
end


# Shrink / James-Stein Estimator

@doc raw"""
    shrink(data::CountData)

Compute the Shrinkage, or James-Stein estimator of Shannon entropy for `data` in nats.

```math
\hat{H}_{\tiny{SHR}} = - \sum_{i=1}^{K} \hat{p}_x^{\tiny{SHR}} \log(\hat{p}_x^{\tiny{SHR}})
```
where

```math
\hat{p}_x^{\tiny{SHR}} = \lambda t_x + (1 - \lambda) \hat{p}_x^{\tiny{ML}}
```

and
```math
\lambda = \frac{ 1 - \sum_{x=1}^{K} (\hat{p}_x^{\tiny{SHR}})^2}{(n-1) \sum_{x=1}^K (t_x - \hat{p}_x^{\tiny{ML}})^2}
```

with
```math
t_x = 1 / K
```

# Notes
Based on the implementation in the R package [entropy](https://cran.r-project.org/web/packages/entropy/index.html)

# External Links
[Entropy Inference and the James-Stein Estimator](https://www.jmlr.org/papers/volume10/hausser09a/hausser09a.pdf)
"""
function shrink(data::CountData)
  freqs = lambdashrink(data)
  mm = [freqs data.multiplicities[2, :]]'
  cd = CountData(mm, dot(mm[1, :], mm[2, :]), data.K)
  estimate_h(cd, MaximumLikelihood)
end

function _lambdashrink(N, u, t)
  varu = u .* (1 .- u) ./ (N - 1)
  msp = sum((u .- t) .^ 2)

  if msp == 0
    return 1
  else
    lambda = sum(varu) / msp
    if lambda > 1
      return 1
    elseif lambda < 0
      return 0
    else
      return lambda
    end
  end
end

function lambdashrink(data::CountData)
  t = 1 / data.K
  u = data.multiplicities[1, :] ./ data.N

  if data.N == 0 || data.N == 1
    lambda = 1
  else
    lambda = _lambdashrink(data.N, u, t)
  end

  lambda .* t .+ (1 - lambda) .* u
end


# Chao Wang Jost Estimator

@doc raw"""
    chao_wang_jost(data::CountData)

Compute the Chao Wang Jost Shannon entropy estimate of `data` in nats.

```math
\hat{H}_{\tiny{CWJ}} = \sum_{1 \leq h_i \leq N-1} \frac{h_i}{N} \left(\sum_{k=h_i}^{N-1} \frac{1}{k} \right) +
\frac{f_1}{N} (1 - A)^{-N + 1} \left\{ - \log(A) - \sum_{r=1}^{N-1} \frac{1}{r} (1 - A)^r \right\}
```

with

```math
A = \begin{cases}
\frac{2 f_2}{(N-1) f_1 + 2 f_2} \, & \text{if} \, f_2 > 0 \\
\frac{2}{(N-1)(f_1 - 1) + 1} \, & \text{if} \, f_2 = 0, \; f_1 \neq 0 \\
1, & \text{if} \, f_1 = f_2 = 0
\end{cases}
```

where $f_1$ is the number of singletons and $f_2$ the number of doubletons in `data`.

# Notes
The algorithm is slightly modified port of that used in the [entropart](https://github.com/EricMarcon/entropart/blob/master/R/Shannon.R) R library.

# External Links
[Entropy and the species accumulation curve](https://www.researchgate.net/publication/263250571_Entropy_and_the_species_accumulation_curve_A_novel_entropy_estimator_via_discovery_rates_of_new_species)
"""
function chao_wang_jost(data::CountData)
  singles = singletons(data)
  doubles = doubletons(data)

  if isnothing(singles)
    f1 = 0
  else
    f1 = singles[2]
  end
  if isnothing(doubles)
    f2 = 0
  else
    f2 = doubles[2]
  end

  A =
    if f2 > 0
      2 * f2 / ((data.N - 1) * f1 + 2 * f2)
    else
      if f1 > 0
        2 / ((data.N - 1) * (f1 - 1) + 2)
      else
        1
      end
    end

  cwj = sum(x[1] / data.N * (digamma(data.N) - digamma(x[1])) * x[2] for x in eachcol(data.multiplicities))

  if A != 1
    p2 = sum(1 / r * (1 - A)^r for r in 1:data.N-1)
    cwj += f1 / data.N * (1 - A)^(1 - data.N) * (-log(A) - p2)
  end
  cwj
end
