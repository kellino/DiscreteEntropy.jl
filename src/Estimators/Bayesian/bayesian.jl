using SpecialFunctions
using QuadGK

@doc raw"""
    bayes(data::CountData, α::AbstractFloat; K=nothing)

Compute an estimate of Shannon entropy given data and a concentration parameter ``α``.
If K is not provided, then the observed support size in `data` is used.

```math
\hat{H}_{\text{Bayes}} = - \sum_{k=1}^{K} \hat{p}_k^{\text{Bayes}} \; \log \hat{p}_k^{\text{Bayes}}
```
where

```math
p_k^{\text{Bayes}} = \frac{K + α}{n + A}
```

and

```math
A = \sum_{x=1}^{K} α_{x}
```

In addition to setting your own α, we have the following suggested choices
1) [jeffrey](https://ieeexplore.ieee.org/document/1056331) : α = 0.5
2) laplace: α = 1.0
3) schurmann_grassberger: α = 1 / K
4) minimax: α = √{n} / K

"""
function bayes(data::CountData, α::AbstractFloat; K=nothing)
  if K === nothing
    K = data.K
  end
  weight = α * K + data.N

  logx(weight) - (1.0 / weight) * sum(xlogx(x[1] + α) * x[2] for x in eachcol(data.multiplicities))
end

@doc raw"""
     jeffrey(data::CountData; K=nothing)

Compute [`bayes`](@ref) estimate of entropy, with $α = 0.5$

"""
function jeffrey(data::CountData; K=nothing)
  return bayes(data, 0.5, K=K)
end

@doc raw"""
     laplace(data::CountData; K=nothing)

Compute [`bayes`](@ref) estimate of entropy, with $α = 1.0$

"""
function laplace(data::CountData; K=nothing)
  return bayes(data, 1.0, K=K)
end

@doc raw"""
     schurmann_grassberger(data::CountData; K=nothing)

Compute [`bayes`](@ref) estimate of entropy, with $α = \frac{1}{K}$.
If K is nothing, then use data.K
"""
function schurmann_grassberger(data::CountData; K=nothing)
  if K === nothing
    K = data.K
  end
  return bayes(data, 1.0 / K, K=K)
end

@doc raw"""
     minimax(data::CountData; K=nothing)

Compute [`bayes`](@ref) estimate of entropy, with $α = √\frac{data.N}{K}$ where
K = data.K if K is nothing.
"""
function minimax(data::CountData; K=nothing)
  if K === nothing
    K = data.K
  end
  return bayes(data, sqrt(data.N) / K, K=K)
end
