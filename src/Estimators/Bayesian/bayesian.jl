using SpecialFunctions
using QuadGK

@doc raw"""
    bayes(data::CountData, α::AbstractFloat)

Returns an estimate of Shannon entropy given data and a concentration parameter ``α``.

```math
\hat{H}_{\text{Bayes}} = - \sum_{k=1}^{K} \hat{p}_k^{\text{Bayes}} \; \log \hat{p}_k^{\text{Bayes}}
```
where

```math
p_k^{\text{Bayes}} = \frac{k + α}{n + A}
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


function bayes(data::CountData, α::AbstractFloat, K)
    weight = α * K + data.N

    logx(weight) - (1.0 / weight) * sum(xlogx(x[1] + α) * x[2] for x in eachcol(data.multiplicities))
end

function jeffrey(data::CountData, K)
    return bayes(data, 0.5, K)
end

function laplace(data::CountData, K)
    return bayes(data, 1.0, K)
end

function schurmann_grassberger(data::CountData, K)
    return bayes(data, 1.0 / K, K)
end

function minimax(data::CountData, K)
    return bayes(data, sqrt(data.N) / K, K)
end