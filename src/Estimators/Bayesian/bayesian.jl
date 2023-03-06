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
function bayes(data::CountData, α::AbstractFloat)
    weight = α * data.K + data.N

    return logx(weight) - (1.0 / weight) * sum([(kx + α) * logx(kx + α) * nx for (kx, nx) in data.histogram])
end

function jeffrey(data::CountData)
    return bayes(data, 0.5)
end

function laplace(data::CountData)
    return bayes(data, 1.0)
end

function schurmann_grassberger(data::CountData)
    return bayes(data, 1.0 / data.K)
end

function minimax(data::CountData)
    return bayes(data, sqrt(data.N) / data.K)
end
