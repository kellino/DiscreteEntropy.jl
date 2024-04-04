```@meta
CurrentModule = DiscreteEntropy
DocTestSetup = quote
    using DiscreteEntropy
end

```

# [DiscreteEntropy](@id index)

## Summary
`DiscreteEntropy` is a Julia package to estimate the Shannon entropy of discrete data.

DiscreteEntropy implements a large collection of entropy estimators.

## [Installing DiscreteEntropy](@id installing-DiscreteEntropy)

1. If you have not done so already, install [Julia](https://julialang.org/downloads/). Julia 1.8 and
higher are supported. Nightly is not (yet) supported.

2. Install `DiscreteEntropy` using

```
using Pkg; Pkg.add("DiscreteEntropy")
```

or 

```
] add DiscreteEntropy
```

## Basic Usage

```@example examples
using DiscreteEntropy

data = [1,2,3,4,3,2,1]
h = estimate_h(from_data(data, Histogram), ChaoShen)
```
