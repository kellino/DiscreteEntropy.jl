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

```@example quick
using DiscreteEntropy

data = [1,2,3,4,3,2,1]
```

Most of the estimators take a [`CountData`](@ref) object. This is a compact representation of the histogram of the random variable. The easiest
way to create it is via `from_data`

```@example quick
# if `data` is a histogram already
cd = from_data(data, Histogram)

# or if `data` is actually a vector of samples

cds = from_data(data, Samples)
```

```@example quick
# or if `data` is actually a vector of samples

cds = from_data(data, Samples)
```

```
# now we can estimate
h = estimate_h(from_data(data, Histogram), ChaoShen)
```

```@example quick
# treating data as a vector of samples
h = estimate_h(from_data(data, Samples), ChaoShen)
```

`DiscreteEntropy.jl` outputs Shannon measures in `nats`. 

```@example quick
h = to_bits(estimate_h(cd, ChaoShen))
```


