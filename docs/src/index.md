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

data = [1,2,3,4,3,2,1];
```

Most of the estimators take a [`CountData`](@ref) object. This is a compact representation of the histogram of the random variable. 
It can be pretty easy to forget whether a vector represents a histogram or a set of samples, so `DiscreteEntropy` forces you to say which
it is when creating a [`CountData`](@ref) object. The easiest way to create a [`CountData`](@ref) object is using [`from_data`](@ref).

```@example quick
# if `data` is a histogram already
cd = from_data(data, Histogram)
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

`DiscreteEntropy.jl` outputs Shannon measures in `nats`. There are helper functions to convert [`to_bits`](@ref) and [`to_bans`](@ref)

```@example quick
h = to_bits(estimate_h(cd, ChaoShen))
```

```@example quick
h = to_bans(estimate_h(cd, ChaoShen))
```


