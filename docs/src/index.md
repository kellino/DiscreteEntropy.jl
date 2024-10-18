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

At present, we have implementations for:

- [`maximum_likelihood`](@ref) MaximumLikelihood
- [`jackknife_mle`](@ref) JackknifeMLE
- [`miller_madow`](@ref) MillerMadow
- [`grassberger`](@ref) Grassberger
- [`schurmann`](@ref) Schurmann
- [`schurmann_generalised`](@ref) SchurmannGeneralised
- [`bub`](@ref) BUB
- [`chao_shen`](@ref) ChaoShen
- [`zhang`](@ref) Zhang
- [`bonachela`](@ref) Bonachela
- [`shrink`](@ref) Shrink
- [`chao_wang_jost`](@ref) ChaoWangJost
- [`unseen`](@ref) Unseen
- [`bayes`](@ref) Bayes
- [`jeffrey`](@ref) Jeffrey
- [`laplace`](@ref) Laplace
- [`schurmann_grassberger`](@ref) SchurmannGrassberger
- [`minimax`](@ref) Minimax
- [`nsb`](@ref) NSB
- [`ansb`](@ref) ANSB
- [`pym`](@ref) PYM

We also have some non-traditional mixed estimators, such as [`jackknife`](@ref), which allows jackknife
resampling to be applied to any estimator, [`bayesian_bootstrap`](@ref)
which applies bootstrap resampling to an estimator, and [`pert`](@ref),
which is a three point estimation technique combining pessimistic and optimistic estimations.

In addition, we also provide a number of other information theoretic measures which use these
estimators under the hood:

- [`mutual_information`](@ref)
- [`conditional_entropy`](@ref)
- [`cross_entropy`](@ref)
- [`kl_divergence`](@ref)
- [`jensen_shannon_divergence`](@ref)
- [`jensen_shannon_distance`](@ref)
- [`jeffreys_divergence`](@ref)
- [`uncertainty_coefficient`](@ref)

## [Installing DiscreteEntropy](@id installing-DiscreteEntropy)

1. If you have not done so already, install [Julia](https://julialang.org/downloads/).
   Julia 1.8 to Julia <=1.10 are currently supported. Nightly and Julia 1.11 are not (yet) supported.

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

## Contributing

All contributions are welcome! Please see
[CONTRIBUTING.md](https://github.com/kellino/DiscreteEntropy.jl/blob/main/CONTRIBUTING.md) for details.
Anyone wishing to add an estimator is particularly welcome. Ideally, the estimator will take a [`CountData`](@ref) struct,
though this might not always be suitable (eg [`schurmann_generalised`](@ref)) and also added to [`estimate_h`](@ref). Any
estimator will also have to come with tests.
