# DiscreteEntropy.jl

[![Build Status](https://github.com/kellino/DiscreteEntropy.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/kellino/DiscreteEntropy.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/kellino/DiscreteEntropy.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/kellino/DiscreteEntropy.jl)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/kellino/DiscreteEntropy.jl/blob/main/LICENSE)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://kellino.github.io/DiscreteEntropy.jl/dev)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.07334/status.svg)](https://doi.org/10.21105/joss.07334)

`DiscreteEntropy` is a Julia package to estimate the Shannon entropy of discrete random variables. It contains implementations of
many popular entropy estimators, such as Chao-Shen, NSB, Miller-Madow and various Bayesian estimators. Moreoever, it also contains functions
for estimating cross entropy, KL divergence, mutual information, conditional information, Theil's U and other entropy measures.
It supports Jackknife and Bayesian Bootstrap resampling for data poor estimation.

[For more information, see the documentation.](https://kellino.github.io/DiscreteEntropy.jl/dev/)

# Quick Example

```
julia> using DiscreteEntropy
julia> data = [1,2,3,4,3,2,1];
julia> h = estimate_h(from_data(data, Histogram), ChaoShen)
julia> 2.0775715569320012
```

# Contributing and Bugs

Please see [CONTRIBUTING.md](/CONTRIBUTING.md) for details on how to contribute to the project through pull requests or issues.


# CITATION

If you have found `DiscreteEntropy.jl` useful, or used it in your work, please consider citing it:

```
@article{Kelly2024,
doi = {10.21105/joss.07334},
url = {https://doi.org/10.21105/joss.07334}, 
year = {2024}, publisher = {The Open Journal}, 
volume = {9}, number = {103}, pages = {7334}, 
author = {David A. Kelly and Ilaria Pia La Torre}, 
title = {DiscreteEntropy.jl: Entropy Estimation of Discrete Random Variables with Julia}, 
journal = {Journal of Open Source Software} 
} 
```
