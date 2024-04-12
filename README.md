# DiscreteEntropy.jl

[![Build Status](https://github.com/kellino/DiscreteEntropy.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/kellino/DiscreteEntropy.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/kellino/DiscreteEntropy.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/kellino/DiscreteEntropy.jl)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/kellino/DiscreteEntropy.jl/blob/main/LICENSE)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://kellino.github.io/DiscreteEntropy.jl/dev)

`DiscreteEntropy` is a Julia package to estimate the Shannon entropy of discrete random variables. It contains efficient implementations of
many popular entropy estimator, such as Chao-Shen, NSB, Miller-Madow and various Bayesian estimators. Furthermore, it also contains functions
for estimating cross entropy, KL divergence, mutual information and conditional information. It supports Jackknife and Bayesian Bootstrap resampling
for data poor estimation.

[For more information, see the documentation.](https://kellino.github.io/DiscreteEntropy.jl/dev/)
