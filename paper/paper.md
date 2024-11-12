---
title: "DiscreteEntropy.jl: Entropy Estimation of Discrete Random Variables with Julia"
tags:
  - julia
  - discrete entropy estimation
  - information theory
  - mutual information
authors:
  - name: David A. Kelly
    orcid: 0000-0002-5368-6769
    corresponding: true
    affiliation: 1
  - name: Ilaria Pia La Torre
    orcid: 0009-0006-2733-5283
    equal-contrib: false
    affiliation: 2

affiliations:
  - name: King's College London, UK
    index: 1
  - name: University College London, UK
    index: 2
data: 20 September 2024
bibliography: paper.bib
---

# Summary

`DiscreteEntropy.jl` is a Julia package to facilitate entropy estimation of discrete random variables.
The entropy of a random variable, `X`, is the average amount of surprise associated with
the different outcomes of that variable. When `X` is known completely, calculating the entropy is easy. It is
given by

$$H(X) = - \sum_{x \in X} p(x) \log (p(x))$$

However, it is a very hard problem when knowledge of the distribution is incomplete. It is well known that
the `MaximumLikelihood` (or Plugin) estimator, underestimates the true entropy on average [@basharin].
This difficulty has led to a large number of improved estimators. [@Rodriguez2021EntropyEst], for example,
evaluate 18 different estimators, among which are _Grassberger_ [@grassberger2008entropy],
_Chao Shen_ [@chaoshen] , _NSB_ [@nemenman2002entropy], _Zhang_ [@zhang] and _James-Stein_ [@hausser2009entropy].
These estimators were scattered across 3 different programming languages
and 7 different libraries. Some of these estimators are hard to find or poorly maintained. Each implementation
calculates and reports entropy to a different number of significant digits, which can lead to difficulties in comparison.

If one can estimate entropy more accurately, then one can also estimate mutual information more accurately. There
are numerous, cross-domain, applications for entropy and mutual information, such as in telecommunications,
machine learning [@MacKay2003] and software engineering [@bohme:fse:2020;@blackwell2025hyperfuzzing]. `DiscreteEntropy.jl` makes
it easy to apply different estimators to the problem of mutual information, cross entropy and Kullback–Leibler divergence[@Cover2006], amongst other
measures.

`DiscreteEntropy.jl` provides a comprehensive collection of popular entropy estimators and utilities for working with other Shannon measures.
`DiscreteEntropy.jl` implements a variety of different entropy estimators, which were previously scattered over
different languages and libraries. Some of these scattered implementations are slow, hard to find, or difficult to compile.
`DiscreteEntropy.jl` removes all of these problems. `DiscreteEntropy.jl` also provides functions for estimating cross entropy,
KL divergence, mutual information and many other Shannon measures. `DiscreteEntropy.jl` is intended to be
easy to use, with a flexible but type safe interface.

# Statement of need

The `DiscreteEntropy.jl` package has native Julia implementations of all of the estimators explored
in [@Rodriguez2021EntropyEst] and a number of estimators which were not considered.
`DiscreteEntropy.jl` provides a unified and consistent interface for those who wish to estimate entropy and other
Shannon measures for their research, or those who want to research entropy estimation directly.

There is no other open-source software package known to us, in any language, with similar features or similar breadth of
estimators. The R entropy[@hausser2009entropy] package covers many basic estimators, such as
the maximum likelihood, Miller-Madow, Chao Shen and many bayesian estimators.
Code for PYM[@pym], BUB[@bub] and Unseen[@unseenimp] estimators are found only in the Matlab implementations by the authors of the original papers.
Other estimators, such as Zhang and Grassberger, can be found in the R
Entropart[@entropart] library. Code for the NSB estimator exists in multiple different versions, in C++[@nsb], Matlab[@nsb]
and Python[@ndd]. The estimators in `DiscreteEntropy.jl` were mostly implemented from the original papers,
though some (such as BUB and Unseen) are idiomatic ports of original Matlab code.

`DiscreteEntropy.jl` is a fast, simple to use, library that fills a gap between the scattered implementations available online.
It ensures type safety throughout, even preventing confusion between vectors of samples or vectors which represent histograms of samples.

# Example

`DiscreteEntropy.jl` allows the user to call each estimator directly, or to use a helper function `estimate_h`.
The `estimate_h` function is the easiest entry to the library. This function takes a `CountData` object, which
can be constructed from a vector using either `from_data`, `from_counts` or `from_samples`. Both `from_data` and
`estimate_h` are parameterised by types, making them both typesafe and allowing for simple autocompletion. All results
are in `nats`, but `DiscreteEntropy.jl` provides helper functions to convert between units.

```julia
data = [1,2,3,4,5,4,3,2,1]
count_data = from_data(data, Histogram)

estimate_h(count_data, MaximumLikelihood)
2.078803548653078
```

Unsurprisingly, different estimators give different results, depending on their underlying assumptions:

```julia
estimate_h(count_data, ChaoShen)
2.2526294444274044
```

These assumptions can have a profound effect on estimations of more complex measures, such as mutual information:

```julia
to_bits(mutual_information(Matrix([1 0; 0 1]), MaximumLikelihood))
1.0

to_bits(mutual_information(Matrix([1 0; 0 1]), ChaoWangJost))
1.7548875021634693
```

# References
