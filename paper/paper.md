---
title: "DiscreteEntropy: Entropy Estimation of Discrete Random Variables with Julia"
tags:
  - julia
  - discrete entropy estimation
  - information theory
  - mutual information
authors:
  - name: David Kelly
    orcid: 0000-0002-5368-6769
    corresponding: true
    affiliation: 1
  - name: Ilaria Pia La Torre
    affiliation: 2

affiliation:
  - name: King's College London
    index: 1
  - name: University College London
    index: 2
data: 20 September 2024
bibliography: paper.bib
---

# Summary

`DiscreteEntropy` is Julia package to facilitate entropy estimation of discrete random variables.
The entropy of a random variable, `X`, is the average amount of surprise associated with
the different outcomes of that variable. When `X` is known completely, calculating the entropy is easy. It is
given by

\begin{equation}
H(X) = - \sum\_{x \in X} p(x) \log (p(x))
\end{equation}

However, it is a very hard problem when knowledge of the distribution is incomplete. It is well know that
the `MaximumLikelihood` (or Plugin) estimator, underestimates the true entropy on average [@basharin].
This difficulty has lead to a large number of improved estimators. [@Rodriguez2021EntropyEst], for example,
evaluate 18 different estimators, among which are _Grassberger_ [@grassberger2008entropy],
_Chao Shen_ [@chaoshen] , _NSB_ [@nemenman2002entropy], _Zhang_ [@zhang] and _James-Stein_ [@hausser2009entropy].
These estimators were scattered across 3 different programming languages
and 7 different libraries. Some of these estimators are hard to find or poorly maintained. Each implementation had
a different precision, making comparison of estimations difficult.

If one can estimate entropy more accurately, then one can also estimate mutual information more accurately. There
are numerous, cross-domain, applications for entropy and mutual information, such as in telecommunications,
machine learning [@MacKay2003] and software engineering [@bohme:fse:2020, @blackwell2023hyperfuzzing]. `DiscreteEntropy` makes
it easy to apply different estimators to the problem of mutual information, cross entropy and KL divergence, amongst other
measures.

`DiscreteEntropy` provides a comprehensive collection of popular entropy estimators and utilities for working with other Shannon measures.
`DiscreteEntropy` implements a variety of different entropy estimators, which were previously scattered over
different languages and libraries. Some of these scattered implementations are slow, hard to find, or difficult to compile.
`DiscreteEntropy` removes all of these problems. `DiscreteEntropy` also provides functions for computing cross entropy,
KL divergence, mutual information and many other Shannon measures. `DiscreteEntropy` is intended to be
easy to use, with a flexible but type safe interface.

# Statement of need

The `DiscreteEntropy` package has native Julia implementations of all of the estimators explored
in [@Rodriguez2021EntropyEst] and a number of estimators which were not considered.
`DiscreteEntropy` provides a unified and consistent interface for those who with to estimate entropy and other
Shannon measures for their research, or those who want to research entropy estimation directly.

There is no other open-source software package known to us, in any language, with similiar features or similiar breadth of
estimators. The estimators here were mostly implemented from the original papers, those some (such as Bub and Unseen) are
idiomatic ports of original Matlab code.

`DiscreteEntropy` is a fast, simple to use, library that fills a gap between the scattered implementations available online.
It ensures type safety throughout, even preventing confusion between vectors of samples or vectors which represent histograms of samples.

# Example

`DiscreteEntropy` allows the user to call each estimator directly, or to use a helper function `estimate_h`.
The `estimate_h` function is the easiest entry to the library. This function takes a `CountData` object, which
can be constructed from a vector using either `from_data`, `from_counts` or `from_samples`. Both `from_data` and
`estimate_h` are parameterised by types, making it both typesafe and allowing for simple autocompletion. All results
are in `nats`, but `DiscreteEntropy` provides helper functions to convert between units.

```
data = [1,2,3,4,5,4,3,2,1]
count_data = from_data(data, Histogram)
estimate_h(count_data, ChaoShen)
2.2526294444274044

estimate_h(count_data, MaximumLikelihood)
2.078803548653078
```

Unsurprisingly, different estimators will give different results, depending on their underlying assumptions:

```
estimate_h(count_data, MaximumLikelihood)
2.078803548653078

estimate_h(count_data, Unseen)
1.4748194918254784
```

# References
