---
title: "DiscreteEntropy: Performant Entropy Estimation of Discrete Random Variables with Julia"
tags:
  - julia
  - discrete entropy estimation
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
data: 15 April 2024
bibliography: paper.bib
---

# Summary
`DiscreteEntropy` is Julia package to facilitate entropy estimation of discrete random variables. `DiscreteEntropy`
provides a comprehensive collection of entropy estimators and utilities for working with other Shannon measures.
`DiscreteEntropy` implements a variety of different entropy estimators, which were previously scattered over 
different languages and libraries. `DiscreteEntropy` also provides functions for computing cross entropy, 
KL divergence, mutual information and many other Shannon measures. `DiscreteEntropy` is intended to be efficient, with
an easy, flexible but type safe interface.

# Statement of need
The entropy of a random variable, `X`, is the average amount of surprise associated with
the different outcomes of that variable. When `X` known completely, calculating the entropy is easy. It is 
given by 

```math
H(X) = - \sum_{x \in X} p(x) \log (p(x))
```
However, it is extremely hard when knowledge of the distribution is incomplete. It is well know that 
the `MaximumLikelihood` (or Plugin) estimator, underestimates the true entropy on average. The entropy of `X` is
*at least* as large...

