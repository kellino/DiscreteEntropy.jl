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
different languages and libraries. Some of these scattered implementations are very slow, very hard to find, or difficult to compile.
`DiscreteEntropy` removes all of these problems. `DiscreteEntropy` also provides functions for computing cross entropy, 
KL divergence, mutual information and many other Shannon measures. `DiscreteEntropy` is intended to be efficient, with
an easy, flexible but type safe interface.

# Statement of need
The entropy of a random variable, `X`, is the average amount of surprise associated with
the different outcomes of that variable. When `X` known completely, calculating the entropy is easy. It is 
given by 

```math
H(X) = - \sum_{x \in X} p(x) \log (p(x))
```
However, it is a very hard problem when knowledge of the distribution is incomplete. It is well know that 
the `MaximumLikelihood` (or Plugin) estimator, underestimates the true entropy on average [@basharin]. 
This difficulty has lead to a large number of improved estimators. [@Rodriguez2021EntropyEst], for example,
evaluate 18 different estimators, among which are *Grassberger* [@grassberger2008entropy], 
*Chao Shen* [@chaoshen] , *NSB* [@nemenman2002entropy], *Zhang* [@zhang] and *James-Stein* [@hausser2009entropy].
These estimators were scattered across 3 different programming languages 
and 7 different libraries. Some of these estimators are difficult to find or poorly maintained. `DiscreteEntropy` 
has native Julia implementations of the majority of the estimators explored in [@Rodriguez2021EntropyEst] and 
a number of estimators which were not considered. `DiscreteEntropy` provides a unified, consistent and performant 
interface for those who with to estimate entropy and other Shannon measures for their research, or those who
want to research entropy estimation directly.

If one can estimate entropy more accurately, then one can also estimate mutual information more accurately. There 
are numerous, cross-domain, applications for entropy and mutual information, such as in telecommunications, 
machine learning [@MacKay2003] and software engineering [@bohme:fse:2020, @blackwell2023hyperfuzzing]. `DiscreteEntropy` makes
it easy to apply different estimators to the problem of mutual information, cross entropy and KL divergence, amonst other
measures.
