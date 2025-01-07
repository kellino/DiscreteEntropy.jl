# Estimators

We split the estimators into two broad categories, which we call _Frequentist_ and _Bayesian_. We also
have a few composite estimators that either take an averaging or resampling approach to estimation.

[`estimate_h`](@ref) is parameterised on the _type_ of the estimator. The complete list of types is currently:

- MaximumLikelihood
- JackknifeMLE
- MillerMadow
- Grassberger
- ChaoShen
- Zhang
- Bonachela
- Shrink
- ChaoWangJost
- Unseen
- Schurmann
- SchurmannGeneralised
- BUB
- Bayes
- NSB
- PYM
- ANSB
- Jeffrey
- Laplace
- SchurmannGrassberger
- Minimax
- PERT
- WuYang

## Frequentist Estimators

```@docs
maximum_likelihood
jackknife_mle
miller_madow
grassberger
schurmann
schurmann_generalised
bub
chao_shen
zhang
bonachela
shrink
chao_wang_jost
unseen
```

## Bayesian Estimators

```@docs
bayes
jeffrey
laplace
schurmann_grassberger
minimax
nsb
ansb
pym
```

## Other Estimators

```@docs
wu_yang_poly
pert
jackknife
bayesian_bootstrap
```

## Types

Estimator types for developers. Estimators are either parameterised, or non-parameterised.

```@docs
AbstractEstimator
NonParameterisedEstimator
ParameterisedEstimator
```
