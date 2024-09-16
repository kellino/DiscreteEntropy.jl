# Estimators

We split the estimators into two broad categories, which we call _Frequentist_ and _Bayesian_. We also
have a few composite estimators that either take an averaging or resampling approach to estimation.

```@docs
AbstractEstimator
NonParameterisedEstimator
ParameterisedEstimator
```

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

## Mixed Estimators

```@docs
pert
jackknife
bayesian_bootstrap
```
