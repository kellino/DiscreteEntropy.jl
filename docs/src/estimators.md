# Estimators

We split the estimators into two broad categories, which we call *Frequentist* and *Bayesian*. We also
have a few composite estimators that either take an averaging or resampling approach to estimation.

## Frequentist Estimators

```@docs
maximum_likelihood
jackknife_mle
miller_madow
schurmann
schurmann_generalised
chao_shen
zhang
bonachela
shrink
chao_wang_jost
```

## Bayesian Estimators

```@docs
bayes
jeffrey
laplace
schurmann_grassberger
minimax
nsb
pym
```

## Mixed Estimators

```@docs
pert
jackknife
bayesian_bootstrap
```
