# DiscreteEntropy

A [Julia] (http://julialang.org) package for the estimation of Shannon entropy of discrete distributions.

## Multiplicities
DiscreteEntropy uses the multiplicities representation of data. Given a histogram of samples

## Frequentist Estimators

```@docs
maximum_likelihood
miller_madow
grassberger
schurmann
schurmann_generalised
zhang
chao_shen
bonachela
```

## Bayesian Estimators

```@docs
bayes
nsb
ansb
pym
```


## Utilities

```@docs
xlogx
to_bits
to_bans
```
