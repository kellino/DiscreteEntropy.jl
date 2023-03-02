# DiscreteEntropy

A [Julia] (http://julialang.org) package for the estimation of Shannon entropy of discrete distributions.

## Multiplicities
DiscreteEntropy uses the multiplicities representation of data. Given a histogram of samples

## Frequentist Estimators

```@docs
maximum_likelihood
jackknife_ml
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

## Resampling
We can also resample data

```@docs
jackknife
```

## Divergence
```@docs
kl_divergence
jeffreys_divergence
jensen_shannon_divergence
jensen_shannon_distance
```

## Conditional Entropy and Conditional Mutual Information
[Conditional Entropy](https://en.wikipedia.org/wiki/Conditional_entropy)
```@docs
conditional_entropy
```

## Utilities

```@docs
logx
xlogx
to_bits
to_bans
```

