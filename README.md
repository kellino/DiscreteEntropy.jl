# DiscreteEntropy.jl

[![Build Status](https://github.com/kellino/DiscreteEntropy.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/kellino/DiscreteEntropy.jl/actions/workflows/CI.yml?query=branch%3Amain)

## The Julia Discrete Entropy Estimation Toolkit

A collection of discrete entropy estimators and other information theoretic tools. DiscreteEntropy aims to be a 

# Estimators to add
Ideally, DiscreteEntropy will have an efficient implementation of the most common/cited entropy estimators. It is
still missing implementations for 

+ bub
+ unseen
+ cdm


# TODO
+ tests
+ bootstrap (not a priority)
+ unseen
+ cdm
+ check bonachela, as gives weird results

# Docs
  To build docs, run ``julia --project=. docs/make.jl`` in root directory of the project.
  To view docs, open ``docs/build/index.html``
