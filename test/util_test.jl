using DiscreteEntropy
using Test

@test logx(1.0) == 0.0
@test logx(0.0) == 0.0

@test xlogx(0.5) == 0.5 * log(0.5)
@test xlogx(0.0) == 0.0

f(x) = x + 1.0

@test DiscreteEntropy.xFx(f, 0.0) == 0.0
@test DiscreteEntropy.xFx(f, 1.0) == 2.0
@test DiscreteEntropy.xFx(log, 0.5) == 0.5 * log(0.5)

@test to_bits(0.0) == 0.0
@test to_bits(log(2)) == 1.0

@test to_bans(0.0) == 0.0
@test to_bans(log(10)) == 1.0

@test DiscreteEntropy.gammalndiff(1.0, 0.0) == 0.0

m = [1 2 3; 4 5 6]

@test DiscreteEntropy.marginal_counts(m, 1) == [6, 15]
@test DiscreteEntropy.marginal_counts(m, 2) == [5, 7, 9]
@test DiscreteEntropy.marginal_counts(m, 3) === nothing
@test DiscreteEntropy.marginal_counts(m, 1, normalise=true) ==
    [0.2857142857142857, 0.7142857142857143]
@test DiscreteEntropy.marginal_counts(m, 2, normalise=true) ==
    [0.23809523809523808, 0.3333333333333333, 0.42857142857142855]
@test DiscreteEntropy.marginal_counts(m, 3, normalise=true) === nothing

@test DiscreteEntropy.logspace(0, 0, 1) == [1.0]
@test DiscreteEntropy.logspace(0, 1, 5) ==
    [1.0, 1.7782794100389228, 3.1622776601683795, 5.623413251903491, 10.0]
