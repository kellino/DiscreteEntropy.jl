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
