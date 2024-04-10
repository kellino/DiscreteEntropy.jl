using DiscreteEntropy
using Test

P = cvector([1,2,3,2,3,4,5,4,3,4])
Q = cvector([10,4,3,9,12,14,8,4,10,32])

@test cross_entropy(P, P, MaximumLikelihood) ≈ estimate_h(from_counts(P), MaximumLikelihood)
@test cross_entropy(Q, Q, MaximumLikelihood) ≈ estimate_h(from_counts(Q), MaximumLikelihood)
@test cross_entropy(P, P, Bayes, 0.0) ≈ estimate_h(from_counts(P), MaximumLikelihood)

@test kl_divergence(P, P, MaximumLikelihood) == 0.0
# @test kl_divergence(P, P, MillerMadow) == 0.0
# @test kl_divergence(P, P, LaPlace) == 0.0
# @test kl_divergence(P, P, Jeffrey) == 0.0
# @test kl_divergence(P, P, SchurmannGrassberger) == 0.0
# @test kl_divergence(P, P, Minimax) == 0.0

# @test jensen_shannon_distance(P, P, MillerMadow) == 0.0
# @test jeffreys_divergence(P, P, MaximumLikelihood) == 0.0
