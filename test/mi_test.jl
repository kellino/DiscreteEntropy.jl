using DiscreteEntropy
using Test

m = [4 9 4 5 8; 10 2 7 9 6; 3 5 6 9 6; 4 2 1 5 8; 4 5 43 8 3]
x = DiscreteEntropy.marginal_counts(m, 1)
y = DiscreteEntropy.marginal_counts(m, 2)

@test mutual_information(m, MaximumLikelihood) ≈ kl_divergence(cvector(m), cvector(y * x'), MaximumLikelihood)
# @test mutual_information(m, MillerMadow) ≈ kl_divergence(cvector(m), cvector(y * x'), MillerMadow)

@test mutual_information(m, MaximumLikelihood) == estimate_h(from_data(y, Histogram), MaximumLikelihood) - conditional_entropy(m, MaximumLikelihood)
@test mutual_information(m, ChaoWangJost) == estimate_h(from_data(y, Histogram), ChaoWangJost) - conditional_entropy(m, ChaoWangJost)
