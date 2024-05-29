using DiscreteEntropy
using Test

m = [4 9 4 5 8; 10 2 7 9 6; 3 5 6 9 6; 4 2 1 5 8; 4 5 43 8 3]
x = DiscreteEntropy.marginal_counts(m, 1)
y = DiscreteEntropy.marginal_counts(m, 2)

@test mutual_information(m, MaximumLikelihood) ≈ kl_divergence(cvector(m), cvector(y * x'), MaximumLikelihood)
@test mutual_information(Matrix([0.2 0.4; 0.2 0.4]), MaximumLikelihood) == 0.0

@test mutual_information(m, MaximumLikelihood) == estimate_h(from_data(y, Histogram), MaximumLikelihood) - conditional_entropy(m, MaximumLikelihood)
@test mutual_information(m, ChaoWangJost) == estimate_h(from_data(y, Histogram), ChaoWangJost) - conditional_entropy(m, ChaoWangJost)


@test conditional_entropy(from_data(x, Histogram), from_data(m, Histogram), MaximumLikelihood) - conditional_entropy(m, MaximumLikelihood) == 0.0
@test conditional_entropy(m, Bayes, 0.0) - conditional_entropy(m, MaximumLikelihood) == 0.0

# I don't have a good test for conditional_entropy with NSB, so will just go for regression for the moment
@test conditional_entropy(m, NSB) == 1.4536388136068217

@test uncertainty_coefficient(Matrix([0.2 0.4; 0.2 0.4]), MaximumLikelihood) == 0.0
@test uncertainty_coefficient(m, MaximumLikelihood) == 0.12637968128410482
@test uncertainty_coefficient(m, MaximumLikelihood, symmetric=true) == 0.12677798482043318
