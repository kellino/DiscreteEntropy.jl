using DiscreteEntropy
using Test

# randomised tests are difficult for this module, so
# results are checked against default implementations


c = from_data([1,2,3,2,1], Histogram)

# from R entropy library
@test round(estimate_h(c, ChaoShen), digits=6) == 1.876548
@test round(estimate_h(c, MaximumLikelihood), digits=6) == 1.522955
@test round(estimate_h(c, Shrink), digits=6) == 1.609438
@test round(estimate_h(c, MillerMadow), digits=6) == 1.745177
@test round(estimate_h(c, LaPlace), digits=6) == 1.574097
@test round(estimate_h(c, Jeffrey), digits=6) == 1.556911
@test round(estimate_h(c, Minimax), digits=6) == 1.561237
@test round(estimate_h(c, SchurmannGrassberger), digits=6) == 1.539698
@test round(estimate_h(c, Bayes, 3.0), digits=6) == 1.597417
@test round(estimate_h(c, Bayes, 0.2), digits=6) == 1.539698

xi = ℯ^(-1/2)

@test estimate_h(c, Schurmann, 0.7) ==
    estimate_h(cvector([1,2,3,2,1]), SchurmannGeneralised, xivector([0.7, 0.7, 0.7, 0.7, 0.7]))
@test estimate_h(c, Schurmann) ==
    estimate_h(cvector([1,2,3,2,1]), SchurmannGeneralised, xivector(fill(xi, 5)))

# the matlab PYM implementation has much more limited precision that Julia, so we round to digits=3
@test round(estimate_h(c, PYM), digits=3) == 2.674

@test round(estimate_h(c, Zhang), digits=6) == 1.773413
@test round(estimate_h(c, Bonachela), digits=6) == 1.54045
@test round(estimate_h(c, ChaoWangJost), digits=6) == 1.84787

# we test BUB against the author's own implemention, though we have to do a little bit of manual rounding as julia's precision
# is quite different from matlab's
@test round(estimate_h(from_counts([1,2,3,4,5,4,3,2,1]), BUB, truncate=true), digits=4) == 2.2388
@test round(estimate_h(from_counts([1,2,3,2,1]), BUB, truncate=true), digits=4) == 1.7089
@test round(estimate_h(from_counts([1]), BUB, truncate=true), digits=4) == 0.1812



# testing NSB is difficult, as every implementation I've seen gives a different answer, so instead
# we settle for a regression test approach
@test round(estimate_h(c, NSB), digits=6) == 1.569751
@test round(estimate_h(c, ANSB), digits=6) == 3.0224

@test estimate_h(c, MaximumLikelihood) ≈ estimate_h(c, PERT, MaximumLikelihood, MaximumLikelihood)
