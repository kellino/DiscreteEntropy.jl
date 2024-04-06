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
