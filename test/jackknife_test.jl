using DiscreteEntropy
using Test

function oracle(samples, estimator::Type{T}) where {T<:AbstractEstimator}
    tot = estimate_h(from_data(samples, Samples), estimator)
    for i in 1:length(samples)
        orig = samples[i]
        samples[i] = 0
        new = filter(x -> x > 0, samples)
        cd = from_data(new, Samples)
        println("$new => $cd")
        tot += estimate_h(cd, estimator)
        samples[i] = orig
    end
    return tot / (length(samples) + 1)
end

t1 = [1, 2, 2, 3, 3, 3, 4, 4, 5]
t2 = [4, 5, 4, 3, 2, 1, 2, 3, 5]
# t3 = [1, 2, 3, 4, 5, 6, 7] Failing test


@test oracle(t1, MaximumLikelihood) ≈ jackknife_mle(from_data(t1, Samples))[1]
@test oracle(t2, MaximumLikelihood) ≈ jackknife_mle(from_data(t2, Samples))[1]
# @test oracle(t3, MaximumLikelihood) == jackknife_ml(from_data(t3, Samples))[1]
