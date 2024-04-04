using DiscreteEntropy
using StatsBase: countmap;
using Test

function update_or_insert!(d::Dict, k, v)
    if k == 0
        return d
    end
    if haskey(d, k)
        d[k] += v
    else
        d[k] = v
    end
end

function simple_jackknife(samples)
    d = Dict{CountData, Int64}()
    for i in 1:length(samples)
        orig = samples[i]
        samples[i] = 0
        new = filter(x -> x > 0, samples)
        countdata = DiscreteEntropy.from_data(new, Samples)
        mm = countdata.multiplicities
        countdata.multiplicities = sortslices(mm, dims=2)
        update_or_insert!(d, countdata, 1)
        samples[i] = orig
    end

    return d
end

t1 = [1, 2, 2, 3, 3, 3, 4, 4, 5]
t2 = [4, 5, 4, 3, 2, 1, 2, 3, 5]
t3 = [1, 2, 3, 4, 5, 6, 7]
t4 = [1, 1, 4, 10, 7, 3, 4, 2, 4, 7, 8, 7, 4, 1, 9, 4, 7, 1, 8, 10, 4, 1, 9, 7, 4, 1, 3, 3, 7, 9, 6, 7, 9, 10, 1, 10, 8, 8, 5, 5, 6, 9, 7, 3, 2, 5, 7, 9, 2, 3, 8, 7, 1, 10, 3, 2, 4, 7, 1, 3, 7, 8, 10, 7, 4, 4, 7, 10, 5, 6, 10, 3, 9, 6, 9, 10, 7, 2, 2, 9, 6, 8, 6, 5, 4, 5, 4, 4, 1, 2, 7, 2, 7, 5, 6, 1, 4, 6, 3, 7]
t5::Vector{Int64} = []

@test simple_jackknife(t1) == DiscreteEntropy.jk(from_data(t1, Samples))
@test simple_jackknife(t2) == DiscreteEntropy.jk(from_data(t2, Samples))
@test simple_jackknife(t3) == DiscreteEntropy.jk(from_data(t3, Samples))
@test simple_jackknife(t4) == DiscreteEntropy.jk(from_data(t4, Samples))
@test simple_jackknife(t5) == DiscreteEntropy.jk(from_data(t5, Samples))
