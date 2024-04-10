using DiscreteEntropy: CountData
using DiscreteEntropy
using Test


@test svector([1, 2, 3]) isa SampleVector
@test cvector([1, 2, 3]) isa CountVector
@test xivector([0.1, 0.2]) isa XiVector

@test svector([1, 2, 3]) == SampleVector([1, 2, 3])
@test cvector([1, 2, 3]) == CountVector([1, 2, 3])
@test xivector([0.5, 0.6, 0.7]) == XiVector([0.5, 0.6, 0.7])

@test from_data([1, 2], Samples) == CountData([1.0; 2.0;;], 2.0, 2)
@test from_data([1, 2], Histogram) == CountData([2.0 1.0; 1.0 1.0], 3.0, 2)


c = from_data([1,2,3,4,3,2,1], Histogram)

@test DiscreteEntropy.bins(c) == [4.0, 2.0, 3.0, 1.0]
@test DiscreteEntropy.multiplicities(c) == [1.0, 2.0, 2.0, 2.0]

d = from_data([1,2,3,4,0,3,2,1,0], Histogram)
@test DiscreteEntropy.bins(d) == [4.0, 2.0, 3.0, 1.0]
@test DiscreteEntropy.multiplicities(d) == [1.0, 2.0, 2.0, 2.0]

dk = d.K
dc = copy(d)
dc.K = 20
@test DiscreteEntropy.set_K(d, 20) == dc
@test DiscreteEntropy.set_K!(dc, dk) == d


v = [1,2,3,4,0,3,2,1,0]
@test from_data(v, Histogram, remove_zeros=true) == CountData([4.0 2.0 3.0 1.0; 1.0 2.0 2.0 2.0], 16.0, 7)
@test from_data(v, Histogram, remove_zeros=false) == CountData([0.0 4.0 2.0 3.0 1.0; 2.0 1.0 2.0 2.0 2.0], 16.0, 9)
@test from_data(v, Samples, remove_zeros=true) == CountData([2.0 1.0; 3.0 1.0], 7.0, 4)
@test from_data(v, Samples, remove_zeros=false) == CountData([2.0 1.0; 4.0 1.0], 9.0, 5)

@test from_counts(v) == d

@test from_samples(svector(v), remove_zeros=true) == CountData([2.0 1.0; 3.0 1.0], 7.0, 4)
@test from_samples(svector(v), remove_zeros=false) == CountData([2.0 1.0; 4.0 1.0], 9.0, 5)



e::Vector{Float64} = []

@test from_data(e, Histogram) == CountData([;;], 0.0, 0)
@test from_samples(svector(e)) == CountData([;;], 0.0, 0)
@test from_counts(e) == CountData([;;], 0.0, 0)

data = """
col1,col2
1,4
2,5
3,6
4,1
3,100
2,5
1,0
"""
file = IOBuffer(data)

# CSV
@test from_csv(file, 1, Samples, header=1) == CountData([2.0 1.0; 3.0 1.0], 7.0, 4)
@test from_csv(file, 2, Samples, header=1) == CountData([2.0 1.0; 1.0 5.0], 7.0, 6)
@test from_csv(file, 2, Samples, header=1, remove_zeros=true) == CountData([2.0 1.0; 1.0 4.0], 6.0, 5)

p = cvector([4,5,4,3,2,3,4,3,2,1,2,32,9])

@test DiscreteEntropy.pmf(p, 20) === nothing
@test round(sum(DiscreteEntropy.pmf(p)), digits=10) == 1.0

@test DiscreteEntropy.to_csv_string(from_counts(p)) ==
    "[5.0,1.0,4.0,3.0,32.0,1.0,2.0,3.0,9.0,1.0,3.0,3.0,1.0,1.0],74,13"
