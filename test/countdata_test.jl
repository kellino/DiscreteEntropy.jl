using DiscreteEntropy
using OrderedCollections: OrderedDict
using Test


@test from_samples([1, 2]) == DiscreteEntropy.CountData(OrderedDict(1.0 => 2), 2.0, 2)
