```@meta
CurrentModule = DiscreteEntropy
DocTestSetup = quote
    using DiscreteEntropy
end

```

# DiscreteEntropy

A [Julia](http://julialang.org) package for the estimation of Shannon entropy of discrete distributions.

## Data Representation
DiscreteEntropy uses the multiplicities representation of data. 

```@docs
CountData
from_counts
from_data
from_samples
```

## Vector Types

```@docs
AbstractCounts
```
