include("DiscreteEntropy.jl")
using .DiscreteEntropy

input_size = [25, 50, 100, 200, 400]
n_runs = 1000

for n in input_size
    mutual_information_estimation(n, n, n_runs)
end