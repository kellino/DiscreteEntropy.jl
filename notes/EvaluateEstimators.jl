include("../src/DiscreteEntropy.jl")
include("h_estimation.jl")
include("mi_estimation.jl")
include("cmi_estimation.jl")

using .DiscreteEntropy

input_size = [25, 50, 100, 200, 400]
n_runs = 1000

for n in input_size
    entropy_estimation(n, n_runs)
    mutual_information_estimation(n, n, n_runs)
    conditional_mutual_information_estimation(n, n, n, n_runs)
end