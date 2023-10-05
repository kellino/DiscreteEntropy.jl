include("EvaluateEstimators.jl")
include("InfoTheory/mutual_information.jl")
include("InfoTheory/entropy.jl")

using Distributions
using EmpiricalDistributions
using StatsBase: fit, Histogram as hist
using FreqTables
using Plots


function odd_even(x)
    if iseven(x)
        return 0 #even
    else
        return 1 #odd
    end
end


# number of inputs
n = 10000
# support set size
sup_ss = 6
# sample size
ss = 1000

# bell shape: α = 5.0, β = 5.0
α = 5.0
β = 5.0
dist = BetaBinomial(sup_ss, α, β)

x = rand(dist, n)
y = []

for i in 1:n
    push!(y, odd_even(x[i]))
end


f_xy = freqtable(x, y)
# Joint prob. distr.
j_xy = prop(f_xy)
# Marginal prob. distr.
m_x = marginal_counts(Matrix{Float64}(j_xy), 1)
m_y = marginal_counts(Matrix{Float64}(j_xy), 2)

# Ground Truth
Hx = _entropy(m_x)
Hy = _entropy(m_y)
I = mutual_information(x, y)

println("GROUND TRUTH H")
println(Hx)
println("GROUND TRUTH MI")
println(I)

X_hist = fit(hist, x, nbins = sup_ss+1)
X_dist = UvBinnedDist(X_hist)

samples = round.(rand(X_dist, ss))
data = from_samples(svector(samples),true)

#histogram(x, nbins = sup_ss + 1, show = true)
#histogram(samples, nbins = sup_ss + 1, show = true)
#plot(bar([1:sup_ss], m_X), show=true)

H_estimation(data)
MI_estimation(data)