### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ d2c5f75e-c732-11ed-2d67-933a47e47531
begin
	import Pkg;
	Pkg.activate(Base.current_project())
	Pkg.instantiate()
	using Distributions
	using DiscreteEntropy
	using Plots
	using Random
end

# ╔═╡ ccd66770-34fb-4065-9501-41f5bf6206ff
Random.seed!(42)

# ╔═╡ b03a1272-0fdf-4055-a95f-28b4723fccd3
function true_thiel(contingency_matrix)
	X = [sum(x) for x in eachcol(contingency_matrix)]
	mutual_information(contingency_matrix, MaximumLikelihood) / maximum_likelihood(from_data(X, Histogram))
end

# ╔═╡ fedaf719-d206-410a-b9b6-a9d147a1caa2
function to_contingency_matrix(samples, size)
	mat = zeros(size)
	for s in samples
		@inbounds mat[s] += 1
	end
	mat
end

# ╔═╡ b11aab1e-5481-49e4-820a-d3d1829a3cd6
function theil_est(d, size, estimator::Type{T}, step, lim) where {T<:AbstractEstimator}
	out = zeros(floor(Int, lim/step))
	Threads.@threads for i in 10:step:lim
		samples = rand(d, i)
		ctm = to_contingency_matrix(samples, size)
		out[floor(Int, i / step)] = uncertainty_coefficient(ctm, estimator)
	end
	out
end	

# ╔═╡ 18b4206a-b316-4b94-a4de-11dfe8a7e161
function run(step, lim, estimator)
	sz = rand(2:1000, 2)
	raw = rand(Uniform(), sz[1], sz[2])
	P = raw ./ sum(raw)
	dist = Categorical(vec(P))
	truth = true_thiel(P)
	ests = theil_est(dist, size(P), estimator, step, lim)
	truth .- ests
end

# ╔═╡ bb5846a3-0104-4377-83f1-c083c4a3807d
plot(run(10, 100000, ChaoShen), label="Chao Shen", xlabel="samples", ylabel="bias")

# ╔═╡ 996aaa02-2016-493d-b60d-92b458d2dc58
plot(run(10, 10_0000, Schurmann), label="Schurmann", xlabel="samples", ylabel="bias")

# ╔═╡ 874262bb-84bc-40c9-a553-ce031bb47baf
plot(run(10, 10_0000, Zhang), label="Zhang", xlabel="samples", ylabel="bias")

# ╔═╡ 5f9584ce-0496-4c8c-83de-57651e123234
plot(run(10, 10_0000, Bonachela), label="Bonachela", xlabel="samples", ylabel="bias")

# ╔═╡ 74fb4d41-9faa-4317-9d40-6002ae0d8bfc
plot(run(10, 10_0000, MillerMadow), label="MillerMadow", xlabel="samples", ylabel="bias")

# ╔═╡ b0540708-1c20-41bb-ae47-c93661bb8ed4
plot(run(10, 10_0000, ChaoWangJost), label="ChaoWangJost", xlabel="samples", ylabel="bias")

# ╔═╡ 19d88f27-2e16-43d8-b0c5-0800c22c1525
plot(run(10, 10_0000, MaximumLikelihood), label="MaximumLikelihood", xlabel="samples", ylabel="bias")

# ╔═╡ Cell order:
# ╠═d2c5f75e-c732-11ed-2d67-933a47e47531
# ╠═ccd66770-34fb-4065-9501-41f5bf6206ff
# ╠═b03a1272-0fdf-4055-a95f-28b4723fccd3
# ╠═fedaf719-d206-410a-b9b6-a9d147a1caa2
# ╠═b11aab1e-5481-49e4-820a-d3d1829a3cd6
# ╠═18b4206a-b316-4b94-a4de-11dfe8a7e161
# ╠═bb5846a3-0104-4377-83f1-c083c4a3807d
# ╠═996aaa02-2016-493d-b60d-92b458d2dc58
# ╠═874262bb-84bc-40c9-a553-ce031bb47baf
# ╠═5f9584ce-0496-4c8c-83de-57651e123234
# ╠═74fb4d41-9faa-4317-9d40-6002ae0d8bfc
# ╠═b0540708-1c20-41bb-ae47-c93661bb8ed4
# ╠═19d88f27-2e16-43d8-b0c5-0800c22c1525
