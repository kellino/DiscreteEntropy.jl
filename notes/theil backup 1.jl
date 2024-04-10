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
function theil_est(d, size, estimator::Type{T}, start, step, stop) where {T<:AbstractEstimator}
	range = start:step:stop
	out = zeros(length(range))
	Threads.@threads for i in 1:length(range)
		samples = rand(d, range[i])
		ctm = to_contingency_matrix(samples, size)
		out[i] = uncertainty_coefficient(ctm, estimator)
	end
	out
end	

# ╔═╡ 18b4206a-b316-4b94-a4de-11dfe8a7e161
function run(start, step, stop, estimator)
	sz = rand(2:1000, 2)
	raw = rand(Uniform(), sz[1], sz[2])
	P = raw ./ sum(raw)
	dist = Categorical(vec(P))
	truth = true_thiel(P)
	ests = theil_est(dist, size(P), estimator, start, step, stop)
	sz, truth .- ests
end

# ╔═╡ cfd4adf3-1a7d-44df-8277-cfa2a66ac209
function plot_theil(start, step, stop, estimator; ylims=nothing)
	sz, res = run(start, step, stop, estimator)

	if !isnothing(ylims)
		plot(start:step:stop, res, label=string(estimator), xlabel="no. samples", ylabel="bias", ylims=ylims, title="Theil's U Estimation on $sz matrix", minorgrid=true)
	else 
		plot(start:step:stop, res, label=string(estimator), xlabel="no. samples", ylabel="bias", title="Theil's U Estimation on $sz matrix", minorgrid=true)
	end
end

# ╔═╡ 5e778def-b89f-48a9-8752-aeb8ee424ba7
plot_theil(100, 100, 100_000, ChaoShen, ylims=(-0.2, 0.1))

# ╔═╡ a09b0732-956d-477f-9c7f-56f90d7f9320
plot_theil(100, 100, 100_000, ChaoWangJost, ylims=(-0.2, 0.1))

# ╔═╡ e4b06d53-d5d9-4869-9c7b-41cd8d9ae551
plot_theil(100, 100, 100_000, Schurmann)

# ╔═╡ 7c869efa-980e-44a1-90f9-2b71e8b6e143
plot_theil(100, 100, 100_000, Zhang)

# ╔═╡ aacadd5e-18d1-4245-9f06-f9f61826f0a9
plot_theil(100, 100, 1000, Bonachela)

# ╔═╡ 084c5339-cbab-45a6-961d-c7fad1752e38
plot_theil(100, 100, 100_000, Grassberger)

# ╔═╡ dc4391b0-55d5-4891-ba15-df2d421bddea
plot_theil(100, 100, 100_000, MillerMadow)

# ╔═╡ a52f6136-909e-4e4b-8d0b-92f331db3eb0
plot_theil(100, 100, 100_000, MaximumLikelihood)

# ╔═╡ a5416eb7-920c-4a1e-bd39-de29118239cb
plot_theil(100, 100, 100_000, Shrink)

# ╔═╡ 1338b59c-06ed-4909-8ae4-b49288096c19
plot_theil(100, 100, 100_000, LaPlace)

# ╔═╡ 74bcc9c4-227a-44f3-bcc7-10851d79964c
plot_theil(100, 100, 100_000, Minimax)

# ╔═╡ 0908a4f7-5e52-437a-97c7-9ff269174b65
plot_theil(100, 100, 100_000, SchurmannGrassberger)

# ╔═╡ 64101851-9fab-4fa6-becc-97d4edbc4a49
plot_theil(100, 100, 100_000, ANSB)

# ╔═╡ be48eab1-afb8-411e-b127-164ecf40fb81
plot_theil(10, 100, 1000, AutoNSB)

# ╔═╡ ae593aa5-e9ac-4690-b22e-70bb7ebfab1e


# ╔═╡ Cell order:
# ╠═d2c5f75e-c732-11ed-2d67-933a47e47531
# ╠═ccd66770-34fb-4065-9501-41f5bf6206ff
# ╠═b03a1272-0fdf-4055-a95f-28b4723fccd3
# ╠═fedaf719-d206-410a-b9b6-a9d147a1caa2
# ╠═b11aab1e-5481-49e4-820a-d3d1829a3cd6
# ╠═18b4206a-b316-4b94-a4de-11dfe8a7e161
# ╠═cfd4adf3-1a7d-44df-8277-cfa2a66ac209
# ╠═5e778def-b89f-48a9-8752-aeb8ee424ba7
# ╠═a09b0732-956d-477f-9c7f-56f90d7f9320
# ╠═e4b06d53-d5d9-4869-9c7b-41cd8d9ae551
# ╠═7c869efa-980e-44a1-90f9-2b71e8b6e143
# ╠═aacadd5e-18d1-4245-9f06-f9f61826f0a9
# ╠═084c5339-cbab-45a6-961d-c7fad1752e38
# ╠═dc4391b0-55d5-4891-ba15-df2d421bddea
# ╠═a52f6136-909e-4e4b-8d0b-92f331db3eb0
# ╠═a5416eb7-920c-4a1e-bd39-de29118239cb
# ╠═1338b59c-06ed-4909-8ae4-b49288096c19
# ╠═74bcc9c4-227a-44f3-bcc7-10851d79964c
# ╠═0908a4f7-5e52-437a-97c7-9ff269174b65
# ╠═4d28c900-3a1e-4c66-a9a6-78dc4700e5d7
# ╠═64101851-9fab-4fa6-becc-97d4edbc4a49
# ╠═be48eab1-afb8-411e-b127-164ecf40fb81
# ╠═ae593aa5-e9ac-4690-b22e-70bb7ebfab1e
