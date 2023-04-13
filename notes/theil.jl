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

# ╔═╡ ae593aa5-e9ac-4690-b22e-70bb7ebfab1e
begin
	Pkg.add("Changepoints")
end

# ╔═╡ 2e696cd1-15bd-45d9-a26b-31dd68c8105b
using Changepoints

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
# ╠═╡ disabled = true
#=╠═╡
plot_theil(10, 100, 1000, AutoNSB)
  ╠═╡ =#

# ╔═╡ 1bdfbefe-ae28-4974-90d7-47f784f8d23c
sz, bs = run(10, 10, 10_000, ChaoShen)

# ╔═╡ 5308d85a-b5d5-43a3-8f27-194ae8f0c8f4
pelt_cps, cost = @PELT bs Normal(0.0, :?)

# ╔═╡ 90cea916-cf87-48f0-9e6b-02a2c4e728ec
changepoint_plot(bs, pelt_cps)

# ╔═╡ 8a69207a-2968-45fb-8f89-ec794c80bf62
cost

# ╔═╡ 99e4c487-e1a7-4141-a40b-92ce60f50add
bs_cps = @BS bs Normal(0.0, :?)

# ╔═╡ 2e911a10-a11f-46de-a083-8857d040c5aa
begin
	changepoint_plot(bs, bs_cps[1])
	hline!([0.0], c=:lightblue, w=2)
end

# ╔═╡ 5c6435d5-8599-4fb3-aef4-0233cd2854c8
begin 
	cps = bs_cps[1]
	out = []
	for i in 2:length(cps)-1
		segment = bs[cps[i]: cps[i+1]]
		μ = mean(segment)
		σ = std(segment)
		push!(out, (μ, σ))
	end
	for elem in out
		println(elem)
	end
end

# ╔═╡ 1a07c523-ff8e-4d06-880d-7ae6378a6426
plot([x[2] for x in out])

# ╔═╡ 4f954e39-7811-4529-9f5e-5773dd4bd026
_, bsm = run(10, 10, 100_000, Schurmann)

# ╔═╡ 07fca792-071a-4427-8a5f-13689b4a6567
plot(bsm)

# ╔═╡ 492249d8-d7ee-4ef1-978d-de9fb052fb23
begin
	crops_output = @PELT bsm Normal(0.0, :?) 4.0 100.00
end

# ╔═╡ af62087c-39e1-4ca8-b906-0684f6af5c09
mean(bsm)

# ╔═╡ 45cb48b2-1a48-4594-8f96-1034eb87d78d
std(bsm)

# ╔═╡ 8ed0acb0-9c2b-4932-8558-4452fadb9bca
elbow_plot(crops_output)

# ╔═╡ 40ed2d03-1b1f-4ffd-b449-fa1783a75052
function cs_aver(start, step, stop)
	_, bs = run(start, step, stop, ChaoShen)
	bs_cps = @BS bs Normal(0.0, :?)
	cps = bs_cps[1]
	out = []
	for i in 2:length(cps)-1
		segment = bs[cps[i]: cps[i+1]]
		μ = mean(segment)
		σ = std(segment)
		push!(out, (μ, σ))
	end
	for i in 1:length(out)
		if abs(out[i][1]) < 0.01
			return cps[i] * step
		end
	end
	return NaN
end	

# ╔═╡ 817401b2-ab87-41a8-a98a-b6b9f97ecb27
pos = cs_aver(10, 10, 10000)

# ╔═╡ c6ea1ebb-b7e5-43db-8c2f-01e773eecdbb
cps_

# ╔═╡ cd31641d-6631-4509-a7fe-a94bb27eda3f
pos

# ╔═╡ 92b1e71a-e7c4-419a-bc40-b21399802ee2
N = 50

# ╔═╡ eb6de3d7-c905-4e7e-bb23-bd027400f95c
function run_(N)
	avg = zeros(N)
	for i in 1:N
		t = cs_aver(10, 10, 10_000)
		if !isnan(t)
			avg[i] = t
		end
	end
	return mean(avg), std(avg), mean(avg)/N
end

# ╔═╡ 98602c18-0a9f-46d6-960b-778526400994
run_(50)

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
# ╠═64101851-9fab-4fa6-becc-97d4edbc4a49
# ╟─be48eab1-afb8-411e-b127-164ecf40fb81
# ╠═ae593aa5-e9ac-4690-b22e-70bb7ebfab1e
# ╠═2e696cd1-15bd-45d9-a26b-31dd68c8105b
# ╠═1bdfbefe-ae28-4974-90d7-47f784f8d23c
# ╠═5308d85a-b5d5-43a3-8f27-194ae8f0c8f4
# ╠═90cea916-cf87-48f0-9e6b-02a2c4e728ec
# ╠═8a69207a-2968-45fb-8f89-ec794c80bf62
# ╠═99e4c487-e1a7-4141-a40b-92ce60f50add
# ╠═2e911a10-a11f-46de-a083-8857d040c5aa
# ╠═5c6435d5-8599-4fb3-aef4-0233cd2854c8
# ╠═1a07c523-ff8e-4d06-880d-7ae6378a6426
# ╠═4f954e39-7811-4529-9f5e-5773dd4bd026
# ╠═07fca792-071a-4427-8a5f-13689b4a6567
# ╠═492249d8-d7ee-4ef1-978d-de9fb052fb23
# ╠═af62087c-39e1-4ca8-b906-0684f6af5c09
# ╠═45cb48b2-1a48-4594-8f96-1034eb87d78d
# ╠═8ed0acb0-9c2b-4932-8558-4452fadb9bca
# ╠═40ed2d03-1b1f-4ffd-b449-fa1783a75052
# ╠═817401b2-ab87-41a8-a98a-b6b9f97ecb27
# ╠═c6ea1ebb-b7e5-43db-8c2f-01e773eecdbb
# ╠═cd31641d-6631-4509-a7fe-a94bb27eda3f
# ╠═92b1e71a-e7c4-419a-bc40-b21399802ee2
# ╠═eb6de3d7-c905-4e7e-bb23-bd027400f95c
# ╠═98602c18-0a9f-46d6-960b-778526400994
