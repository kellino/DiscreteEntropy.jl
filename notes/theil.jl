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
end

# ╔═╡ 67cf7a89-2ec5-4eff-ba61-68ad43b40fd7
begin
	raw = rand(Uniform(), 3, 3)
	gt = raw ./ sum(raw)
end

# ╔═╡ 84d1992e-7889-4e3a-8183-7a195c33425c
sum(gt)

# ╔═╡ 9dfe66f1-0c8a-40ed-b5ac-08c28214d35b
dist = Categorical(vec(gt))

# ╔═╡ 3eb2496c-8fb5-4c37-bf0f-40451b5186ca
entropy(dist)

# ╔═╡ 1ecb77d8-8ea5-41b0-b2b7-49eca1533b6a
function marginals(mm, i)
	i2s = CartesianIndices(mm)
	@inbounds i2s[i].I
end

# ╔═╡ b11aab1e-5481-49e4-820a-d3d1829a3cd6
function ent(d, estimator::Type{T}) where {T<:AbstractEstimator}
	out = []
	for i in 10:10:1000
		push!(out, estimate_h(from_data(rand(d, i), Samples), estimator))
	end
	out
end	

# ╔═╡ 364f5ad3-718d-42c1-aa4a-75f1a9a47b9f
ests = ent(dist, MaximumLikelihood)

# ╔═╡ 6148fbac-79c4-4646-95b7-20d7dfa060a0
maximum(ests)

# ╔═╡ 6e633852-cf91-4301-b220-6451cd7334f9
md"""
now we have to figure out how to get the marginals from the counts
"""

# ╔═╡ Cell order:
# ╠═d2c5f75e-c732-11ed-2d67-933a47e47531
# ╠═67cf7a89-2ec5-4eff-ba61-68ad43b40fd7
# ╠═84d1992e-7889-4e3a-8183-7a195c33425c
# ╠═9dfe66f1-0c8a-40ed-b5ac-08c28214d35b
# ╠═3eb2496c-8fb5-4c37-bf0f-40451b5186ca
# ╠═1ecb77d8-8ea5-41b0-b2b7-49eca1533b6a
# ╠═b11aab1e-5481-49e4-820a-d3d1829a3cd6
# ╠═364f5ad3-718d-42c1-aa4a-75f1a9a47b9f
# ╠═6148fbac-79c4-4646-95b7-20d7dfa060a0
# ╟─6e633852-cf91-4301-b220-6451cd7334f9
