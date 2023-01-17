### A Pluto.jl notebook ###
# v0.19.19

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ d3bb58d4-9271-11ed-3c17-4b633b603f59
begin
	import Pkg
	Pkg.activate("/Users/davidkelly/.julia/dev/DiscreteEntropy")
	Pkg.instantiate()

	using Plots, PlutoUI, DiscreteEntropy, Distributions, Loess, Random
end

# ╔═╡ 9c5fb067-4432-4098-bdb0-26154b23025f
@bind xi Slider(0.01:0.001:5.0)

# ╔═╡ 61c6c8f6-6c59-484b-a651-1c84d9b93ba0
xi

# ╔═╡ d68253e3-ecf5-4afb-8da1-e53fc3a31ea5
@bind distribution_style Select([Bernoulli, BetaBinomial])

# ╔═╡ 86997c98-6c1d-4806-98c4-6fe052cbc6b3
function bias(size, samples, distribution, xi)
	return entropy(distribution) - schurmann(from_samples(samples[1:size]), xi)
	#return schurmann(from_samples(samples[1:size]), xi) - entropy(distribution)
end

# ╔═╡ 82a220b2-0e0e-4015-8fb0-fd32976b6bc0
dist = distribution_style(0.5)

# ╔═╡ cabe711d-cdd4-49b0-b591-2ce6618cd6c0
entropy(dist)

# ╔═╡ cb4ce992-eb00-4f06-8192-f482a96da594
to_bits(entropy(dist))

# ╔═╡ e7f451f6-707c-45a1-bbc8-e731a9e3c044
samples = rand(dist, 10000)

# ╔═╡ f1a4d101-3108-45b0-82f9-debb04313acc
function plot()
	xs = 10:10:1000
	ys = [bias(x, samples, dist, xi) for x in xs]
	scatter(xs, ys)
	model = loess(xs, ys)
	us = range(extrema(xs)...; step = 0.1)
	vs = predict(model, us)
	plot!(us, vs, legend=false)
end

# ╔═╡ 06bffe2b-b2b0-4866-89f3-ea14ce80eedd
to_bits(schurmann(from_samples(samples[1:100]), xi))

# ╔═╡ 49a57d1e-db54-4a3c-bc80-d6e9b5da5042
plot()

# ╔═╡ Cell order:
# ╟─d3bb58d4-9271-11ed-3c17-4b633b603f59
# ╠═9c5fb067-4432-4098-bdb0-26154b23025f
# ╟─61c6c8f6-6c59-484b-a651-1c84d9b93ba0
# ╟─d68253e3-ecf5-4afb-8da1-e53fc3a31ea5
# ╟─86997c98-6c1d-4806-98c4-6fe052cbc6b3
# ╟─82a220b2-0e0e-4015-8fb0-fd32976b6bc0
# ╟─cabe711d-cdd4-49b0-b591-2ce6618cd6c0
# ╟─cb4ce992-eb00-4f06-8192-f482a96da594
# ╟─e7f451f6-707c-45a1-bbc8-e731a9e3c044
# ╟─f1a4d101-3108-45b0-82f9-debb04313acc
# ╟─06bffe2b-b2b0-4866-89f3-ea14ce80eedd
# ╠═49a57d1e-db54-4a3c-bc80-d6e9b5da5042
