### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ ba45412c-b351-11ed-21ff-1d741ebf1a27
begin
	using Markdown
	import Pkg
	Pkg.activate(Base.current_project())
	Pkg.instantiate()
	using DiscreteEntropy
	using Latexify
	using SpecialFunctions: digamma
	using QuadGK
	using Distributions
	using Optim
	using Plots
	using StatsBase: countmap
	using Random
	using LaTeXStrings
end

# ╔═╡ 0d170cc3-a823-4aa3-af4d-8ae8a749bf7f
html"""
<style>
	main {
		margin: 0 auto;
		max-width: 1600px;
    	padding-left: max(300px, 10%);
    	padding-right: max(160px, 10%);
	}
</style>
"""

# ╔═╡ 9058721d-6c3a-437a-ae45-7b86b205cba3
md"""
We want to improve the Schurmann (and PYM) estimator

$\hat{H}_{SHU} = \psi(N) - \frac{1}{N} \sum_{k=1}^{K} \, y_x \Big( \psi(y_x) + (-1)^{y_x} ∫_0^{\frac{1}{\xi} - 1} \frac{t^{y_x}-1}{1+t}dt \Big)$

"""

# ╔═╡ f59e18eb-469b-4f97-80e0-fae8e14876e7
begin
function schurmann(samples::Vector, ξ::Float64)::Float64
    N = length(samples)
    map = countmap(samples)

    return digamma(N) - (1.0 / N) *
                        sum([(digamma(y) + (-1)^y * quadgk(t -> t^(y - 1) / (1 + t), 0, (1 / ξ) - 1.0)[1]) * y for (_, y) in map])
end
end

# ╔═╡ 115cf1ea-1e61-4745-9471-b5ec8ad13b52
samples = [1,2,3,2,3,4,3,2,3,2,1,2,3,4]

# ╔═╡ 995a138f-3255-46c6-9bce-169de96934c3
histogram(samples, xlabel="sample value", ylabel="count", label="counts")

# ╔═╡ 45be52fc-8979-43ba-9502-f307a2cbab68
md"""
There are two bars of 5, and two bars of 2
"""

# ╔═╡ 3b55b4e7-20fc-4b8b-9848-40a66b2fc49b
data = from_samples(samples)

# ╔═╡ 10ff7758-fc0e-44ef-bc8d-3de883044e30
md"""
We can think of this as the matrix
"""

# ╔═╡ 90df04df-0768-423f-8082-3fb5bcdb0207
latexify([5 2; 2 2])

# ╔═╡ f7f8525a-356e-4bc4-84ea-467b28bcb7a5
md"""
with N = 14 (total samples = dot product) and K = 4 (support size)
"""

# ╔═╡ 20e77956-ab46-4843-bea2-d689d153165d
begin
function get_best(dist, s)
    function loss(ξ)
        return abs(schurmann(s, ξ) - entropy(dist))
    end
    res = Optim.optimize(loss, 0.1, 10.0)
    return res
end
end


# ╔═╡ 71b5e41e-0c2d-4025-a3b8-7d32747da7ac
bernoulli = Bernoulli()

# ╔═╡ f791954e-ed12-4548-9f19-fdb5645dda18
res = get_best(bernoulli, rand(bernoulli, 100))

# ╔═╡ c16602c4-cb4c-4e80-96a7-660e206b2497
begin
	function ξ_limit()
    Random.seed!(42)
    dist = BetaBinomial(100, 1.0, 2.0)
    samples = rand(dist, 100)

    function loss(ξ)
        return abs(schurmann(samples, ξ) - entropy(dist))
    end

    xs = 0.3:0.01:10.0
    ys = [loss(ξ) for ξ in xs]
    return (xs, ys)
end
end

# ╔═╡ cb41e8d0-006a-477f-8f15-77b5bf7d8540
xs, ys = ξ_limit()

# ╔═╡ 438f0e37-be18-4d23-a246-d4cf18e9d829
begin
	m = Optim.minimizer(res)
	println("The minimiser is $(m)")
end

# ╔═╡ 1aa6c2c2-9e6a-4da8-a3ff-de1bef78c728
begin
	plot(xs, ys, xlabel=L"$\xi$", ylabel="absolute bias")
	title!("Bias on a BetaBinomial Distribution")
end

# ╔═╡ Cell order:
# ╟─ba45412c-b351-11ed-21ff-1d741ebf1a27
# ╟─0d170cc3-a823-4aa3-af4d-8ae8a749bf7f
# ╟─9058721d-6c3a-437a-ae45-7b86b205cba3
# ╟─f59e18eb-469b-4f97-80e0-fae8e14876e7
# ╟─115cf1ea-1e61-4745-9471-b5ec8ad13b52
# ╠═995a138f-3255-46c6-9bce-169de96934c3
# ╟─45be52fc-8979-43ba-9502-f307a2cbab68
# ╠═3b55b4e7-20fc-4b8b-9848-40a66b2fc49b
# ╟─10ff7758-fc0e-44ef-bc8d-3de883044e30
# ╟─90df04df-0768-423f-8082-3fb5bcdb0207
# ╟─f7f8525a-356e-4bc4-84ea-467b28bcb7a5
# ╠═20e77956-ab46-4843-bea2-d689d153165d
# ╟─71b5e41e-0c2d-4025-a3b8-7d32747da7ac
# ╠═f791954e-ed12-4548-9f19-fdb5645dda18
# ╟─c16602c4-cb4c-4e80-96a7-660e206b2497
# ╟─cb41e8d0-006a-477f-8f15-77b5bf7d8540
# ╟─438f0e37-be18-4d23-a246-d4cf18e9d829
# ╟─1aa6c2c2-9e6a-4da8-a3ff-de1bef78c728
