using JuMP, Ipopt, Distributions, Statistics

bernoulli = Bernoulli()
samples = from_samples(rand(bernoulli, 100))

function sch(ξ)
    abs(schurmann(samples, ξ) - entropy(bernoulli))
end


function test()
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, ξ >= 0, start = 0.1)
    register(model, :sch, 1, sch; autodiff=true)
    @NLobjective(model, Min, sch(ξ))
    optimize!
end

# function example()
#     bernoulli = Bernoulli()
#     samples = from_samples(rand(bernoulli, 100))
# model = Model(Ipopt.Optimizer)
# @variable(model, ξ >= 0)
# bias(ξ) = schurmann(samples, ξ) - entropy(bernoulli)
# register(model, :bias, 1, bias; autodiff=true)
# @NLobjective(model, Min, bias(ξ))
# optimize!(model)


#     # f(samples, ξ) = schurmann(samples, ξ)
#     # register(model, :foo, 2, f, autodiff=true)
#     # set_silent(model)
#     # @NLobjective(model, Min, foo(samples, ξ) - entropy(bernoulli))
#     # optimize!(model)
# end


# # If the input dimension is more than 1
# x = [1.0, 2.0]
# my_function(a, b) = a^2 + b^2
# ForwardDiff.gradient(x -> my_function(x...), x)
# ```

# ```julia
# import ForwardDiff

# # If the input dimension is 1
# x = 1.0
# my_function(a) = a^2
# ForwardDiff.derivative(my_function, x)

# # If the input dimension is more than 1
# x = [1.0, 2.0]
# my_function(a, b) = a^2 + b^2
# ForwardDiff.gradient(x -> my_function(x...), x)
# ```
