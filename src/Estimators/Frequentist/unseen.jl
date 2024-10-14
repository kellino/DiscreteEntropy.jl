using StatsBase: countmap;
using Distributions: Poisson, pdf;
using JuMP;
using GLPK;

function linprog(c, A, b, Aeq, beq, lb, ub)
  model = Model(GLPK.Optimizer)
  N = length(lb)
  @variable(model, lb[i] <= x[i=1:N] <= ub[i])
  @objective(model, Min, c'x)
  @constraint(model, A * x .<= b)
  @constraint(model, Aeq' * x .== beq)
  optimize!(model)

  sol = value.(x)
  val = objective_value(model)
  return sol, val
end

@doc raw"""
  unseen(data::CountData)

Compute the Unseen estimatation of Shannon entropy.

# Example

``@jldoctest
n = [1,2,3,4,5,4,3,2,1]
h = unseen(from_counts(n))
1.4748194918254784

# External Links
[Estimating the Unseen: Improved Estimators for Entropy and Other Properties](https://drive.google.com/file/d/1mdmbAZm22uH-Shr18YQTtKeMnlwypnpp/view)
"""
function unseen(data::CountData)
  finger_dict = countmap(data.multiplicities[2, :])
  finger = zeros(convert(Int, findmax(collect(keys(finger_dict)))[1]))

  for (k, v) in finger_dict
    finger[convert(Int, k)] = v
  end

  grid_factor = 1.05
  alpha = 0.5

  xLPmin = 1 / (data.K * max(10, data.K))

  min_i = minimum(findall(x -> x > 0, finger))


  if min_i > 1
    xLPmin = min_i / data.K
  end

  x::Vector{Float32} = [0.0]
  histx = [0]

  f_lp = zeros(Int, length(finger))

  for i in 1:length(finger)
    if finger[i] > 0
      wind = [max(1, i - ceil(Int, sqrt(i))), min(i + ceil(Int, sqrt(i)), length(finger))]
      if sum(finger[wind[1]:wind[2]]) < sqrt(i)
        append!(x, i / data.K)
        append!(histx, finger[i])
        f_lp[i] = 0
      else
        f_lp[i] = finger[i]
      end
    end
  end

  f_max_list = findall(x -> x > 0, f_lp)

  if isempty(f_max_list)
    x = x[2:end]
    h = histx[2:end]
    return abs(sum(-h .* (x .* log.(x))))
  else
    f_max = maximum(f_max_list)
  end


  LP_mass = 1 - (sum(x .* histx))

  f_lp = append!(f_lp[1:f_max], zeros(ceil(Int, sqrt(f_max))))

  xLPmax = f_max / data.K

  xLP = xLPmin * grid_factor .^ (0:ceil(log(xLPmax / xLPmin) / log(grid_factor)))

  objf = zeros(length(xLP) + 2 * length(f_lp))
  objf[length(xLP)+1:2:end] .= 1 ./ sqrt.(f_lp .+ 1)
  objf[length(xLP)+2:2:end] .= 1 ./ sqrt.(f_lp .+ 1)

  A = zeros(2 * length(f_lp), length(xLP) + 2 * length(f_lp))
  b = zeros(2 * length(f_lp))
  for i = 1:length(f_lp)
    A[2*i-1, 1:length(xLP)] = pdf.(Poisson.(data.K * xLP), i)
    A[2*i, 1:length(xLP)] = (-1) .* A[2*i-1, 1:length(xLP)]
    A[2*i-1, length(xLP)+2*i-1] = -1
    A[2*i, length(xLP)+2*i] = -1
    b[2*i-1] = f_lp[i]
    b[2*i] = -f_lp[i]
  end

  Aeq = zeros(length(xLP) + 2 * length(f_lp))
  Aeq[1:length(xLP)] = xLP

  for i in 1:length(xLP)
    A[:, i] = A[:, i] / xLP[i]
    Aeq[i] = Aeq[i] / xLP[i]
  end

  lb = zeros(length(xLP) + 2 * length(f_lp))
  ub = Inf * ones(length(xLP) + 2 * length(f_lp))
  _, fval = linprog(objf, A, b, Aeq, LP_mass, lb, ub)

  objf2 = 0 .* objf
  objf2[1:length((xLP))] .= 1

  A2 = [A; objf']

  b2 = [b; fval + alpha]


  for i in 1:length(xLP)
    objf2[i] = objf2[i] / xLP[i]
  end

  sol2, _ = linprog(objf2, A2, b2, Aeq, LP_mass, lb, ub)

  sol2[1:length(xLP)] .= sol2[1:length(xLP)] ./ xLP

  x = [x; xLP]
  histx = [histx; sol2]
  inds = sortperm(x)
  x = sort(x)
  histx = histx[inds]
  ind = findall(x -> x > 0.0, histx)

  h = histx[ind]
  x = x[ind]

  return sum(-h .* (x .* log.(x)))
end

