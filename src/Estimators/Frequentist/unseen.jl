using StatsBase: countmap;
using Distributions: Poisson, pdf;
using JuMP;
using GLPK;

function linprog(c, A, sense, b, l, u)
    N = length(c)
    model = Model(GLPK.Optimizer)
    @variable(model, l[i] <= x[i=1:N] <= u[i])
    @objective(model, Min, c' * x)
    # eq_rows, ge_rows, le_rows = sense .== '=', sense .== '>', sense .== '<'
    # @constraint(model, A[eq_rows, :] * x .== b[eq_rows])
    # @constraint(model, A[ge_rows, :] * x .>= b[ge_rows])
    # @constraint(model, A[le_rows, :] * x .<= b[le_rows])
    # optimize!(model)
    return (
        status = termination_status(model),
        objval = objective_value(model),
        sol = value.(x)
    )
end


# function linprog(c, A, b, Aeq, beq, l, u)
#     N = length(c)
#     model = Model(GLPK.Optimizer)
#     @variable(model, l[i] <= x[i=1:N] <= u[i])
#     @objective(model, Min, c' * x)
#     @constraint(model, A * x .<= b)
#     @constraint(model, Aeq * x = beq)
#     optimize!(model)
#     return (
#         status = termination_status(model),
#         objval = objective_value(model),
#         sol = value.(x)
#     )
# end

function unseen(data::CountData)
  finger_dict = countmap(data.multiplicities[2, :])
  finger = zeros(findmax(finger_dict)[1])

  for (k, v) in finger_dict
    finger[convert(Int, k)+1] = v
  end

  finger = finger[2:end]


  grid_factor = 1.05
  alpha = 0.5
  #
  xLPmin = 1 / (data.K * max(10, data.K))
  #
  min_i = minimum(filter(x -> x > 0, finger))
  #
  #
  if min_i > 1
    xLPmin = min_i / data.K
  end

  x = 0
  histx = 0

  f_lp = zeros(Int, length(finger))

  for i in 1:length(finger)
    if finger[i] > 0
      wind = [max(1, i - ceil(Int, sqrt(i))), min(i + ceil(Int, sqrt(i)), length(finger))]
      if sum(finger[wind]) < sqrt(i)
        x = [x, i / data.K]
        histx = [histx, finger[i]]
        f_lp[i] = 0
      else
        f_lp[i] = finger[i]
      end
    end
  end

  f_max = argmax(f_lp)

  if f_max <= 0
    return -1
  end


  LP_mass = 1 - x * histx

  f_lp = append!(f_lp[1:f_max], zeros(ceil(Int, sqrt(f_max))))

  xLPmax = f_max / data.K

  xLP = xLPmin .* grid_factor .^ (0:ceil(log(xLPmax / xLPmin) / log(grid_factor)))

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

  return objf, A, b, Aeq, LP_mass


end

