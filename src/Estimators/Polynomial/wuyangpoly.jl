using JLD2;

@doc raw"""
  wu_yang_poly(data::CountData; L::Int=0, M::Float64=0.0, N::Int=0)

Compute the Wu Yang Polynomial [`wu_yang_poly`](@ref) estimate of data. This implementation uses the precomputed coefficients 
found [here](https://github.com/Albuso0/entropy)

# Optional Parameters
L::Int : polynomial degree, default = floor(1.6 * log(data.K))
M::Float64 : endpoint of approximation interval, default = 3.5 * log(data.K)
N::Int : threshold for polynomial estimator application, default = floor(1.6 * log(data.K))

# External Links
[Minimax Rates of Entropy Estimation on Large Alphabets via Best Polynomial Approximation](https://ieeexplore.ieee.org/document/7444171)
"""
function wu_yang_poly(data::CountData; L::Int=0, M::Float64=0.0, N::Int=0)
  if iszero(L)
    degree = convert(Int, floor(1.6 * log(data.K)))
  else
    degree = L
  end

  if iszero(M)
    ratio = 3.5 * log(data.K)
  else
    ratio = M
  end

  if iszero(N)
    threshold = degree
  else
    threshold = N
  end


  coeff_path = pkgdir(DiscreteEntropy, "data", "polydata.jld2")
  a_coeffs = load(coeff_path, "data")[degree]

  g_coeffs = zeros(threshold + 1)

  for j in 0:threshold
    if j > degree
      start = degree
    else
      start = j
    end
    g_coeffs[j+1] = a_coeffs[start+1]
    for i in reverse(1:start)
      g_coeffs[j+1] = a_coeffs[i] + g_coeffs[j+1] * (j - i + 1) / ratio
    end
    g_coeffs[j+1] = (g_coeffs[j+1] * ratio + log(data.N / ratio) * j) / data.N
  end

  h_estimate = 0
  sym_num = 0
  for x in eachcol(data.multiplicities)
    freq = convert(Int, x[1]) + 1
    sym_num += x[2]
    if x[1] > threshold
      p_hat = 1.0 * x[1] / data.N
      h_estimate += (-p_hat * log(p_hat) + 0.5 / data.N) * x[2]
    else
      h_estimate += g_coeffs[freq] * x[2]
    end
  end


  h_estimate += 1 * g_coeffs[1] + (data.K - sym_num)
  if h_estimate < 0
    h_estimate = 0
  end

  return h_estimate
end
