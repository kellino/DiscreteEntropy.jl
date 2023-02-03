using SpecialFunctions;
using LinearAlgebra;
using Optim;

mm = [1, 2, 3]
icts = [2, 2, 3]

function convert_from_py_params(α::Float64, δ::Float64)
    digamma_1_minus_delta = digamma(1.0 - δ)
    eta = digamma(α + 1.0) - digamma_1_minus_delta
    gam = (digamma(1.0) - digamma_1_minus_delta) / eta

    return (eta, gam)

end

function common_prior_jacobian(α::Float64, δ::Float64)
    (eta, gam) = convert_from_py_params(α, δ)

    dha = trigamma(α + 1.0)
    dhd = trigamma(1.0 - δ)
    dga = -(gam / eta) * dha
    dgd = dhd * (digamma(α + 1.0) - digamma(1.0)) / (eta^2.0)
    J = dha * dgd - dhd * dga
    return J
end

mm = [1, 2, 3];
icts = [2, 2, 3];
eps = 10e-4;

function gammalndiff(x::Float64, dx::Float64)
    return loggamma(x + dx) - loggamma(x)
end

function logliPyOccupancy(α::Float64, δ::Float64, mm::Vector{Int64}, icts::Vector{Int64})
    N = dot(mm, icts)
    K = sum(mm)
    logp = -gammalndiff(α + 1.0, N - 1.0) +
           sum(map(x -> log(α + x * δ), 1:K-1)) +
           dot(mm, map(x -> gammalndiff(1.0 - δ, Float64(x)), icts .- 1))

    KK = 1:K-1
    dlogp = zeros(1, 2)
    Z = 1.0 ./ (α .+ KK .* δ)
    dlogp[:1] = sum(Z) - digamma(α + N) + digamma(α + 1.0)
    dlogp[:2] = dot(KK, Z) - dot(mm, digamma.(icts .- δ) .- digamma.(1.0 - δ))

    ddlogp = zeros(2, 2)
    Z2 = Z .^ 2.0
    ddlogp[1, 1] = trigamma(1.0 + α) - trigamma(α + N) - sum(Z2)
    diag = -sum(KK .* Z2)
    ddlogp[1, :2] = diag
    ddlogp[2, :1] = diag
    ddlogp[2, :2] = -sum((KK .^ 2.0) .* Z2) + dot(mm, trigamma.(icts .- δ) .- trigamma.(1.0 - δ))

    return (logp, dlogp, ddlogp)

end

function pymPrior(param::Float64, α::Float64, δ::Float64)
    prior = 1
    dprior = 0
    ddprior = 0

    return (prior, dprior, ddprior)
end

function nlogPostPyoccupancy(α::Float64, δ::Float64, mm::Vector{Int64}, icts::Vector{Int64}, param::Float64)
    # check for negatives
    (logp, dlogp, ddlogp) = logliPyOccupancy(α, δ, mm, icts)
    println(logp, dlogp, ddlogp)
    (prior, dprior, ddprior) = pymPrior(param, α, δ)
    println(prior)
    nlogp = -logp - log(prior)
    ndlogp = -dlogp .- dprior ./ prior

    # println(ndlogp)
    nddlogp = -ddlogp .- (prior * ddprior - dprior * dprior) ./ prior^2


    # return nlogp
    return (nlogp, ndlogp, nddlogp)

end

function h_pym(mm::Vector{Int64}, icts::Vector{Int64})
    Hbls = 0.0
    Hvar = 0.0
    if !any(x -> x > 1, icts)
        return Inf64
    end

    # min_alpha = 10 * eps

    N = dot(mm, icts)
    K = sum(mm)

    if K == 1
        Hbls = NaN64
        Hvar = Inf64
    end



    Hbls, Hvar

end
