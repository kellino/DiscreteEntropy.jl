using SpecialFunctions;
using LinearAlgebra;
using Optim;

@enum PriorType begin
    default
    identity
end

ψ₂(x) = polygamma(2, x)

mm = [1, 2, 3]
icts = [2, 2, 3]

struct Prior
    type::PriorType
    param::Float64
    prior::Float64
    d::Float64
    dprior
    ddprior::Array{Float64,2}
end

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
    return (J, gam, eta, dga, dgd, dha, dhd)
end

function computePrior(α::Float64, δ::Float64, param::Float64, type::PriorType)
    # only handling default case at the moment
    (_, gg_t, hh_t, dga, dgd, dha, dhd) = common_prior_jacobian(α, δ)

    if type == default
        f = exp(-1 ./ (1 - gg_t) / param)
        dg = -f / (1 - gg_t)^2 / param

        dp::Vector{Float64} = []
        push!(dp, dg * dga)
        push!(dp, dg * dgd)

        ddp = zeros(2, 2)

        ddg = f / (1 - gg_t)^4 / param^2 * (1 + 2 * (gg_t - 1) * param)
        ddp[1, 1] = ddg * dga^2 + dg * (2 * gg_t / hh_t^2 * dha^2 - gg_t / hh_t * ψ₂(1 + α))
        ddp[1, 2] = ddg * dga * dgd - dg * (dhd * dha / hh_t^2 - (2 * gg_t * dha * dhd / hh_t^2))
        ddp[2, 1] = ddp[1, 2]
        ddp[2, 2] = ddg * dgd^2 + dg * (ψ₂(1.0 - δ) * (digamma(1) - digamma(α + 1.0)) / (hh_t^2) +
                                        2 * trigamma(1.0 - δ)^2 * (digamma(1) - digamma(1.0 + α)) / hh_t^3)

        return Prior(type, param, f, dg, dp, ddp)
    else
        println("not yet done")
    end

end

function pymPrior(α::Float64, δ::Float64; param::Float64=0.1, type::PriorType=default)::Prior
    return computePrior(α, δ, param, type)
end

function logliPyOccupancy(α::Float64, δ::Float64, mm::Vector{Int64}, icts::Vector{Int64})
    N = dot(mm, icts)
    K = sum(mm)
    logp = -gammalndiff(α + 1.0, N - 1.0) +
           sum(map(x -> logx(α + x * δ), 1:K-1)) +
           dot(mm, map(x -> gammalndiff(1.0 - δ, Float64(x)), icts .- 1))

    KK = 1:K-1
    dlogp = zeros(1, 2)
    Z = 1.0 ./ (α .+ KK .* δ)
    dlogp[:1] = sum(Z) - digamma(α + N) + digamma(α + 1.0)
    dlogp[:2] = dot(KK, Z) - dot(mm, digamma.(icts .- δ) .- digamma.(1.0 - δ))

    # ddlogp = zeros(2, 2)
    # Z2 = Z .^ 2.0
    # ddlogp[1, 1] = trigamma(1.0 + α) - trigamma(α + N) - sum(Z2)
    # diag = -sum(KK .* Z2)
    # ddlogp[1, :2] = diag
    # ddlogp[2, :1] = diag
    # ddlogp[2, :2] = -sum((KK .^ 2.0) .* Z2) + dot(mm, trigamma.(icts .- δ) .- trigamma.(1.0 - δ))

    return (logp, dlogp)

end

function hessian(α::Float64, δ::Float64, mm::Vector{Int64}, icts::Vector{Int64})
    N = dot(mm, icts)
    K = sum(mm)
    KK = 1:K-1
    Z = 1.0 ./ (α .+ KK .* δ)

    prior = pymPrior(α, δ)

    ddlogp = zeros(2, 2)
    Z2 = Z .^ 2.0
    ddlogp[1, 1] = trigamma(1.0 + α) - trigamma(α + N) - sum(Z2)
    diag = -sum(KK .* Z2)
    ddlogp[1, :2] = diag
    ddlogp[2, :1] = diag
    ddlogp[2, :2] = -sum((KK .^ 2.0) .* Z2) + dot(mm, trigamma.(icts .- δ) .- trigamma.(1.0 - δ))

    return -ddlogp - (prior.prior * prior.ddprior .- prior.dprior' .* prior.dprior) ./ prior.prior^2
    # return ddlogp
end

function nlogPostPyoccupancy(α::Float64, δ::Float64, mm::Vector{Int64}, icts::Vector{Int64})

    if α < 0.0 || δ < 0.0 || δ > 1.0
        return (Inf, -[α, δ])
    end


    (logp, dlogp) = logliPyOccupancy(α, δ, mm, icts)
    prior = pymPrior(α, δ)

    nlogp = -logp - logx(prior.prior)

    ndlogp = -dlogp - prior.dprior' ./ prior.prior

    # nddlogp = -ddlogp - (prior.prior * prior.ddprior .- prior.dprior' .* prior.dprior) ./ prior.prior^2

    # return nlogp
    return (nlogp, ndlogp)
end

@doc raw"""
    pym(mm::Vector{Int64}, icts::Vector{Int64})::Float64

A more or less faithful port of the original [matlab code](https://github.com/pillowlab/PYMentropy)
to Julia

"""
function pym(mm::Vector{Int64}, icts::Vector{Int64})
    Hbls = 0.0
    Hvar = 0.0
    if !any(x -> x > 1, icts)
        return Inf64
    end

    eps = 2.2204e-16
    min_alpha = 10 * eps

    N = dot(mm, icts)
    K = sum(mm)

    if K == 1
        Hbls = NaN64
        Hvar = Inf64
    end

    mpt = [1.0, 0.01]
    nlpy(x) = nlogPostPyoccupancy(x[1], x[2], mm, icts)[1]
    res = optimize(nlpy, mpt)

    params = Optim.minimizer(res)

    hess = hessian(params[1], params[2], mm, icts)

    if params[1] < min_alpha
        @warn("MAP α is very small, this might be unstable")
    end


    p::Bool = false
    if any(i -> isnan(i) || isinf(i), hess)
        @warn("Hessian contains nan or inf")
        p = true
    else
        p = isposdef(hess)
    end

    if !p || rank(hess) < 2
        @warn("Hessian is not positive definite, computing integral on full semi-infinite interval")
        dl = eps
        du = 1 - eps
    else
        invHessian = 6.0 * sqrt(inv(hess))
        al = max(params[1] - invHessian[1, 1], eps)
        au = params[1] + invHessian[1, 1]
        dl = max(params[2] - invHessian[2, 2], eps)
        du = min(1 - eps, max(params[2] + invHessian[2, 2], eps))
    end


    return (al, au, dl, du)

end
