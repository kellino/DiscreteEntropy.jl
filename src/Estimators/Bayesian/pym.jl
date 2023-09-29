using SpecialFunctions;
using LinearAlgebra;
using Optim;

@enum PriorType begin
    default
    identity
end

const ComplexOrReal{T} = Union{T,Complex{T}}
global m
global icts

ψ₂(x) = polygamma(2, x)

#mm = [1, 2, 3]
#icts = [2, 2, 3]

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
    iSz = size(α)
    dha = trigamma(α + 1.0)
    dhd = trigamma(1.0 - δ)
    dga = -(gam / eta) * dha
    dgd = dhd .* (digamma(α + 1.0) .- digamma.(1.0)) / (eta^2.0)
    J = dha * dgd - dhd * dga

    return (J, gam, eta, iSz, dga, dgd, dha, dhd)
end

function computePrior(α::Float64, δ::Float64, param::Float64, type::PriorType, op::Integer)
    # only handling default case at the moment
    (_, gg_t, hh_t, iSz, dga, dgd, dha, dhd) = common_prior_jacobian(α, δ)
    
    if type == default

        f = exp(-1 ./ (1 - gg_t) / param)
    
        if op == 1

            return f
        end 
    

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

pymPrior(a::Number, d::Number, p::Number, t::PriorType, op::Integer) = _pymPrior(float(a), float(d), float(p), t, op)

function _pymPrior(α::ComplexOrReal{Float64}, δ::ComplexOrReal{Float64}, param::Float64, type::PriorType, n_opar::Integer)
    a = real(α)
    d = real(δ)
    p = real(param)
    t = type
    op = Integer(n_opar)

    return computePrior(a, d, p, t, op)
end

logliPyOccupancy(a::Number, d::Number, op::Integer) = _logliPyOccupancy(float(a), float(d), op)

function _logliPyOccupancy(α::ComplexOrReal{Float64}, δ::ComplexOrReal{Float64}, n_opar::Integer)
    # via the derivative of the Kölbig digamma formulation
    a = real(α)
    d = real(δ)
    op = Integer(n_opar)

    N = dot(mm, icts)
    K = sum(mm)

    logp = -gammalndiff(a + 1.0, N - 1.0) + sum(map(x -> logx(a + x * d), 1:K-1)) + dot(mm, map(x -> gammalndiff(1.0 - d, Float64(x)), icts .- 1))
    
    if op == 1
        return logp
    else 
        KK = 1:K - 1
        dlogp = zeros(1, 2)
        Z = 1.0 ./ (a .+ KK .* d)
        dlogp[:1] = sum(Z) - digamma(a + N) + digamma(a + 1.0)
        dlogp[:2] = dot(KK, Z) - dot(mm, digamma.(icts .- d) .- digamma.(1.0 - d))

        return logp, dlogp
    end
end

function hessian(α::Float64, δ::Float64)
    N = dot(mm, icts)
    K = sum(mm)
    KK = 1:K-1
    Z = 1.0 ./ (α .+ KK .* δ)

    prior = pymPrior(α, δ, 0.1, default, 3)

    ddlogp = zeros(2, 2)
    Z2 = Z .^ 2.0
    ddlogp[1, 1] = trigamma(1.0 + α) - trigamma(α + N) - sum(Z2)
    diag = -sum(KK .* Z2)
    ddlogp[1, :2] = diag
    ddlogp[2, :1] = diag
    ddlogp[2, :2] = -sum((KK .^ 2.0) .* Z2) + dot(mm, trigamma.(icts .- δ) .- trigamma.(1.0 - δ))

    return -ddlogp - (prior.prior * prior.ddprior .- prior.dprior' .* prior.dprior) ./ prior.prior^2
end

function nlogPostPyoccupancy(α::Float64, δ::Float64)

    if α < 0.0 || δ < 0.0 || δ > 1.0
        return (Inf, -[α, δ])
    end

    (logp, dlogp) = logliPyOccupancy(α, δ, 2)

    prior = pymPrior(α, δ, 0.1, default, 2)

    nlogp = -logp - logx(prior.prior)

    ndlogp = -dlogp - prior.dprior' ./ prior.prior

    return (nlogp, ndlogp)
end

@doc raw"""
    pym(mm::Vector{Int64}, icts::Vector{Int64})::Float64

A more or less faithful port of the original [matlab code](https://github.com/pillowlab/PYMentropy)
to Julia

"""

function meshgrid(x, y)
    X = [i for i in x, j in 1:length(y)]
    X = transpose(X)
    Y = [j for i in 1:length(x), j in y]
    Y = transpose(Y)

    return X, Y
end

function condH(a, d)
    m = computeHpy(mm, icts, a[:], d[:])
    m = reshape(m, size(a))

    return m
end

function computeHpy(mm, icts, alphas, ds)
    # [Hpy,Hvar] = computeHpy(mm,icts,alphas,ds)

    # Compute posterior mean of P(H|n,alpha,d) under a Pitman-Yor process prior
    # with parameter alpha and d.

    # Accepts same size "alphas" and "ds" and returns posterior mean
    # at each alpha and d with the same size.

    # Posterior process is Dir(n_1-d, ..., n_K-d, alpha+d*K)
    # So the first size biased sample is distributed as a mixture of Beta's:

    # sum_{k=1}^K (n_k-d)/(n+alpha) Beta(1+n_k-d, n+alpha-n_k+d*(K-1))
    # + (alpha+d*K)/(n+alpha) Beta(1-d, alpha+n+d*K) % 2->1 +n

    # Hence the entropy is,
    # sum_{k=1}^K (n_k-d)/(n+alpha)
    # (digamma(1+n+alpha+d*(K-2)) - digamma(1+n_k-d))
    # + (alpha+d*K)/(n+alpha)
    # (digamma(1+alpha+n+d*(K-1)) - digamma(1-d))

    # Inputs:
    #   mm = multiplicities (mm(j) is # bins with icts(j) elements)
    #   icts = vector of unique counts
    #   K = # total bins in distribution
    #   alphas = scalar (or vector) of concentration parameter
    #   ds = scalar (or vector) of discount parameter

    # Ouptuts:
    #   Hpy = mean entropy at each alpha
    #   Hvar = variance of entropy at each alpha

    if isempty(mm) || isempty(icts)
        icts = 0
        mm = 0
    end

    #if nargin < 4: We haven't passed any d's - we want Dirichlet Process posterior
    #ds = zeros(size(alphas));

    # make alphas,icts,mm column vectors
    originalSize = size(alphas)
    alphas = alphas[:]
    ds = ds[:]
    icts = icts[:]
    mm = mm[:]

    N = icts' * mm # number of samples
    K = sum(mm[icts.>0]) # number of tables

    Hp = zeros(size(ds))

    Hpi = computeHpyPrior(alphas + K * ds, ds)

    if N == 0
        # if we have no data, return prior mean & variance
        Hpy = Hpi

        return
    end

    # compute E[p_*] and E[(1-p_*)]
    oneminuspstarmean = (N .- K .* ds) ./ (alphas .+ N)
    pstarmean = (alphas .+ K .* ds) ./ (alphas .+ N)
    # compute E[(p_*).^2] and E[(1-p_*)^2]
    # compute E[h(p_*)]
    Hpstar = digamma.(alphas .+ N .+ 1) -
             (alphas .+ K .* ds) ./ (alphas .+ N) .* digamma.(alphas .+ K .* ds .+ 1) -
             (N .- K .* ds) ./ (alphas .+ N) .* digamma.(N .- K .* ds .+ 1)

    for k = 1:length(ds)
        Hp[k] = digamma(N .- K .* ds[k] .+ 1.0) - mm' * ((icts .- ds[k]) ./ (N .- K .* ds[k]) .* digamma.(1.0 .+ icts .- ds[k]))
    end

    Hpy = oneminuspstarmean .* Hp + pstarmean .* Hpi + Hpstar
    Hpy = reshape(Hpy, originalSize)

    return Hpy
end

function computeHdir(mm, icts, K, alphas)
    # [Hdir,Hvar] = computeHdir(mm,icts,K,alphas);

    # Compute posterior mean of P(H|n,alpha), the expected entropy under a
    # fixed Dirichlet prior with Dirichlet parameter alpha

    # Accepts a vector "alphas" and returns posterior mean at each alpha

    # Inputs:
    #   mm = multiplicities (mm(j) is # bins with icts(j) elements)
    #   icts = vector of unique counts
    #   K = # total bins in distribution
    #   alphas = scalar (or vector) of Dirichlet parameters

    # Ouptuts:
    #   Hdir = mean entropy at each alpha
    #   Hvar = variance of entropy at each alpha

    if isempty(mm) || isempty(icts)
        icts = 0
        mm = K
    end

    # modify mm and icts to account for #zeros (finite bin case)
    nZ = K - sum(mm)

    # Make alphas,icts,mm column vectors
    if size(alphas, 1) == 1
        alphas = alphas'  # column vec
    end
    if size(icts, 1) == 1
        icts = icts'
    end
    if size(mm, 1) == 1
        mm = mm'
    end

    N = icts' * mm # number of samples
    A = N .+ K .* alphas # number of effective samples (vector)
    aa = alphas .+ icts' # posterior Dirichlet priors

    # Compute posterior mean over entropy
    Hdir = digamma.(A .+ 1) - (1.0 ./ A) .* ((aa .* digamma.(aa .+ 1)) * mm)

    return Hdir
end

function computeHpyPrior(alphas, ds)
    # Compute prior mean of P(H|alpha,d) under a Pitman-Yor process prior
    # with parameter alpha and d.
    # Accepts same size "alphas" and "ds" and returns posterior mean
    # at each alpha and d with the same size.

    # Ouptuts:
    #   Hpy = mean entropy at each alpha
    #   Hvar = variance of entropy at each alpha

    Hpy = digamma.(1.0 .+ alphas) .- digamma.(1.0 .- ds)

    return Hpy
end

function gq100(a, b, ngrid)
    N = 25

    (_, _, y, Lp) = lgwt(N, -1, 1)

    N_persistent = N
    y_persistent = y
    Lp_persistent = Lp

    N = N_persistent
    y = y_persistent
    Lp = Lp_persistent

    N1 = N
    N2 = N + 1
    x = (a .* (1.0 .- y) + b .* (1.0 .+ y)) ./ 2
    w = (b - a) ./ ((1.0 .- y .^ 2) .* Lp[:, 1] .^ 2) .* (N2 / N1)^2

    return (x, w, N)
end

function lgwt(N, a, b)

    eps = 2.2204e-16
    N = N - 1
    N1 = N + 1
    N2 = N + 2

    xu = range(-1, 1, N1)'

    #println(vcat(xu))

    # Initial guess
    y = cos.((2 * (0:N)' .+ 1) * pi / (2 * N + 2)) + (0.27 / N1) * sin.((pi * xu) * N / N2)

    # Legendre-Gauss Vandermonde Matrix
    L = zeros(N1, N2)

    # Derivative of LGVM
    Lp = zeros(N1, N2)

    # Compute the zeros of the N+1 Legendre Polynomial
    # using the recursion relation and the Newton-Raphson method
    y0 = 2

    # Iterate until new points are uniformly within epsilon of old points

    while maximum(vec(abs.(y .- y0))) > eps

        L[:, 1] .= 1
        Lp[:, 1] .= 0

        L[:, 2] .= vec(y)
        Lp[:, 2] .= 1

        for k = 2:N1
            L[:, k+1] .= ((2 * k - 1) * vec(y) .* L[:, k] - (k - 1) * L[:, k-1]) / k
        end

        Lp[:, 1] = (N2) * (L[:, N1] - vec(y) .* L[:, N2]) ./ (1.0 .- vec(y) .^ 2)

        y0 = y
        y = vec(y0) - L[:, N2] ./ Lp[:, 1]

    end

    # Linear map from[-1,1] to [a,b]
    x = (a .* (1.0 .- vec(y)) + b .* (1.0 .+ vec(y))) ./ 2

    # Compute the weights
    w = (b .- a) ./ ((1.0 .- vec(y) .^ 2) .* Lp[:, 1] .^ 2) .* (N2 / N1)^2

    return (x, w, y, Lp[:, 1])
end

#------------------------------------------------------------#

#function pym(mm::Vector{Int64}, icts::Vector{Int64})::Float64
function pym(_mm::Vector{Int64}, _icts::Vector{Int64})
    Hbls = 0.0

    global mm = _mm
    global icts = _icts
 
    if !any(x -> x > 1, icts)
        return Inf64
    end

    eps = 2.2204e-16
    min_alpha = 10 * eps

    N = dot(mm, icts)
    K = sum(mm)

    if K == 1
        Hbls = NaN64
    end

    mpt = [1.0, 0.01]
    nlpy(x) = nlogPostPyoccupancy(x[1], x[2])[1]
    res = optimize(nlpy, mpt)

    #new mpt
    params = Optim.minimizer(res)
    fval = Optim.minimum(res)

    hess = hessian(params[1], params[2])

    fval = -fval

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

    Na = []
    Nd = []

    (ax, aw, Na) = gq100(al, au, Na)
    (dx, dw, Nd) = gq100(dl, du, Nd)

    if Nd * Na < 1e4
        (aa, dd) = meshgrid(ax, dx)
        loglik = logliPyOccupancy.(aa[:], dd[:], 1)
        lik = exp.(loglik .- maximum(loglik))
        prior = pymPrior.(aa[:], dd[:], 0.1, default, 1)
        mc = condH(aa[:], dd[:])
        A = ((lik .* prior) .* vec(dw * aw'))
        Z = sum(A)
        Hbls = sum(A .* mc)
        Hbls = Hbls / Z
    end

    #return (al, au, dl, du)
    return Hbls
end
