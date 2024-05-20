using LinearAlgebra: I, dot, diagm

# A Julia port of the original code found on
# Liam Paninski's [homepage](https://www.stat.columbia.edu/~liam/research/code/BUBfunc.m)

function bub(data::CountData; upper_bound=false, k_max=11)
    if data.N < 20.0
        return under(data, upper_bound)
    else
        return over(data, upper_bound, k_max)
    end
end

function under_ub(N, K, a, mesh)
    p, P = get_mesh(N, mesh)
    Pn = [dot(a, x) for x in eachcol(P)]

    maxbias = K * maximum(Pn .+ log.(p .^ p))

    sqrt(maxbias^2 + N * (maximum(abs.(diff(a))))^2) / log(2)

end

function get_mesh(N, mesh)
    fa = loggamma.(1:2*N+1)
    Ni = fa[N+1] .- fa[1:N+1] - reverse(fa[1:N+1])
    q = (0:N*mesh) ./ (N * mesh)
    p = q[2:end-1]
    lp = log.(p)
    lq = reverse(lp)

    P = zeros(N + 1, length(p) + 2)
    for i in 0:N
        P[i+1, 2:end-1] = Ni[i+1] .+ (i .* lp .+ (N - i) .* lq)
    end
    P = exp.(P)

    P[2:end-1, [1 end]] = zeros(N - 1, 2)
    P[end, end] = 1.0
    P[end, 1] = 0.0
    P[1, end] = 0.0
    P[1, 1] = 1.0

    return (q, P)

end

function under(data::CountData, upper_bound)
    N = convert(Integer, data.N)
    p, P = get_mesh(N, 5)

    X = data.K * P'
    XX = X' * X
    XY = data.K * X' * (-log.(p .^ p))

    DD = 2 .* Matrix{Real}(I, N + 1, N + 1)
    for i in 1:N
        DD[i, i+1] = -1.0
        DD[i+1, i] = -1.0
    end
    DD[1] = 1.0
    DD[N+1, N+1] = 1.0

    AA = XX + N * DD
    a = inv(AA) * XY

    h = sum(a[convert(Integer, x[1])+1] * x[2] for x in eachcol(data.multiplicities))

    if upper_bound
        m = under_ub(N, data.K, a, 10)
        return (h, m)
    end

    h
end

function over(data::CountData, upper_bound, k_max; lambda_0=1)
    # N = convert(Integer, data.N)

    if k_max > data.N
        k_max = data.N - 1
    end

    # c = 80
    #c = ceil(min(N, c * maximum(N / data.K, 1)))
    c = Integer(ceil(min(data.N, 80 * maximum([(data.N / data.K), 1]))))
    s = 30
    mesh = 200
    eps = (data.N^-1) * 10^-10
    Ni = loggamma(data.N + 1) .- loggamma.(1:c+1) .- loggamma.(data.N + 1 .- (0:c))

    p = logspace(log10(1e-4 / data.N), log10(min(1, s / data.N) - eps), mesh)
    lp = log.(p)
    lq = log.(1 .- p)

    P = exp.(repeat(Ni, 1, length(p)) .+ (i for i in 0:c) .* lp' .+ (data.N - i for i in 0:c) .* lq')

    epsm = (data.K^-1) * 10^-10
    pm = epsm : min(1, s/data.K) / mesh : min(1, s/data.K) - epsm
    lpm = log.(pm)
    lqm = log.(1 .- pm)

    # Pm = exp.(Ni .+ (i for i in 0:c) .* lpm' .+ (data.N - i for i in 0:c)) .* lq'
    Pm = exp.(repeat(Ni, 1, length(pm)) .+ (i for i in 0:c) .* lpm' .+ (data.N - i for i in 0:c) .* lqm')

    f = [x <= 1 / data.K ? data.K : x^-1 for x in pm]

    a =  [i / data.N for i in 0:data.N]
    a = [-log(x^x) + (1 - x) * (0.5 / data.N) for x in a]

    mda = maximum(abs.(diff(a)))

    best_MM = Inf64
    best_a = nothing
    best_B = nothing
    best_V1 = nothing


    w = size(P)[1]
    for k in 1:min(k_max,Integer(data.N))
        h_mm = a[k+1:c+1]' * selectdim(P, 1, k+1:w)
        XX = data.K^2 * selectdim(P, 1, 1:k) * selectdim(P, 1, 1:k)'
        XY = data.K^2 * selectdim(P, 1, 1:k) * ((-log.(p.^p)) .- h_mm')
        XY[k] += data.N * a[k]
        DD = 2 * I(k) - diagm(1 => ones(k-1)) - diagm(-1 => ones(k-1))
        DD[1] = 1
        DD[k, k] = 1
        AA = XX .+ data.N .* DD
        AA[1] += lambda_0
        AA[k, k] += data.N
        a[1:k] = pinv(AA) * XY
        B = data.K .* (a[1:c+1]' * P .+ log.(p.^p)')
        maxbias = maximum(abs.(B))
        V1 = ((0:c)./data.N) .* (a[1:c+1] .- [0; a[1:c]]).^2
        V1 = V1' * Pm
        l = minimum((k+2, length(a)))
        mmda = max(mda , maximum(abs.(diff(a[1 : l])) ))
        MM = sqrt(maxbias^2 + data.N * minimum((mmda^2, 4 * maximum(f .* V1')))) / log(2)


        if MM < best_MM
            best_MM = MM
            best_a = a
            best_B = B
            best_V1 = V1
        end
    end


    return (best_a, best_MM)
end
