using LinearAlgebra: I, dot

# A Julia port of the original code found on
# Liam Paninski's [homepage](ht, upperboundtps://www.stat.columbia.edu/~liam/research/code/BUBfunc.m)

function bub(data::CountData; upper_bound=false)
    if data.N < 20.0
        return under(data, upper_bound)
    else
        return over(data, upper_bound)
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

    # f = [x <= 1 / data.K ? 5 : x^-1 for x in q] # why is this not used?
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

function over(data::CountData, upper_bound)
end
