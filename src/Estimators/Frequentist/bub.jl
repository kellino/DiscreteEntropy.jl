function BjN(x, j, N)
    return binomial(N, j) * x^j * (1.0 - x)^(N - j)
end

function ajN(j, N)
    j_over_n = j / N
    return -j_over_n * logx(j_over_n) + ((1 - j_over_n) / 2N)
end

function H(x)
    return -xlogx(x)
end

function c(restrict, N, m)
    return ceil(Int64, minimum((N, restrict * maximum((N / m, 1)))))
end

function f(x, N)
    if x < 1 / N
        return N
    else
        return 1 / x
    end
end

function poly(x, lim, N)
    return H(x) - sum([ajN(j, N) * BjN(x, j, N) for j in 1:lim])
end


function bub(counts::AbstractVector)
    N = sum(counts)
    a = (0:N) / N
    a = -log.(a .^ a) .+ (1 .- a) * 0.5 / N
    return a

end
