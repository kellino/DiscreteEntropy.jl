# TODO this need thorough testing!
function jk(data::CountData)
    res::Dict{CountData,Int64} = Dict()
    ks = collect(keys(data.histogram))
    vs = collect(values(data.histogram))

    res[data] = 1
    for i in 1:length(ks)
        (d, mm) = reduce(i, ks, vs)
        res[from_dict(d)] = mm
    end

    return res
end

function jackknife(data::CountData, statistic::Function, corrected=false)
    entropies = [(statistic(c), mm) for (c, mm) in jk(data)]
    len = sum([mm for (_, mm) in entropies])

    μ = 1.0 / len * sum(h * mm for (h, mm) in entropies)

    denom = 0
    if corrected
        denom = 1.0 / (len - 1)
    else
        denom = 1.0 / len
    end

    v = denom * sum([(h - μ)^2.0 * mm for (h, mm) in entropies])

    return μ, v


end

function reduce(i, ks, vs)
    d::Dict{Int64,Int64} = Dict()
    mm = 1
    if vs[i] == 1
        n = ks[i] - 1
        ks[i] = n
        for j in 1:length(ks)
            update_or!(d, ks[j], vs[j])
        end
        ks[i] += 1
    elseif vs[i] > 1
        mm = vs[i]
        vs[i] -= 1
        for j in 1:length(ks)
            update_or!(d, ks[j], vs[j])
        end
        vs[i] = 1
    end
    return (d, mm)
end

# TODO add bootstrap resampling
