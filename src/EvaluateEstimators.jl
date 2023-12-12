include("DiscreteEntropy.jl")
using .DiscreteEntropy
using Random


function h_estimation(data)
    estimations = []

    function _estimate(_data, _estimator::Type{T}) where {T<:AbstractEstimator}
        h = estimate_h(_data, _estimator)
        #println(string(_estimator) * " " * string(round(h; digits=4)))
        println(string(round(h; digits=4)))
        return h
    end

    function _estimate(_data, _estimator::Type{T}) where {T<:NonParameterisedEstimator} 
        h = estimate_h(_data, _estimator)
        #println(string(_estimator) * " " * string(round(h; digits=4)))
        println(string(round(h; digits=4)))
        return h
    end

    function _estimate(_data, _estimator::Type{T}, param) where {T<:ParameterisedEstimator}
        h = estimate_h(_data, _estimator, param)
        #println(string(_estimator) * " " * string(round(h; digits=4)))
        println(string(round(h; digits=4)))
        return h
    end


    println("Frequentist h estimators")
    F_estimators = [MaximumLikelihood, MillerMadow, Grassberger88, Grassberger03, Schurmann, ChaoShen, Zhang, Shrink, Bonachela, ChaoWangJost]
    for F_estimator in F_estimators
        if F_estimator == Schurmann
            push!(estimations, _estimate(data, F_estimator, exp(-1 / 2)))
        else
            push!(estimations, _estimate(data, F_estimator))
        end
    end 

    println("Bayesian h estimators")
    B_estimators = [PYM, Bayes, LaPlace, Jeffrey, SchurmannGrassberger, Minimax, NSB, ANSB, PERT]
    for B_estimator in B_estimators
        if B_estimator == Bayes
            push!(estimations, _estimate(data, B_estimator, 0.0))
        elseif B_estimator == NSB
            push!(estimations, _estimate(data, B_estimator, false))
        elseif B_estimator == PYM
            push!(estimations, _estimate(data, B_estimator, nothing))
        else
            push!(estimations, _estimate(data, B_estimator))
        end
    end 

    return estimations
end


function mi_estimation(x::CountData, y::CountData, xy::CountData)
    estimations = []

    function _estimate(_x, _y, _xy, _estimator::Type{T}) where {T<:AbstractEstimator}
        mi = estimate_h(_x, _estimator) + estimate_h(_y, _estimator) - estimate_h(_xy, _estimator)
        #println(string(_estimator) * " " * string(round(mi; digits=4)))
        println(string(round(mi; digits=4)))
        return mi
    end

    function _estimate(_x, _y, _xy, _estimator::Type{T}) where {T<:NonParameterisedEstimator} 
        mi = estimate_h(_x, _estimator) + estimate_h(_y, _estimator) - estimate_h(_xy, _estimator)
        #println(string(_estimator) * " " * string(round(mi; digits=4)))
        println(string(round(mi; digits=4)))
        return mi
    end

    function _estimate(_x, _y, _xy, _estimator::Type{T}, param) where {T<:ParameterisedEstimator}
        mi = estimate_h(_x, _estimator, param) + estimate_h(_y, _estimator, param) - estimate_h(_xy, _estimator, param)
        #println(string(_estimator) * " " * string(round(mi; digits=4)))
        println(string(round(mi; digits=4)))
        return mi
    end


    println("Frequentist mi estimators")
    F_estimators = [MaximumLikelihood, MillerMadow, Grassberger88, Grassberger03, Schurmann, ChaoShen, Zhang, Shrink, Bonachela, ChaoWangJost]
    for F_estimator in F_estimators
        if F_estimator == Schurmann
            push!(estimations, _estimate(x, y, xy, F_estimator, exp(-1 / 2)))
        else
            push!(estimations, _estimate(x, y, xy, F_estimator))
        end
    end 

    println("Bayesian mi estimators")
    B_estimators = [PYM, Bayes, LaPlace, Jeffrey, SchurmannGrassberger, Minimax, NSB, ANSB, PERT]
    for B_estimator in B_estimators
        if B_estimator == Bayes
            push!(estimations, _estimate(x, y, xy, B_estimator, 0.0))
        elseif B_estimator == NSB
            push!(estimations, _estimate(x, y, xy, B_estimator, false))
        elseif B_estimator == PYM
            push!(estimations, _estimate(x, y, xy, B_estimator, nothing))
        else
            push!(estimations, _estimate(x, y, xy, B_estimator))
        end
    end 

    return estimations
end

