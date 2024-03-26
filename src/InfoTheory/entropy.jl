using Random


# -----------------------------------Shannon Entropy------------------------------------ #

# Array of probability distribution
function _entropy(X::AbstractArray{T}) where T<:Real
    h = zero(T)
    z = zero(T)
    for i in 1:length(X)
        if X[i] > z
            h += X[i] * log2(X[i])
        end
    end
    return -h
end

# Array of probability distribution
function _jointentropy(XY::AbstractArray{T}) where T<:Real
    jh = zero(T)
    z = zero(T)
    for i in 1:length(XY)
        if XY[i] > z
            jh += XY[i] * log2(XY[i])
        end
    end
    return -jh
end

# -----------------------------Shannon Entropy - Estimation ---------------------------- #

Schurmann_param = exp(-1 / 2)
Bayes_param = 0.0
NSB_param = false
PYM_param = nothing

# Entropy Estimation
function h_estimations(data)
    estimations = []
    function _estimate(_data, _estimator::Type{T}) where {T<:AbstractEstimator}
        h = estimate_h(_data, _estimator)
        return h
    end

    function _estimate(_data, _estimator::Type{T}) where {T<:NonParameterisedEstimator} 
        h = estimate_h(_data, _estimator)
        return h
    end

    function _estimate(_data, _estimator::Type{T}, param) where {T<:ParameterisedEstimator}
        h = estimate_h(_data, _estimator, param)
        return h
    end

    # Frequentist Estimators
    println("Frequentist h estimations")
    F_estimators = [MaximumLikelihood, MillerMadow, Grassberger88, Grassberger03, Schurmann, ChaoShen, Zhang, Shrink, Bonachela, ChaoWangJost]
    for F_estimator in F_estimators
        if F_estimator == Schurmann
            push!(estimations, _estimate(data, F_estimator, Schurmann_param))
        else
            push!(estimations, _estimate(data, F_estimator))
        end
        #print_data(F_estimator, last(estimations))
        print_data(last(estimations))
    end 

    # Bayesian Estimators
    println("Bayesian h estimations")
    B_estimators = [PYM, Bayes, LaPlace, Jeffrey, SchurmannGrassberger, Minimax, NSB, ANSB, PERT]
    for B_estimator in B_estimators
        if B_estimator == Bayes
            push!(estimations, _estimate(data, B_estimator, Bayes_param))
        elseif B_estimator == NSB
            push!(estimations, _estimate(data, B_estimator, NSB_param))
        elseif B_estimator == PYM
            push!(estimations, _estimate(data, B_estimator, PYM_param))
        else
            push!(estimations, _estimate(data, B_estimator))
        end
        #print_data(B_estimator, last(estimations))
        print_data(last(estimations))
    end 

    return estimations
end