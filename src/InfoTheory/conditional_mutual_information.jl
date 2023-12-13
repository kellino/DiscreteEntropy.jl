using Random

# -----------------------Shannon Conditional Mutual Information ------------------------ #

# Arrays of real values (probability distribution) 
function _conditional_mutual_information(j_XZ::Vector, j_YZ::Vector, j_XYZ::Vector, m_Z::Vector)
    hXZ = _entropy(j_XZ)
    hYZ = _entropy(j_YZ)
    hXYZ = _entropy(j_XYZ)
    hZ = _entropy(m_Z)
    cmi = hXZ + hYZ - hXYZ - hZ
    return cmi
end

# ----------------Shannon Conditional Mutual Information - Estimation ------------------ #

Schurmann_param = exp(-1 / 2)
Bayes_param = 0.0
NSB_param = false
PYM_param = nothing

function cmi_estimation(XZ::CountData, YZ::CountData, XYZ::CountData, Z::CountData)
    estimations = []

    function _estimate(_XZ, _YZ, _XYZ, _Z, _estimator::Type{T}) where {T<:AbstractEstimator}
        cmi = estimate_h(_XZ, _estimator) + estimate_h(_YZ, _estimator) - estimate_h(_XYZ, _estimator) - estimate_h(_Z, _estimator)
        return cmi
    end

    function _estimate(_XZ, _YZ, _XYZ, _Z, _estimator::Type{T}) where {T<:NonParameterisedEstimator} 
        cmi = estimate_h(_XZ, _estimator) + estimate_h(_YZ, _estimator) - estimate_h(_XYZ, _estimator) - estimate_h(_Z, _estimator)
        return cmi
    end

    function _estimate(_XZ, _YZ, _XYZ, _Z, _estimator::Type{T}, arg) where {T<:ParameterisedEstimator}
        cmi = estimate_h(_XZ, _estimator, arg) + estimate_h(_YZ, _estimator, arg) - estimate_h(_XYZ, _estimator, arg) - estimate_h(_Z, _estimator, arg)
        return cmi
    end


    println("Frequentist cmi estimations")
    F_estimators = [MaximumLikelihood, MillerMadow, Grassberger88, Grassberger03, Schurmann, ChaoShen, Zhang, Shrink, Bonachela, ChaoWangJost]
    for F_estimator in F_estimators
        if F_estimator == Schurmann
            push!(estimations, _estimate(XZ, YZ, XYZ, Z, F_estimator, Schurmann_param))
        else
            push!(estimations, _estimate(XZ, YZ, XYZ, Z, F_estimator))
        end
        #print_data(F_estimator, last(estimations))
        print_data(last(estimations))
    end 

    println("Bayesian cmi estimations")
    B_estimators = [PYM, Bayes, LaPlace, Jeffrey, SchurmannGrassberger, Minimax, NSB, ANSB, PERT]
    for B_estimator in B_estimators
        if B_estimator == Bayes
            push!(estimations, _estimate(XZ, YZ, XYZ, Z, B_estimator, Bayes_param))
        elseif B_estimator == NSB
            push!(estimations, _estimate(XZ, YZ, XYZ, Z, B_estimator, NSB_param))
        elseif B_estimator == PYM
            push!(estimations, _estimate(XZ, YZ, XYZ, Z, B_estimator, PYM_param))
        else
            push!(estimations, _estimate(XZ, YZ, XYZ, Z, B_estimator))
        end
        #print_data(B_estimator, last(estimations))
        print_data(last(estimations))
    end 

    return estimations
end

