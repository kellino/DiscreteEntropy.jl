using Random

# ------------------------------Shannon Mutual Information------------------------------ #

# Arrays of real values (probability distribution) 
#m_x: marginal probability (x)
#m_x: marginal probability (y)
#j_xy: joint probability (xy)
#real_mutual_information
function _mutual_information(m_X::Vector, m_Y::Vector, j_XY::Vector)
    hX = _entropy(m_X)
    hY = _entropy(m_Y)
    hXY = _jointentropy(j_XY)
    mi = hX + hY - hXY
    return mi
end

# Arrays of discrete values (samples)
#discrete_mutual_information
function _mutual_information(X::Vector, Y::Vector)
    matrix = freqtable(X, Y)
    j_XY = prop(matrix)
    m_X = marginal_counts(Matrix{Float64}(j_XY), 1)
    m_Y = marginal_counts(Matrix{Float64}(j_XY), 2)

    hX = _entropy(m_X)
    hY = _entropy(m_Y)
    hXY = _entropy(vec(j_XY))
    mi = hX + hY - hXY
    return mi
end


# ----------------------Shannon Mutual Information - Estimation ------------------------ #

Schurmann_param = exp(-1 / 2)
Bayes_param = 0.0
NSB_param = false
PYM_param = nothing

# Mutual Information Estimation
function mi_estimations(X::CountData, Y::CountData, XY::CountData)
    estimations = []
    function _estimate(_X, _Y, _XY, _estimator::Type{T}) where {T<:AbstractEstimator}
        mi = estimate_h(_X, _estimator) + estimate_h(_Y, _estimator) - estimate_h(_XY, _estimator)
        return mi
    end

    function _estimate(_X, _Y, _XY, _estimator::Type{T}) where {T<:NonParameterisedEstimator} 
        mi = estimate_h(_X, _estimator) + estimate_h(_Y, _estimator) - estimate_h(_XY, _estimator)
        return mi
    end

    function _estimate(_X, _Y, _XY, _estimator::Type{T}, arg) where {T<:ParameterisedEstimator}
        mi = estimate_h(_X, _estimator, arg) + estimate_h(_Y, _estimator, arg) - estimate_h(_XY, _estimator, arg)
        return mi
    end

    # Frequentist Estimators
    println("Frequentist mi estimations")
    F_estimators = [MaximumLikelihood, MillerMadow, Grassberger88, Grassberger03, Schurmann, ChaoShen, Zhang, Shrink, Bonachela, ChaoWangJost]
    for F_estimator in F_estimators
        if F_estimator == Schurmann
            push!(estimations, _estimate(X, Y, XY, F_estimator, Schurmann_param))
        else
            push!(estimations, _estimate(X, Y, XY, F_estimator))
        end
        #print_data(F_estimator, last(estimations))
        print_data(last(estimations))
    end 

    # Bayesian Estimators
    println("Bayesian mi estimations")
    B_estimators = [PYM, Bayes, LaPlace, Jeffrey, SchurmannGrassberger, Minimax, NSB, ANSB, PERT]
    for B_estimator in B_estimators
        if B_estimator == Bayes
            push!(estimations, _estimate(X, Y, XY, B_estimator, Bayes_param))
        elseif B_estimator == NSB
            push!(estimations, _estimate(X, Y, XY, B_estimator, NSB_param))
        elseif B_estimator == PYM
            push!(estimations, _estimate(X, Y, XY, B_estimator, PYM_param))
        else
            push!(estimations, _estimate(X, Y, XY, B_estimator))
        end
        #print_data(B_estimator, last(estimations))
        print_data(last(estimations))
    end 

    return estimations
end


function mi_estimations(contingency_matrix::Matrix)
    estimations = []

    function _estimate(_contingency_matrix, _estimator::Type{T}) where {T<:AbstractEstimator}
        hX = estimate_h(from_data(marginal_counts(_contingency_matrix, 1), Histogram), _estimator)
        hY = estimate_h(from_data(marginal_counts(_contingency_matrix, 2), Histogram), _estimator)
        hXY = estimate_h(from_data(_contingency_matrix, Histogram), _estimator)
        mi = hX + hY - hXY
        return mi
    end

    function _estimate(_contingency_matrix, _estimator::Type{T}) where {T<:NonParameterisedEstimator}
        hX = estimate_h(from_data(marginal_counts(_contingency_matrix, 1), Histogram), _estimator)
        hY = estimate_h(from_data(marginal_counts(_contingency_matrix, 2), Histogram), _estimator)
        hXY = estimate_h(from_data(_contingency_matrix, Histogram), _estimator)
        mi = hX + hY - hXY
        return mi
    end
    
    function _estimate(_contingency_matrix, _estimator::Type{T}, arg) where {T<:ParameterisedEstimator}
        hX = estimate_h(from_data(marginal_counts(_contingency_matrix, 1), Histogram), _estimator, arg)
        hY = estimate_h(from_data(marginal_counts(_contingency_matrix, 2), Histogram), _estimator, arg)
        hXY = estimate_h(from_data(_contingency_matrix, Histogram), _estimator, arg)
        mi = hX + hY - hXY
        return mi
    end

    # Frequentist Estimators
    println("Frequentist mi estimations")
    F_estimators = [MaximumLikelihood, MillerMadow, Grassberger88, Grassberger03, Schurmann, ChaoShen, Zhang, Shrink, Bonachela, ChaoWangJost]
    for F_estimator in F_estimators
        if F_estimator == Schurmann
            push!(estimations, _estimate(contingency_matrix, F_estimator, Schurmann_param))
        else
            push!(estimations, _estimate(contingency_matrix, F_estimator))
        end
        #print_data(F_estimator, last(estimations))
        print_data(last(estimations))
    end 

    # Bayesian Estimators
    println("Bayesian mi estimations")
    B_estimators = [PYM, Bayes, LaPlace, Jeffrey, SchurmannGrassberger, Minimax, NSB, ANSB, PERT]
    for B_estimator in B_estimators
        if B_estimator == Bayes
            push!(estimations, _estimate(contingency_matrix, B_estimator, Bayes_param))
        elseif B_estimator == NSB
            push!(estimations, _estimate(contingency_matrix, B_estimator, NSB_param))
        elseif B_estimator == PYM
            push!(estimations, _estimate(contingency_matrix, B_estimator, PYM_param))
        else
            push!(estimations, _estimate(contingency_matrix, B_estimator))
        end
        #print_data(B_estimator, last(estimations))
        print_data(last(estimations))
    end 

    return estimations
end


@doc raw"""
    uncertainty_coefficient(counts::Matrix, estimator::Type{T}; dim=Axis, symmetric=false) where {T<:AbstractEstimator}

Return the estimated uncertainty coefficient, or *Thiel's U* of `counts`, using estimator `estimator` along axis `dim`.

```math
C_X = \frac{I(X;Y)}{H(X)}
```

Return the symmetrical uncertainty coefficient if `symmetric` is `true`.
Do not set a value for `dim` is using `symmetric`: it will be ignored.

```math
2 \left(\frac{H(X) + H(Y) - I(X;Y)}{H(X) + H(Y)}\right)
```

# External Links
[Theil's U](https://en.wikipedia.org/wiki/Uncertainty_coefficient)
"""
function uncertainty_coefficient(contingency_matrix::Matrix, estimator::Type{T}; dim=Axis, symmetric=false) where {T<:AbstractEstimator}
    hx = estimate_h(from_data(marginal_counts(contingency_matrix, 2), Histogram), estimator)
    hy = estimate_h(from_data(marginal_counts(contingency_matrix, 1), Histogram), estimator)
    hxy = estimate_h(from_data(contingency_matrix, Histogram), estimator)

    mi = hx + hy - hxy

    if symmetric
        return 2.0 * (hx + hy - mi) / (hx + hy)
    else
        if dim == X
            return mi / hx
        else
            return mi / hy
        end
    end
end

@doc raw"""
    redundancy(data::CountData, estimator::Type{T}) where {T<:AbstractEstimator}

Return the estimated information redundancy of `data`, with `K` the sampled support size.

```math
R = \log(K) - \hat{H}(data)
```

    redundancy(data::CountData, estimator::Type{T}, K::Int64) where {T<:AbstractEstimator}

Return the estimated information redundancy of `data`, with `K` set by the user.

# External Links
[Redundancy on wikipedia](https://en.wikipedia.org/wiki/Redundancy_(information_theory))

"""
function redundancy(data::CountData, estimator::Type{T}) where {T<:AbstractEstimator}
    log(data.K) - estimate_h(data, estimator)
end

function redundancy(data::CountData, estimator::Type{T}, K::Int64) where {T<:AbstractEstimator}
    log(K) - estimate_h(data, estimator)
end

@doc raw"""
    information_variation(X::CountData, Y::CountData, XY::CountData, H::Function)::Float64

Return the [Variation of Information](https://en.wikipedia.org/wiki/Variation_of_information). This
satisfies the properties of a metric (triangle inequality, non-negativity, indiscernability and symmetry).

```math
VI(X, Y) = 2 * H(X, Y) - H(X) - H(Y)
"""
function information_variation(contingency_matrix::Matrix{T}, estimator::Type{E}) where {T<:Real,E<:AbstractEstimator}
    2.0 *
    # estimate_h(from_data(contingency_matrix), Histogram), estimator) -
    estimate_h(from_data(marginal_counts(contingency_matrix, 1), Histogram), estimator) -
    estimate_h(from_data(marginal_counts(contingency_matrix, 2), Histogram), estimator)
end

function information_variation(X::CountData, Y::CountData, XY::CountData, H::Function)
    return 2.0 * H(XY) - H(X) - H(Y)
end

@doc raw"""
    mutual_information(X::CountData, Y::CountData, XY::CountData, estimator::Function)::Float64
    mutual_information(X::CountData, Y::CountData, XY::CountData, estimator::Type{T}) where {T<:NonParameterisedEstimator}
    mutual_information(X::CountData, Y::CountData, XY::CountData, estimator::Type{T}, args...) where {T<:ParameterisedEstimator}

Compute the mutual information between X and Y, given their joint countdata $XY$, using
the specified estimator. Due to bias in the estimators, this might return a negative number.

```math
I(X;Y) = H(X) + H(Y) - H(X,Y)

```
"""


function mutual_information(X::CountData, Y::CountData, XY::CountData, estimator::Function)
    return estimator(X) + estimator(Y) - estimator(XY)
end

function mutual_information(X::CountData, Y::CountData, XY::CountData, estimator::Type{T}) where {T<:NonParameterisedEstimator}
    estimate_h(X, estimator) + estimate_h(Y, estimator) - estimate_h(XY, estimator)
end

function mutual_information(X::CountData, Y::CountData, XY::CountData, estimator::Type{T}, args...) where {T<:ParameterisedEstimator}
    # The parameterised estimators so far
    # Schurmann, NSB, Bayes, PYM and SchurmmanGeneralised
    # SchurmannGeneralised is the odd one out, as the others just take a scalar
    # args... is unchecked at the moment. The user must ensure the correct types are passed in
    estimate_h(X, args[1], estimator) + estimate_h(Y, args[2], estimator) - estimate_h(XY, args[3], estimator)
end

function mutual_information(X::CountData, Y::CountData, XY::CountData,
    e1::Type{A}, e2::Type{B}, e3::Type{C}) where {A,B,C<:NonParameterisedEstimator}
    estimate_h(X, e1) + estimate_h(Y, e2) - estimate_h(XY, e3)
end

# function mutual_information(counts::Matrix, args...)::Float64
#     X = from_counts(marginal_counts(counts, 1))
#     Y = from_counts(marginal_counts(counts, 2))
#     XY = from_counts([1, 2, 3]) # TODO need to implement this

#     return mutual_information(X, Y, XY, args...)
# end
