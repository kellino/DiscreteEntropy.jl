@doc raw"""
     mutual_information(XY::CountData, X::CountData, Y::CountData)
"""
function mutual_information(X::CountData, Y::CountData, XY::CountData, estimator::Type{T}) where {T<:AbstractEstimator}
    mi = estimate_h(X, estimator) + estimate_h(Y, estimator) - estimate_h(XY, estimator)
    if mi < 0.0
        return 0.0
    end
    mi
end

function mutual_information(joint::Matrix{Int64}, estimator::Type{T}) where {T<:AbstractEstimator}
    X = from_counts(marginal_counts(joint, 1))
    Y = from_counts(marginal_counts(joint, 2))
    XY = from_counts(vec(joint))

    mutual_information(X, Y, XY, estimator)
end
