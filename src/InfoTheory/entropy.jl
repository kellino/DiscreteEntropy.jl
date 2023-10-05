# Array of prob. distr.
function _entropy(X::AbstractArray{T}) where T<:Real
    H = zero(T)
    z = zero(T)
    for i in 1:length(X)
        if X[i] > z
            H += X[i] * log2(X[i])
        end
    end
    return -H
end