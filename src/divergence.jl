using Match

function kl_divergence(counts1::AbstractVector{Int64}, counts2::AbstractVector{Int64})::Float64
    c1 = sum(counts1)
    c2 = sum(counts2)

    freqs1 = counts1 ./ c1
    freqs2 = counts2 ./ c2

    return sum(freqs1 .* log.(freqs1 ./ freqs2))

end

function mi(counts::Matrix{Int64})::Float64
    #   TODO think about how to do this
end

# function mi(x_count::CountData, y_count::CountData, joint::CountData; estimator::Symbol=:maximum_likelihood, zeta::Vector{Float64})::Float64
#     @match estimator begin
#         :miller_madow =>
#             return miller_madow(x_count) + miller_madow(y_count) - miller_madow(joint)
#         :maximum_likelihood =>
#             return maximum_likelihood(x_count) + maximum_likelihood(y_count) - maximum_likelihood(joint)
#         :grassberger
#         return grassberger(x_count) + grassberger(y_count) - grassberger(joint)
#         :schurmann =>
#             @assert length(zeta) == 3
#         return schurmann(x_count, zeta[1]) + schurmann(y_count[2]) - schurmann(joint[3])
#         _ => error("no match")
#     end
# end

"""
    uncertainty_coefficient(counts::Matrix)

``math
C_{XY} = \frac{I(X;Y)}{H(Y)}
``
"""
function uncertainty_coefficient(counts::Matrix)::Float64
    0.0
end

"""
    redundancy(counts::Matrix)

``math
R = \frac{I(X;Y)}{H(X) + H(Y)}
``
"""

function redundancy(counts::Matrix)::Float64
    0.0
end
