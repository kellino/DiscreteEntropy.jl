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
