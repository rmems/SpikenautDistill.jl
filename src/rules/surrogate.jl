"""
Implementations of surrogate gradient functions for backpropagation through spiking neurons.
"""

"""
    surrogate_heaviside(x, γ=1.0)

A simple rectangular surrogate for the Heaviside step function's derivative.

`σ'(x) = γ * max(0, 1 - |x|)`
"""
function surrogate_heaviside(x::Real, γ::Real=10.0)
    return γ * max(0.0, 1.0 - abs(x))
end

"""
    surrogate_sigmoid(x, β=1.0)

A sigmoid-based surrogate gradient.
"""
function surrogate_sigmoid(x::Real, β::Real=1.0)
    return β * exp(-β * abs(x)) / (1.0 + exp(-β * abs(x)))^2
end

"""
    surrogate_exponential(x, α=1.0)

An exponential-based surrogate gradient, useful for its simplicity.
"""
function surrogate_exponential(x::Real, α::Real=1.0)
    return α * exp(-α * abs(x))
end

# Broadcasted versions for applying to arrays of membrane potentials.
# E.g., surrogate_heaviside.([-1.0, 0.5, 1.2])
