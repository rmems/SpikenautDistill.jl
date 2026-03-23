# distill.jl — Ensemble Weight Distillation
#
# Reduce large N-neuron weight tensors from Monte Carlo SNN training
# down to compact M-channel representations suitable for FPGA deployment.
#
# Extracted from Eagle-Lander Spikenaut-Capital/brain/monte_carlo_spikenaut.jl.

"""
    distill_to_channels(weights::AbstractMatrix{Float32}, n_channels::Int) -> Matrix{Float32}

Distill an `n_neurons × n_params` weight matrix to an
`n_channels × n_params` matrix by averaging over neuron groups.

Each channel covers `floor(n_neurons / n_channels)` neurons.  Any
remainder neurons (when `n_neurons` is not evenly divisible) are
absorbed into the last channel's slice.

# Arguments
- `weights`: `n_neurons × n_params` weight matrix (e.g. from `extract_weights`)
- `n_channels`: number of output channels (e.g. 16 for a 16-channel FPGA model)

# Returns
`n_channels × n_params` matrix where each row is the mean of a neuron group.

# Example
```julia
using SpikenautDistill
net = create_network(256, 16)
W   = extract_weights(net)          # 256 × 16
d   = distill_to_channels(W, 16)    # 16  × 16 — ready for FPGA export
```
"""
function distill_to_channels(weights::AbstractMatrix{Float32}, n_channels::Int)::Matrix{Float32}
    n_neurons, n_params = size(weights)
    neurons_per_ch = n_neurons ÷ n_channels

    distilled = zeros(Float32, n_channels, n_params)
    for ch in 1:n_channels
        start_idx = (ch - 1) * neurons_per_ch + 1
        # Last channel absorbs any remainder
        end_idx   = ch == n_channels ? n_neurons : ch * neurons_per_ch
        slice     = @view weights[start_idx:end_idx, :]
        distilled[ch, :] = vec(mean(slice; dims=1))
    end
    return distilled
end

"""
    distill_to_channels(weights::AbstractArray{Float32,3}, n_channels::Int) -> Matrix{Float32}

3-D variant: distills an `n_trajectories × n_neurons × n_params` batch tensor
(as produced by a Monte Carlo ensemble solve) to `n_channels × n_params` by
first averaging over trajectories, then distilling neuron groups.
"""
function distill_to_channels(weights::AbstractArray{Float32,3}, n_channels::Int)::Matrix{Float32}
    # Average over trajectory dimension → (n_neurons × n_params)
    avg = dropdims(mean(weights; dims=1); dims=1)
    return distill_to_channels(avg, n_channels)
end
