# mem_export.jl — Q8.8 Fixed-Point .mem File Export
#
# Writes Vivado-compatible $readmemh hex files for threshold, weight,
# and decay parameter RAMs on Artix-7 / Basys3 FPGAs.
#
# Q8.8 encoding: value → round(value × 256) clamped to [0, 65535] → 4-digit hex.
#
# Extracted from Eagle-Lander CORE/training/julia_eprop.jl and
# Spikenaut-Capital/brain/monte_carlo_spikenaut.jl.

"""
    weights_to_q88_hex(value::Float32) -> UInt16

Convert a floating-point weight to Q8.8 unsigned fixed-point (16-bit).
Clamps to `[0.0, 255.996]` (the representable range of unsigned Q8.8).
"""
@inline function weights_to_q88_hex(value::Float32)::UInt16
    UInt16(round(clamp(value * 256.0f0, 0.0f0, 65535.0f0)))
end

"""
    export_parameters_mem(network::EPropNetwork, output_dir::String)

Write three Vivado `\$readmemh`-compatible files to `output_dir`:
- `parameters.mem` — neuron thresholds in Q8.8
- `parameters_weights.mem` — synaptic weights in Q8.8 (row-major, neuron × channel)
- `parameters_decay.mem` — membrane decay rates in Q8.8

Creates `output_dir` if it does not exist.

# Example
```julia
using SpikenautDistill
net = create_network()
# ... train ...
export_parameters_mem(net, "vivado_params/")
```
"""
function export_parameters_mem(network::EPropNetwork, output_dir::String)
    mkpath(output_dir)

    # Thresholds
    open(joinpath(output_dir, "parameters.mem"), "w") do f
        for neuron in network.neurons
            @printf(f, "%04X\n", weights_to_q88_hex(neuron.threshold))
        end
    end

    # Weights (row-major: for each neuron, all channel weights)
    open(joinpath(output_dir, "parameters_weights.mem"), "w") do f
        for neuron in network.neurons
            for w in neuron.weights
                @printf(f, "%04X\n", weights_to_q88_hex(w))
            end
        end
    end

    # Decay rates
    open(joinpath(output_dir, "parameters_decay.mem"), "w") do f
        for neuron in network.neurons
            @printf(f, "%04X\n", weights_to_q88_hex(neuron.decay_rate))
        end
    end
end

"""
    export_distilled_mem(distilled::AbstractMatrix{Float32}, output_dir::String;
                         threshold_col=1, weight_cols=2:17, decay_cols=18:33)

Write Q8.8 `.mem` files from a distilled `n_channels × n_params` matrix.
Column layout follows the convention used in `monte_carlo_spikenaut.jl`:
- Column 1 → thresholds
- Columns 2–17 → 16 synaptic weights (scaled by ×256 for Q8.8)
- Columns 18–33 → decay rates

Adjust `threshold_col`, `weight_cols`, `decay_cols` to match your parameter
layout.  Creates `output_dir` if it does not exist.
"""
function export_distilled_mem(
    distilled::AbstractMatrix{Float32},
    output_dir::String;
    threshold_col::Int = 1,
    weight_cols::UnitRange{Int} = 2:17,
    decay_cols::UnitRange{Int} = 18:33,
)
    mkpath(output_dir)
    n_channels, n_params = size(distilled)

    open(joinpath(output_dir, "parameters.mem"), "w") do f
        for ch in 1:n_channels
            raw = Int16(clamp(round(distilled[ch, threshold_col] * 128), -128, 127))
            @printf(f, "%02X\n", reinterpret(UInt8, raw))
        end
    end

    if last(weight_cols) ≤ n_params
        open(joinpath(output_dir, "parameters_weights.mem"), "w") do f
            for ch in 1:n_channels
                for col in weight_cols
                    val = distilled[ch, col] * 256.0f0
                    raw = Int8(clamp(round(val), -128, 127))
                    @printf(f, "%02X\n", reinterpret(UInt8, raw))
                end
            end
        end
    end

    if last(decay_cols) ≤ n_params
        open(joinpath(output_dir, "parameters_decay.mem"), "w") do f
            for ch in 1:n_channels
                for col in decay_cols
                    raw = Int8(clamp(round(distilled[ch, col] * 128), -128, 127))
                    @printf(f, "%02X\n", reinterpret(UInt8, raw))
                end
            end
        end
    end
end
