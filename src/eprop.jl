# eprop.jl — E-prop + OTTT Training Core
#
# High-performance Julia implementation of eligibility propagation with
# online temporal trace training (OTTT). Designed for sub-50µs per tick
# on modern multi-core CPUs using @simd + @inbounds optimization.
#
# Extracted from Eagle-Lander CORE/training/julia_eprop.jl.

# ── Constants ──────────────────────────────────────────────────────────────

const N_NEURONS    = 16
const N_CHANNELS   = 16
const DECAY        = 0.85f0
const STDP_LR      = 0.01f0
const EPROP_LR     = 0.002f0
const W_MIN        = 0.0f0
const W_MAX        = 2.0f0
const TRACE_LAMBDA = 0.85f0

# ── Data Structures ────────────────────────────────────────────────────────

"""
    LIFNeuron

Single leaky integrate-and-fire neuron with E-prop eligibility traces.

Fields:
- `membrane_potential` — current membrane voltage (resets to 0 on spike)
- `threshold` — firing threshold (sampled from U[0.75, 1.25] at init)
- `decay_rate` — membrane time constant (= `DECAY` constant)
- `weights` — N_CHANNELS synaptic weights
- `last_spike` — whether this neuron fired on the last tick
- `eligibility` — N_CHANNELS eligibility traces for E-prop weight updates
"""
mutable struct LIFNeuron
    membrane_potential::Float32
    threshold::Float32
    decay_rate::Float32
    weights::Vector{Float32}
    last_spike::Bool
    eligibility::Vector{Float32}
end

"""
    EPropNetwork

Full E-prop network with presynaptic traces and statistics.
"""
mutable struct EPropNetwork
    neurons::Vector{LIFNeuron}
    pre_traces::Vector{Float32}
    prev_reward::Float32
    spike_count::Int
    processing_time::Float32
end

"""
    TrainingMetrics

Per-epoch summary statistics from `eprop_update!` calls.
"""
mutable struct TrainingMetrics
    avg_reward::Float32
    spike_rate::Float32
    weight_norm::Float32
    processing_time_us::Float32
end

# ── Fast Sigmoid Surrogate Gradient ───────────────────────────────────────

"""
    fast_sigmoid_grad(v::Float32, θ::Float32) -> Float32

Fast sigmoid surrogate gradient for E-prop learning. Replaces the
Heaviside step derivative with a smooth approximation:

    f′(v) = 1 / (1 + |10 · (v − θ)|)²

Properties:
- f′(θ) = 1.0 (maximum gradient at threshold)
- f′(θ±0.1) ≈ 0.50 (half-maximum nearby)
- No `exp()` calls — just multiply, abs, divide
"""
@inline function fast_sigmoid_grad(v::Float32, θ::Float32)::Float32
    x = 10.0f0 * (v - θ)
    denom = 1.0f0 + abs(x)
    return 1.0f0 / (denom * denom)
end

# ── Network Initialization ─────────────────────────────────────────────────

"""
    create_network(n_neurons=N_NEURONS, n_channels=N_CHANNELS) -> EPropNetwork

Create an `EPropNetwork` with Xavier-like random weight initialization.
"""
function create_network(n_neurons::Int = N_NEURONS, n_channels::Int = N_CHANNELS)
    neurons = [LIFNeuron(
        0.0f0,
        rand(Float32) * 0.5f0 + 0.75f0,
        DECAY,
        rand(Float32, n_channels) .* 0.5f0 .+ 0.75f0,
        false,
        zeros(Float32, n_channels)
    ) for _ in 1:n_neurons]

    EPropNetwork(neurons, zeros(Float32, n_channels), 0.0f0, 0, 0.0f0)
end

# ── Core E-prop + OTTT Update ─────────────────────────────────────────────

"""
    eprop_update!(network::EPropNetwork, spikes::Vector{Float32}, reward::Float32)

Main training step implementing:
1. OTTT presynaptic traces: â_j[t+1] = λ · â_j[t] + s_j[t+1]
2. Forward pass through LIF neurons with membrane dynamics
3. E-prop eligibility traces with surrogate gradients
4. Reward-modulated weight updates (`Δw = reward · e · lr`)
5. L1 synaptic normalization per neuron

Target: <50µs per tick on a modern CPU with `@inbounds` + `@simd`.
"""
@inline function eprop_update!(network::EPropNetwork, spikes::Vector{Float32}, reward::Float32)
    start_time = time_ns()
    n_neurons  = length(network.neurons)
    n_channels = length(network.pre_traces)

    # 1. OTTT presynaptic traces
    @simd for j in 1:n_channels
        @inbounds network.pre_traces[j] = TRACE_LAMBDA * network.pre_traces[j] + spikes[j]
    end
    pre_trace_snapshot = copy(network.pre_traces)

    # 2. Forward pass
    post_spikes         = zeros(Float32, n_neurons)
    membrane_snapshots  = zeros(Float32, n_neurons)

    @simd for i in 1:n_neurons
        @inbounds neuron = network.neurons[i]
        input_current = 0.0f0
        @simd for j in 1:n_channels
            @inbounds input_current += neuron.weights[j] * pre_trace_snapshot[j]
        end
        @inbounds neuron.membrane_potential = DECAY * neuron.membrane_potential + input_current
        spike = neuron.membrane_potential >= neuron.threshold
        neuron.last_spike = spike
        @inbounds post_spikes[i]        = spike ? 1.0f0 : 0.0f0
        @inbounds membrane_snapshots[i] = neuron.membrane_potential
        if spike
            @inbounds neuron.membrane_potential = 0.0f0
        end
    end

    # 3. Eligibility traces with surrogate gradients
    @simd for i in 1:n_neurons
        @inbounds neuron = network.neurons[i]
        pseudo_dz = post_spikes[i] > 0.0f0 ?
            1.0f0 :
            fast_sigmoid_grad(membrane_snapshots[i], neuron.threshold)
        @simd for j in 1:n_channels
            @inbounds neuron.eligibility[j] =
                TRACE_LAMBDA * neuron.eligibility[j] + pre_trace_snapshot[j] * pseudo_dz
        end
    end

    # 4. Reward-modulated weight updates
    @simd for i in 1:n_neurons
        @inbounds neuron = network.neurons[i]
        @simd for j in 1:n_channels
            @inbounds dw = reward * neuron.eligibility[j] * EPROP_LR
            @inbounds neuron.weights[j] = clamp(neuron.weights[j] + dw, W_MIN, W_MAX)
        end
    end

    # 5. L1 synaptic normalization
    @simd for i in 1:n_neurons
        @inbounds neuron = network.neurons[i]
        total_weight = sum(neuron.weights)
        if total_weight > Float32(1e-6)
            scale = 1.0f0 / total_weight
            @simd for j in 1:n_channels
                @inbounds neuron.weights[j] = clamp(neuron.weights[j] * scale, W_MIN, W_MAX)
            end
        end
    end

    network.spike_count    += round(Int, sum(post_spikes))
    network.prev_reward     = reward
    network.processing_time = (time_ns() - start_time) / 1000.0f0  # µs
end

# ── Weight Extraction ──────────────────────────────────────────────────────

"""
    extract_weights(network::EPropNetwork) -> Matrix{Float32}

Return an `n_neurons × n_channels` weight matrix from the network.
"""
function extract_weights(network::EPropNetwork)::Matrix{Float32}
    n = length(network.neurons)
    c = length(network.neurons[1].weights)
    W = Matrix{Float32}(undef, n, c)
    for (i, neuron) in enumerate(network.neurons)
        W[i, :] = neuron.weights
    end
    return W
end
