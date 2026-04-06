"""
Core data structures for SpikenautDistill.
"""

# A batch of spike data, typically received from a simulation or hardware.
struct SpikeBatch
    # Spike trains, represented as a sparse matrix or a vector of vectors.
    spikes::Any 
    # Optional: Timestamps for each spike, if not implicitly defined by array index.
    times::Union{Nothing, Vector{Float32}}
    # Optional: Targets for supervised learning, e.g., from an LLM.
    targets::Any
end

# A batch of eligibility traces, used by learning rules like e-prop.
struct TraceBatch
    # Traces for each neuron/synapse. The structure will depend on the model.
    traces::Any
end

# Holds the state of the training process for a single step or epoch.
mutable struct TrainingState
    # The current loss value.
    loss::Float32
    # Metrics to log, e.g., accuracy, firing rates.
    metrics::Dict{Symbol, Any}
    # Eligibility traces if applicable.
    traces::Union{Nothing, TraceBatch}
    # Gradient-like updates to be applied to the model.
    gradients::Any

    function TrainingState(; loss=0.0f0, metrics=Dict(), traces=nothing, gradients=nothing)
        new(loss, metrics, traces, gradients)
    end
end
