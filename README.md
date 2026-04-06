# SpikenautDistill.jl

**Modular online training for spiking neural networks in Julia — E-prop, OTTT, and more. Works with pure SNNs or hybrid SNN+LLM systems.**

`SpikenautDistill.jl` is a flexible and performant library for training spiking neural networks (SNNs) using online (event-based) learning rules. It is designed to be framework-agnostic, allowing researchers to bring their own models, loss functions, and data sources.

## Core Philosophy

- **Bring any loss**: The training process is driven by a user-provided loss function. This can be a standard metric like mean squared error for pure SNN tasks, or it can be a cross-entropy loss derived from a frozen large language model (LLM) in a hybrid setup.
- **Apply any rule**: The library provides a modular system for learning rules, starting with e-prop and OTTT. Researchers can easily add their own rules.
- **Update only the SNN**: In hybrid systems, the library is designed to update only the SNN parameters, leaving the external model (like an LLM) frozen.

## Quick Start (Pure SNN Training)

Here's a simple example of how to train an SNN using a standard loss function.

```julia
using SpikenautDistill

# 1. Define your SNN model
mutable struct MySNN
    weights::Matrix{Float32}
end

model = MySNN(rand(Float32, 10, 10))

# 2. Create a batch of spike data
spike_data = rand(0:1, 10, 100)
spike_batch = SpikeBatch(spike_data, nothing, nothing)

# 3. Define a loss function
mse_loss(output) = sum(output.logits .^ 2)

# 4. Run a training step
model, state = train_step!(model, spike_batch, mse_loss, rule=:eprop)

println("Loss: ", state.loss)
```

For a complete, runnable example, see [`examples/pure_snn_training.jl`](file:///home/raulmc/SpikenautDistill.jl/examples/pure_snn_training.jl).

## Hybrid Usage (with OLMoE Loss via `spikenaut-spine`)

`SpikenautDistill.jl` is perfect for hybrid systems where an SNN is trained to distill knowledge from a larger model. The key is to provide a loss function that captures the output of the external model.

```julia
# --- In your spikenaut-spine listener ---

# Assume `llm_targets` are received from the LLM via the spine
llm_targets = receive_from_spine().targets

# Create a closure for the loss function that captures the targets
loss_fn = (output) -> cross_entropy(output.logits, llm_targets)

# Run the training step
model, state = train_step!(model, spike_batch, loss_fn, rule=:eprop)

# Send the gradients back to the spine to update the SNN in Rust
send_to_spine(GradientUpdate(state.gradients))
```

For a complete, runnable example, see [`examples/hybrid_olmoe_training.jl`](file:///home/raulmc/SpikenautDistill.jl/examples/hybrid_olmoe_training.jl).

## Available Rules

- `:eprop`: Eligibility propagation.
- `:ottt`: Online Spatio-Temporal Trace Training.

## Custom Loss Functions

Any function that takes the model's output and returns a scalar loss is a valid loss function. The output of the model is whatever the `forward` function of your model returns.

## Integration with the Spikenaut Ecosystem

`SpikenautDistill.jl` is designed to integrate seamlessly with the other components of the Spikenaut ecosystem:

- **`spikenaut-spine`**: The spine can be used to pass spike data and LLM targets to `SpikenautDistill.jl` and to receive gradients for updating the SNN.
- **`spikenaut-hybrid`**: The hybrid system can use `SpikenautDistill.jl` as its training engine for the SNN component.

## Contributing

Contributions are welcome! Please open an issue or pull request to discuss your ideas.
