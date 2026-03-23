"""
    SpikenautDistill

Monte Carlo SNN training utilities and FPGA distillation pipeline.

Provides the complete train-on-GPU → distill → deploy-on-FPGA workflow:

- **E-prop training** (`eprop.jl`) — reward-modulated eligibility propagation
  with OTTT presynaptic traces, surrogate gradients, and L1 normalization.
  Sub-50µs per tick on a modern CPU with `@simd` + `@inbounds` optimization.

- **Distillation** (`distill.jl`) — reduce a large ensemble weight tensor to
  a compact N-channel representation by averaging over neuron groups.

- **Q8.8 `.mem` export** (`mem_export.jl`) — write Vivado-compatible `\$readmemh`
  files for threshold, weight, and decay parameter RAMs on Artix-7 FPGAs.

Typical workflow:
```julia
using SpikenautDistill

# 1. Create and train a network on recorded telemetry
net = create_network()
for (spikes, reward) in training_data
    eprop_update!(net, spikes, reward)
end

# 2. Distill the weight matrix to 16 output channels
distilled = distill_to_channels(extract_weights(net), 16)

# 3. Export Q8.8 hex files for Vivado
export_parameters_mem(net, "output/")
```
"""
module SpikenautDistill

using LinearAlgebra
using Printf
using Statistics

include("eprop.jl")
include("distill.jl")
include("mem_export.jl")

export LIFNeuron, EPropNetwork, TrainingMetrics
export create_network, eprop_update!, fast_sigmoid_grad
export extract_weights, distill_to_channels
export export_parameters_mem, weights_to_q88_hex

end # module
