<p align="center">
  <img src="docs/logo.png" width="220" alt="Spikenaut">
</p>

<h1 align="center">SpikenautDistill.jl</h1>
<p align="center">Monte Carlo SNN training and FPGA distillation pipeline</p>

<p align="center">
  <img src="https://img.shields.io/badge/language-Julia-9558B2" alt="Julia">
  <img src="https://img.shields.io/badge/license-GPL--3.0-orange" alt="GPL-3.0">
</p>

---

Train large spiking neural network ensembles using E-prop, then distill them down to
compact FPGA-ready parameter sets exported as Q8.8 `.mem` files for Vivado/Quartus
`$readmemh` synthesis.

## Features

- `EPropNetwork` — LIF network with eligibility traces and surrogate-gradient training
- `eprop_update!(net, reward)` — online reward-modulated E-prop weight update
- `distill_to_channels(weights, n_channels)` — reduce large networks to deployment size
- `export_parameters_mem(path, weights)` — write Q8.8 hex `.mem` file for FPGA synthesis
- `extract_weights(net)` — snapshot current network parameters

## Installation

```julia
using Pkg
Pkg.add("SpikenautDistill")
```

## Quick Start

```julia
using SpikenautDistill

# Create and train a network
net = create_network()           # 16-neuron, 16-channel LIF network

for t in 1:1000
    reward = compute_reward()    # your reward signal
    eprop_update!(net, reward)
end

# Distill to 4-channel FPGA deployment
W = extract_weights(net)          # 16×16 Float32
W4 = distill_to_channels(W, 4)   # 4×4 Float32

# Export for Vivado $readmemh
export_distilled_mem("weights.mem", W4)
```

## E-prop Learning Rule

Online eligibility-propagation with surrogate gradients:

```
ΔW_ij = η · r(t) · e_ij(t)
e_ij(t) = λ · e_ij(t-1) + x_i(t-1) · σ'(v_j(t))
σ'(v) = 1 / (1 + β|v|)²   (fast sigmoid surrogate)
```

*Bellec et al. (2020); Zenke & Ganguli (2018)*

## Q8.8 Fixed-Point Export

Weights are quantized as `round(w × 256)` and written as 4-digit hex lines, directly
loadable by `WeightRam.sv` from [spikenaut-core-sv](https://github.com/rmems/spikenaut-core-sv).

## Extracted from Production

This package was extracted from [Eagle-Lander](https://github.com/rmems/Eagle-Lander),
a private neuromorphic GPU supervisor. The training and distillation logic has been
decoupled from all trading-specific code so anyone can use it with their own reward
signals and deployment targets.

## Part of the Spikenaut Ecosystem

| Library | Purpose |
|---------|---------|
| [SpikenautLSM.jl](https://github.com/rmems/SpikenautLSM.jl) | GPU sparse liquid state machine |
| [spikenaut-fpga](https://github.com/rmems/spikenaut-fpga) | Rust-side FPGA export |
| [spikenaut-core-sv](https://github.com/rmems/spikenaut-core-sv) | FPGA neuron IP cores |

## License

GPL-3.0-or-later
