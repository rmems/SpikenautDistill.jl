<p align="center">
  <img src="docs/logo.png" width="220" alt="Spikenaut">
</p>

<h1 align="center">SpikenautDistill.jl</h1>
<p align="center">Monte Carlo SNN training and FPGA distillation pipeline</p>

<p align="center">
  <img src="https://img.shields.io/badge/language-Julia-9558B2" alt="Julia">
  <img src="https://img.shields.io/badge/license-GPL--3.0-orange" alt="GPL-3.0">
  <img src="https://img.shields.io/badge/status-active-success" alt="Status: Active">
</p>

---

Train large spiking neural network ensembles using E-prop, then distill them down to
compact FPGA-ready parameter sets exported as Q8.8 `.mem` files for Vivado/Quartus
`$readmemh` synthesis.

Maintained as part of the [Spikenaut organization](https://github.com/Spikenaut/spikenaut-ml).

## Features

- `export_parameters_mem(net, output_dir)` — direct bridge from Julia training to FPGA RAM init files
- `EPropNetwork` — LIF network with eligibility traces and surrogate-gradient training
- `eprop_update!(net, spikes, reward)` — online reward-modulated E-prop weight update
- `distill_to_channels(weights, n_channels)` — reduce large networks to deployment size
- `export_distilled_mem(distilled, output_dir)` — export distilled matrix to FPGA `.mem` files
- `extract_weights(net)` — snapshot current network parameters

## Installation

```julia
using Pkg
Pkg.add("SpikenautDistill")
```

## Quick Start

```julia
using SpikenautDistill
using Random

# Create and train a network
net = create_network()              # 16-neuron, 16-channel LIF network

for t in 1:1000
    spikes = Float32.(rand(Float32, 16) .< 0.1f0)  # sample spike input
    reward = 0.8f0                                  # your reward signal
    eprop_update!(net, spikes, reward)
end

# Distill to 4-channel FPGA deployment
W = extract_weights(net)         # 16×16 Float32
W4 = distill_to_channels(W, 4)   # 4×4 Float32

# Export for Vivado $readmemh
export_distilled_mem(W4, "vivado_params/")

# Or export full network parameters directly
export_parameters_mem(net, "vivado_full/")
```

## FPGA Integration (Vivado/Quartus)

`export_parameters_mem` is the hardware bridge: train in Julia, then drop generated
hex files directly into your FPGA flow.

1. Train your `EPropNetwork` in Julia.
2. Run `export_parameters_mem(net, "vivado_params/")`.
3. Load emitted files in HDL (`$readmemh`):
   - `parameters.mem` (thresholds)
   - `parameters_weights.mem` (synaptic weights, row-major)
   - `parameters_decay.mem` (membrane decay)
4. Synthesize in Vivado or Quartus for Artix-7/Basys 3 style deployments.

This removes manual post-processing between simulation and bare-metal deployment.

## Reproducible Hello World

Run the self-contained OR-gate temporal learning example:

```bash
julia --project=. examples/logic_gate_learning.jl
```

The script trains a small network, prints accuracy checkpoints, and exports FPGA-ready
Q8.8 `.mem` files to `examples/out_logic_or/`.

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
loadable by `WeightRam.sv` from [spikenaut-core-sv](https://github.com/Spikenaut/spikenaut-core-sv).

## Extracted from Production

This package was extracted from [Eagle-Lander](https://github.com/rmems/Eagle-Lander),
a private low-latency temporal signal-processing supervisor. The training and
distillation logic is now decoupled from domain-specific internals for general use in
bio-MEMS control, micro/nano-device automation, and edge robotics.

## Benchmarking

Measure per-tick training latency locally:

```bash
julia --project=. benchmarks/eprop_tick_benchmark.jl
```

The benchmark reports median/p95/p99 µs per `eprop_update!` tick and total ticks/sec,
so you can document results from your target workstation (for example, AMD Ryzen 9
9950X) directly in your org docs.

Sample output format:

```text
=== SpikenautDistill E-prop Tick Benchmark ===
Warmup ticks: 5000
Bench ticks : 50000
Wall time   : 0.078s
Mean tick   : 1.499 us
Median tick : 0.260 us
p95 tick    : 0.370 us
p99 tick    : 0.790 us
Throughput  : 643756 ticks/s
```

## Part of the Spikenaut Ecosystem

| Library | Purpose |
|---------|---------|
| [SpikenautLSM.jl](https://github.com/Spikenaut/SpikenautLSM.jl) | GPU sparse liquid state machine |
| [spikenaut-fpga](https://github.com/Spikenaut/spikenaut-fpga) | Rust-side FPGA export |
| [spikenaut-core-sv](https://github.com/Spikenaut/spikenaut-core-sv) | FPGA neuron IP cores |

## License

GPL-3.0-or-later
