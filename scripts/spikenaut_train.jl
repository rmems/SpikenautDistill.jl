#!/usr/bin/env julia
# spikenaut_train.jl — Spikenaut LIF Trainer (Julia)
#
# Loads temporal event-stream JSONL (or any NeuromorphicSnapshot JSONL),
# runs batched LIF + reward-modulated STDP in Julia's fast matrix ops,
# then writes snn_model.json + .mem files that train_snn already understands.
#
# Usage:
#   julia spikenaut_train.jl [data_path] [epochs] [out_dir]
#   julia spikenaut_train.jl research/sensor_stream.jsonl 20 research

using JSON3, LinearAlgebra, Printf, Statistics

# ── Config ────────────────────────────────────────────────────────────
const N_NEURONS   = 16
const N_CHANNELS  = 16
const DECAY       = 0.85f0
const STDP_LR     = 0.01f0
const EPROP_LR    = 0.002f0
const W_MIN       = 0.0f0
const W_MAX       = 2.0f0
const THRESH_INIT = 1.0f0
const TRACE_LAMBDA = 0.85f0

# ── LIF State ─────────────────────────────────────────────────────────
mutable struct LIFBank
    v       ::Vector{Float32}   # membrane potentials
    thresh  ::Vector{Float32}   # thresholds
    weights ::Matrix{Float32}   # [N_NEURONS × N_CHANNELS]
    decay   ::Vector{Float32}   # per-neuron decay rates
    spikes  ::Vector{Bool}
    pre_tr  ::Vector{Float32}   # OTTT presynaptic traces
    elig    ::Matrix{Float32}   # eligibility traces [N × CH]
end

function LIFBank()
    LIFBank(
        zeros(Float32, N_NEURONS),
        fill(THRESH_INIT, N_NEURONS),
        fill(Float32(1.0 / N_CHANNELS), N_NEURONS, N_CHANNELS),
        fill(0.85f0, N_NEURONS),
        falses(N_NEURONS),
        zeros(Float32, N_CHANNELS),
        zeros(Float32, N_NEURONS, N_CHANNELS),
    )
end

# ── Stimulus/reward extraction from generic sample records ────────────
function to_stimuli(sample)::Vector{Float32}
    raw = if haskey(sample, :spikes)
        sample[:spikes]
    elseif haskey(sample, "spikes")
        sample["spikes"]
    elseif haskey(sample, :inputs)
        sample[:inputs]
    elseif haskey(sample, "inputs")
        sample["inputs"]
    else
        error("Sample is missing `spikes` or `inputs` field")
    end

    stim = zeros(Float32, N_CHANNELS)
    n = min(length(raw), N_CHANNELS)
    for i in 1:n
        stim[i] = clamp(Float32(raw[i]), 0f0, 1f0)
    end
    return stim
end

function sample_reward(sample)::Float32
    r = if haskey(sample, :reward)
        sample[:reward]
    elseif haskey(sample, "reward")
        sample["reward"]
    elseif haskey(sample, :target_reward)
        sample[:target_reward]
    elseif haskey(sample, "target_reward")
        sample["target_reward"]
    elseif haskey(sample, :reward_hint)
        sample[:reward_hint]
    elseif haskey(sample, "reward_hint")
        sample["reward_hint"]
    else
        0.5f0
    end
    return clamp(Float32(r), 0f0, 1f0)
end

# ── Fast-sigmoid surrogate gradient ───────────────────────────────────
@inline surrogate(v, θ) = 1f0 / (1f0 + abs(10f0 * (v - θ)))^2

# ── One training tick ─────────────────────────────────────────────────
function tick!(bank::LIFBank, stim::Vector{Float32}, reward::Float32)
    # 1. Poisson pre-spikes
    pre = Float32.(rand(Float32, N_CHANNELS) .< stim)

    # 2. OTTT presynaptic trace
    bank.pre_tr .= TRACE_LAMBDA .* bank.pre_tr .+ pre
    pre_snap = copy(bank.pre_tr)

    # 3. LIF forward pass (batched)
    input = bank.weights * stim          # [N_NEURONS]
    bank.v .= bank.decay .* bank.v .+ input

    # 4. Fire & reset
    bank.spikes .= bank.v .>= bank.thresh
    bank.v[bank.spikes] .= 0f0

    # 5. STDP: potentiate on co-activation
    for i in 1:N_NEURONS
        if bank.spikes[i]
            bank.weights[i, :] .+= STDP_LR .* pre_snap
        end
    end

    # 6. E-prop eligibility + weight update
    for i in 1:N_NEURONS
        dz = bank.spikes[i] ? 1f0 : surrogate(bank.v[i], bank.thresh[i])
        bank.elig[i, :] .= TRACE_LAMBDA .* bank.elig[i, :] .+ pre_snap .* dz
        bank.weights[i, :] .+= reward .* bank.elig[i, :] .* EPROP_LR
    end

    # 7. L1 synaptic scaling (budget = 1.0 per neuron)
    for i in 1:N_NEURONS
        s = sum(bank.weights[i, :])
        if s > 1f-6
            bank.weights[i, :] ./= s
        end
    end
    clamp!(bank.weights, W_MIN, W_MAX)

    sum(bank.spikes)
end

# ── Load JSONL ────────────────────────────────────────────────────────
function load_jsonl(path)
    samples = []
    open(path) do f
        for line in eachline(f)
            isempty(strip(line)) && continue
            try
                rec = JSON3.read(line)
                sample = if haskey(rec, :sensor_stream)
                    rec[:sensor_stream]
                elseif haskey(rec, "sensor_stream")
                    rec["sensor_stream"]
                elseif haskey(rec, :sample)
                    rec[:sample]
                elseif haskey(rec, "sample")
                    rec["sample"]
                else
                    rec
                end
                push!(samples, sample)
            catch
            end
        end
    end
    samples
end

# ── Load chunked directory ────────────────────────────────────────────
function load_chunked_dir(dir_path)
    samples = []
    # Support chunk files with patterns like chunk_* / *_chunk_*
    chunk_files = String[]
    for entry in readdir(dir_path)
        full_path = joinpath(dir_path, entry)
        isfile(full_path) || continue
        # Match generic chunk files
        if occursin(r"chunk", entry) && !endswith(entry, ".md") && !endswith(entry, ".json")
            push!(chunk_files, full_path)
        end
    end
    
    # Sort to ensure deterministic order
    sort!(chunk_files)
    
    if isempty(chunk_files)
        @warn "No chunk files found in directory: $dir_path"
        return samples
    end
    
    println("Loading $(length(chunk_files)) chunk files...")
    for (i, chunk_file) in enumerate(chunk_files)
        print("  [$i/$(length(chunk_files))] Loading $(basename(chunk_file))... ")
        chunk_samples = load_jsonl(chunk_file)
        append!(samples, chunk_samples)
        println("$(length(chunk_samples)) records")
    end
    
    samples
end

# ── Smart load: auto-detect single file vs chunked directory ──────────
function load_data(data_path)
    if isdir(data_path)
        # Directory mode: load all chunk files
        load_chunked_dir(data_path)
    elseif isfile(data_path)
        # Single file mode: legacy behavior
        load_jsonl(data_path)
    else
        error("Data path not found: $data_path")
    end
end

# ── Main ──────────────────────────────────────────────────────────────
function main()
    length(ARGS) >= 1 || error("Usage: julia spikenaut_train.jl <data_path> [epochs] [out_dir]")
    data_path = ARGS[1]
    epochs    = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 20
    out_dir   = length(ARGS) >= 3 ? ARGS[3] : "out_train"

    isdir(data_path) || isfile(data_path) || error("Data path not found: $data_path")
    mkpath(out_dir)

    println("=== Spikenaut Julia Trainer ===")
    println("Data   : $data_path")
    println("Epochs : $epochs")
    println("Out    : $out_dir")

    print("Loading samples... ")
    samples = load_data(data_path)
    println("$(length(samples)) total records")
    isempty(samples) && error("No valid samples found.")

    bank = LIFBank()

    for epoch in 1:epochs
        total_reward = 0f0
        total_spikes = 0

        # Reset membrane each epoch
        fill!(bank.v, 0f0)

        t0 = time()
        for sample in samples
            stim   = to_stimuli(sample)
            reward = sample_reward(sample)
            total_spikes  += tick!(bank, stim, reward)
            total_reward  += reward
        end
        elapsed = time() - t0

        n = length(samples)
        avg_r  = total_reward / n
        s_rate = total_spikes / (n * N_NEURONS)
        w_mean = mean(bank.weights)
        w_std  = std(bank.weights)
        ms_tick = elapsed * 1000 / n

        @printf("Epoch %3d/%d | reward=%.4f | spike_rate=%.3f | w=%.4f±%.4f | %.3fms/tick\n",
                epoch, epochs, avg_r, s_rate, w_mean, w_std, ms_tick)
    end

    # ── Export ────────────────────────────────────────────────────────
    # snn_model.json  (compatible with train_snn reload)
    neurons_json = [
        Dict(
            "threshold"          => bank.thresh[i],
            "decay_rate"         => bank.decay[i],
            "membrane_potential" => bank.v[i],
            "weights"            => collect(bank.weights[i, :]),
            "last_spike"         => false,
        )
        for i in 1:N_NEURONS
    ]
    open(joinpath(out_dir, "snn_model.json"), "w") do f
        JSON3.write(f, Dict("neurons" => neurons_json, "source" => "spikenaut_julia"))
    end

    # Q8.8 fixed-point .mem files for FPGA
    q88(v) = UInt16(clamp(round(Int, v * 256), 0, 65535))
    fmt(v) = uppercase(string(q88(v), base=16, pad=4))

    open(joinpath(out_dir, "parameters.mem"), "w") do f
        for i in 1:N_NEURONS; println(f, fmt(bank.thresh[i])); end
    end
    open(joinpath(out_dir, "parameters_weights.mem"), "w") do f
        for i in 1:N_NEURONS, ch in 1:N_CHANNELS
            println(f, fmt(bank.weights[i, ch]))
        end
    end
    open(joinpath(out_dir, "parameters_decay.mem"), "w") do f
        for i in 1:N_NEURONS; println(f, fmt(bank.decay[i])); end
    end

    println("\nExported:")
    println("  $(joinpath(out_dir, "snn_model.json"))")
    println("  $(joinpath(out_dir, "parameters.mem"))")
    println("  $(joinpath(out_dir, "parameters_weights.mem"))")
    println("  $(joinpath(out_dir, "parameters_decay.mem"))")
    println("SUCCESS: Spikenaut trained on temporal stream data.")
end

main()
