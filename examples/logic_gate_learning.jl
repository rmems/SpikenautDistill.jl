using SpikenautDistill
using Printf
using Random
using Statistics

Random.seed!(42)

const N_CHANNELS = 16
const TRAIN_STEPS = 8000
const EVAL_INTERVAL = 1000

function encode_or_inputs(a::Bool, b::Bool)::Vector{Float32}
    spikes = Float32.(rand(Float32, N_CHANNELS) .< 0.01f0)
    spikes[1] = a ? 1.0f0 : 0.0f0
    spikes[2] = b ? 1.0f0 : 0.0f0
    return spikes
end

function network_prediction(net::EPropNetwork)::Float32
    return mean(Float32(neuron.last_spike) for neuron in net.neurons[1:4])
end

function one_or_sample()
    a = rand(Bool)
    b = rand(Bool)
    target = (a || b) ? 1.0f0 : 0.0f0
    return a, b, target
end

function train_or_gate!(net::EPropNetwork)
    for step in 1:TRAIN_STEPS
        a, b, target = one_or_sample()
        spikes = encode_or_inputs(a, b)

        eprop_update!(net, spikes, 0.0f0)
        pred = network_prediction(net)
        pred_bin = pred >= 0.25f0 ? 1.0f0 : 0.0f0
        reward = 1.0f0 - 2.0f0 * abs(pred_bin - target)

        eprop_update!(net, spikes, reward)

        if step % EVAL_INTERVAL == 0
            acc = evaluate_or_accuracy(net)
            @printf("Step %d | OR truth-table accuracy: %.1f%%\n", step, 100 * acc)
        end
    end
end

function evaluate_or_accuracy(net::EPropNetwork)::Float32
    inputs = ((false, false, 0.0f0), (false, true, 1.0f0), (true, false, 1.0f0), (true, true, 1.0f0))
    correct = 0
    trials_per_case = 64

    for (a, b, target) in inputs
        for _ in 1:trials_per_case
            spikes = encode_or_inputs(a, b)
            eprop_update!(net, spikes, 0.0f0)
            pred = network_prediction(net) >= 0.25f0 ? 1.0f0 : 0.0f0
            correct += pred == target ? 1 : 0
        end
    end

    return correct / (length(inputs) * trials_per_case)
end

function main()
    net = create_network()

    println("Training OR-gate temporal task...")
    train_or_gate!(net)

    final_acc = evaluate_or_accuracy(net)
    @printf("Final OR truth-table accuracy: %.2f%%\n", 100 * final_acc)

    out_dir = joinpath(@__DIR__, "out_logic_or")
    export_parameters_mem(net, out_dir)
    println("Exported FPGA Q8.8 parameters to: $out_dir")
end

main()
