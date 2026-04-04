using SpikenautDistill
using Printf
using Random
using Statistics

Random.seed!(7)

const N_CHANNELS = 16
const WARMUP_TICKS = 5_000
const BENCH_TICKS = 50_000

function random_spikes()
    Float32.(rand(Float32, N_CHANNELS) .< 0.1f0)
end

function random_reward()
    rand(Float32) * 2.0f0 - 1.0f0
end

function run_benchmark()
    net = create_network()

    for _ in 1:WARMUP_TICKS
        eprop_update!(net, random_spikes(), random_reward())
    end

    per_tick_us = Vector{Float32}(undef, BENCH_TICKS)
    wall_start = time_ns()
    for i in 1:BENCH_TICKS
        eprop_update!(net, random_spikes(), random_reward())
        per_tick_us[i] = net.processing_time
    end
    wall_s = (time_ns() - wall_start) / 1e9

    med = median(per_tick_us)
    p95 = quantile(per_tick_us, 0.95)
    p99 = quantile(per_tick_us, 0.99)
    mean_us = mean(per_tick_us)
    ticks_per_sec = BENCH_TICKS / wall_s

    println("=== SpikenautDistill E-prop Tick Benchmark ===")
    @printf("Warmup ticks: %d\n", WARMUP_TICKS)
    @printf("Bench ticks : %d\n", BENCH_TICKS)
    @printf("Wall time   : %.3fs\n", wall_s)
    @printf("Mean tick   : %.3f us\n", mean_us)
    @printf("Median tick : %.3f us\n", med)
    @printf("p95 tick    : %.3f us\n", p95)
    @printf("p99 tick    : %.3f us\n", p99)
    @printf("Throughput  : %.0f ticks/s\n", ticks_per_sec)
end

run_benchmark()
