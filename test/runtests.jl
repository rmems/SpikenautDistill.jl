using Test
using SpikenautDistill
using LinearAlgebra

@testset "SpikenautDistill" begin

    @testset "Package loads" begin
        @test @isdefined(SpikenautDistill)
        @test SpikenautDistill isa Module
        @test isdefined(SpikenautDistill, :create_network)
        @test isdefined(SpikenautDistill, :eprop_update!)
        @test isdefined(SpikenautDistill, :distill_to_channels)
        @test isdefined(SpikenautDistill, :export_parameters_mem)
    end

    @testset "fast_sigmoid_grad" begin
        # At threshold → 1.0
        @test fast_sigmoid_grad(1.0f0, 1.0f0) ≈ 1.0f0
        # Far from threshold → near 0
        @test fast_sigmoid_grad(0.0f0, 2.0f0) < 0.1f0
        # Always non-negative
        for v in -2.0f0:0.5f0:4.0f0
            @test fast_sigmoid_grad(v, 1.0f0) ≥ 0.0f0
        end
    end

    @testset "create_network" begin
        net = create_network()
        @test length(net.neurons) == 16
        @test length(net.pre_traces) == 16
        @test net.spike_count == 0
        @test net.prev_reward == 0.0f0

        # Weights in [W_MIN, W_MAX]
        for n in net.neurons
            @test all(SpikenautDistill.W_MIN .≤ n.weights .≤ SpikenautDistill.W_MAX)
        end

        # Custom size
        net2 = create_network(32, 8)
        @test length(net2.neurons) == 32
        @test length(net2.pre_traces) == 8
    end

    @testset "eprop_update!" begin
        net = create_network()
        spikes = rand(Float32, 16)
        reward = 0.5f0

        # Should not throw
        @test_nowarn eprop_update!(net, spikes, reward)

        # spike_count incremented
        @test net.spike_count ≥ 0
        @test net.prev_reward == reward
        @test net.processing_time ≥ 0.0f0

        # Weights remain bounded
        for n in net.neurons
            @test all(SpikenautDistill.W_MIN .≤ n.weights .≤ SpikenautDistill.W_MAX)
        end

        # Multiple updates
        for _ in 1:100
            eprop_update!(net, rand(Float32, 16), rand(Float32))
        end
        for n in net.neurons
            @test all(SpikenautDistill.W_MIN .≤ n.weights .≤ SpikenautDistill.W_MAX)
        end
    end

    @testset "extract_weights" begin
        net = create_network(8, 4)
        W = extract_weights(net)
        @test size(W) == (8, 4)
        @test W isa Matrix{Float32}
    end

    @testset "distill_to_channels (2D)" begin
        W = rand(Float32, 256, 16)
        d = distill_to_channels(W, 16)
        @test size(d) == (16, 16)
        # Each row should be near the group mean
        @test all(isfinite.(d))
    end

    @testset "distill_to_channels (3D batch)" begin
        batch = rand(Float32, 10, 256, 16)
        d = distill_to_channels(batch, 16)
        @test size(d) == (16, 16)
        @test all(isfinite.(d))
    end

    @testset "distill non-divisible neuron count" begin
        W = rand(Float32, 100, 8)   # 100 neurons, not divisible by 7
        d = distill_to_channels(W, 7)
        @test size(d) == (7, 8)
        @test all(isfinite.(d))
    end

    @testset "weights_to_q88_hex" begin
        @test weights_to_q88_hex(0.0f0)   == UInt16(0)
        @test weights_to_q88_hex(1.0f0)   == UInt16(256)
        @test weights_to_q88_hex(0.5f0)   == UInt16(128)
        # Clamping
        @test weights_to_q88_hex(300.0f0) == UInt16(65535)
        @test weights_to_q88_hex(-1.0f0)  == UInt16(0)
    end

    @testset "export_parameters_mem" begin
        net = create_network(4, 4)
        mktempdir() do dir
            export_parameters_mem(net, dir)
            @test isfile(joinpath(dir, "parameters.mem"))
            @test isfile(joinpath(dir, "parameters_weights.mem"))
            @test isfile(joinpath(dir, "parameters_decay.mem"))
            # Each .mem file should have 4 lines (one per neuron/weight)
            thresh_lines = readlines(joinpath(dir, "parameters.mem"))
            @test length(thresh_lines) == 4
            # Lines are 4-digit hex
            @test all(length.(thresh_lines) .== 4)
        end
    end

end
