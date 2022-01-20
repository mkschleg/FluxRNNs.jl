module FluxRNNsBench

using ReTest
using FluxRNNs
using Random
using CUDA
using Statistics
using ProgressMeter
using DataFrames
using CSV
using Git



include("bench_utils.jl")

function git_head()
    try
        s = read(`$(Git.git()) rev-parse HEAD`, String)
        s[1:end-1]
    catch
        "0"
    end
end

# rng = Random.MersenneTwister(10)

function get_statistics(t)
    tms = t.times
    mean(tms), var(tms), median(tms), minimum(tms), maximum(tms), t.allocs, t.memory
end

function get_runtimes(cell; Ns = [2, 20, 200], Ts = [1, 8, 16, 64])

    RUNTIMES = Dict(k=>DataFrames.DataFrame(
        cell=String[],
        gpu=Bool[],
        n=Int[],
        t=Int[],
        mean=Float64[],
        variance=Float64[],
        median=Float64[],
        minimum=Float64[],
        maximum=Float64[],
        allocs=Int[],
        memory=Int[]) for k in [:fw, :bw, :fwbw])


    Random.seed!(10)

    @showprogress "MatrixSize: " for n in Ns
        @showprogress "TemporalLength: " 1 for t in Ts
            x = [randn(Float32, n, n) for _ in 1:t]
            rnn_model = cell(n, n)

            timings = run_benchmark_recur(rnn_model, x, cuda=false)
            for k in keys(timings)
                push!(RUNTIMES[k], ("$(cell)", false, n, t, get_statistics(timings[k])...))
            end

            # figure out cuda after cpu figured out...
            # if CUDA.functional()
            #     t_fw, t_bw, t_fwbw = run_benchmark_recur(rnn_model, x, cuda=true)
            # end
        end
    end
    RUNTIMES
end


function get_runtimes_3d(cell; Ns = [2, 20, 200], Ts = [1, 8, 16, 64])

    RUNTIMES = Dict(k=>DataFrames.DataFrame(
        cell=String[],
        gpu=Bool[],
        n=Int[],
        t=Int[],
        mean=Float64[],
        variance=Float64[],
        median=Float64[],
        minimum=Float64[],
        maximum=Float64[],
        allocs=Int[],
        memory=Int[]) for k in [:fw, :bw, :fwbw])

    Random.seed!(10)
    @showprogress "MatrixSize: " for n in Ns
        @showprogress "TemporalLength: " 1 for t in Ts
            x = randn(Float32, n, n, t)
            rnn_model = cell(n, n)

            timings = run_benchmark_recur(rnn_model, x, cuda=false)
            for k in keys(timings)
                push!(RUNTIMES[k], ("$(cell)", false, n, t, get_statistics(timings[k])...))
            end

            # figure out cuda after cpu figured out...
            # if CUDA.functional()
            #     t_fw, t_bw, t_fwbw = run_benchmark_recur(rnn_model, x, cuda=true)
            # end
        end
    end
    RUNTIMES
end

function run_benchmark(cells=[RNN, GRU, LSTM]; save_dir="results/$(git_head())", kwargs...)

    if !isdir(save_dir)
        mkpath(save_dir)
    end

    for cell in cells
        println("$(cell)-CPU")
        cell_svpath = joinpath(save_dir, "$(cell)-CPU")
        if !isdir(cell_svpath)
            mkdir(cell_svpath)
        end
        rt = get_runtimes(cell; kwargs...)
        for k in keys(rt)
            filename = joinpath(cell_svpath, "$(k).csv")
            CSV.write(filename, rt[k])
        end

        println("$(cell)3d-CPU")
        rt = get_runtimes_3d(cell; kwargs...)
        for k in keys(rt)
            filename = joinpath(cell_svpath, "$(k)_3d.csv")
            CSV.write(filename, rt[k])
        end
    end
end

end


