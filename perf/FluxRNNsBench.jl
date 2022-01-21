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

function get_runtimes(create_features::Function, cell; Ns = [2, 20, 200], Ts = [1, 8, 16, 64], cuda=false)

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
            x = create_features(n, t)
            rnn_model = cell(n, n)

            timings = run_benchmark_recur(rnn_model, x, cuda=cuda)
            for k in keys(timings)
                push!(RUNTIMES[k], ("$(cell)", cuda, n, t, get_statistics(timings[k])...))
            end
        end
    end
    RUNTIMES
end

function run_benchmark(cells=[RNN, GRU, LSTM]; name="$(git_head())", cuda=false, kwargs...)

    save_dir = joinpath("results", name)
    if !isdir(save_dir)
        mkpath(save_dir)
    end

    for cell in cells
        println("""$(cell)-$(cuda ? "GPU" : "CPU")""")
        cell_svpath = joinpath(save_dir, """$(cell)-$(cuda ? "GPU" : "CPU")""")
        if !isdir(cell_svpath)
            mkdir(cell_svpath)
        end
        rt = get_runtimes(cell; cuda=cuda, kwargs...) do n, t
            [rand(Float32, n, n) for _ in 1:t]
        end
        for k in keys(rt)
            filename = joinpath(cell_svpath, "$(k).csv")
            CSV.write(filename, rt[k])
        end
        println()
        println("$(cell)3d-CPU")
        rt = get_runtimes(cell; cuda=cuda, kwargs...) do n, t
            rand(Float32, n, n, t)
        end
        for k in keys(rt)
            filename = joinpath(cell_svpath, "$(k)_3d.csv")
            CSV.write(filename, rt[k])
        end
    end
end

end


