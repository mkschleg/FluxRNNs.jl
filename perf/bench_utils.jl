using BenchmarkTools
import Flux: Flux, gradient, gpu
using CUDA
using Zygote: pullback


fw(m, x) = m(x)
bw(back) = back(1f0)
fwbw(m, ps, x) = gradient(() -> sum(m(x)), ps)


fw_recur(m, X::AbstractVector{<:AbstractArray}) = [m(x_t) for x_t in X]
fw_recur(m, X::AbstractArray{<:Number}) = m(X)
bw_recur(back) = back(1f0)
fwbw_recur(m, ps, X) = gradient(() -> sum(sum(fw_recur(m, X))), ps)


function run_benchmark_recur(model, x; cuda=false, show=false)
    
    t_fw, t_bw, t_fwbw = if cuda
        model_gpu = model |> gpu
        x_gpu = x |> gpu

        ps = Flux.params(model_gpu)
        y, back = pullback(() -> sum(sum(fw_recur(model_gpu, x_gpu))), ps)
        
        CUDA.allowscalar(false)
        # CUDA.device!(3)
        fw_recur(model_gpu, x_gpu); GC.gc(); CUDA.reclaim(); #warmup
        t_fw = @benchmark CUDA.@sync(fw_recur($model_gpu, $x_gpu)) teardown=(GC.gc(); CUDA.reclaim())

        bw_recur(back); GC.gc(); CUDA.reclaim(); #warmup
        t_bw = @benchmark CUDA.@sync(bw_recur($back)) teardown=(GC.gc(); CUDA.reclaim())
        
        fwbw_recur(model_gpu, ps, x_gpu); GC.gc(); CUDA.reclaim(); #warmup
        t_fwbw = @benchmark CUDA.@sync(fwbw_recur($model_gpu, $ps, $x_gpu)) teardown=(GC.gc(); CUDA.reclaim())
        
        t_fw, t_bw, t_fwbw
    else
        ps = Flux.params(model)
        y, back = pullback(() -> sum(sum(fw_recur(model, x))), ps)

        Flux.reset!(model)
        fw_recur(model, x)  #warmup
        t_fw = @benchmark fw_recur($model, $x)

        Flux.reset!(model)
        bw_recur(back)  #warmup
        t_bw = @benchmark bw_recur($back)

        Flux.reset!(model)
        fwbw_recur(model, ps, x) # warmup
        t_fwbw = @benchmark fwbw_recur($model, $ps, $x)

        Flux.reset!(model)
        t_fw, t_bw, t_fwbw
    end

    if show
        println("\t\tforward: ")
        @show t_fw
        println("\t\tbackward: ")
        @show t_bw
        println("\t\tforw and back: ")
        @show t_fwbw
    end

    (fw=t_fw, bw=t_bw, fwbw=t_fwbw)
end
