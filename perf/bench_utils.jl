using BenchmarkTools
import Flux: Flux, gradient
using CUDA
using Zygote: pullback


fw(m, x) = m(x)
bw(back) = back(1f0)
fwbw(m, ps, x) = gradient(() -> sum(m(x)), ps)


  
function run_benchmark(model, x; cuda=true)
    
    if cuda 
        model = model |> gpu
        x = x |> gpu
    end

    ps = Flux.params(model)
    y, back = pullback(() -> sum(model(x)), ps)


    if cuda
        CUDA.allowscalar(false)
        # CUDA.device!(3)
        println("  forward")
        fw(model, x); GC.gc(); CUDA.reclaim(); #warmup
        @btime CUDA.@sync(fw($model, $x)) teardown=(GC.gc(); CUDA.reclaim())

        println("  backward")
        bw(back); GC.gc(); CUDA.reclaim(); #warmup
        @btime CUDA.@sync(bw($back)) teardown=(GC.gc(); CUDA.reclaim())
        
        println("  forw and back")
        fwbw(model, ps, x); GC.gc(); CUDA.reclaim(); #warmup
        @btime CUDA.@sync(fwbw($model, $ps, $x)) teardown=(GC.gc(); CUDA.reclaim())
    else
        println("  forward")
        fw(model, x)  #warmup
        @btime fw($model, $x)

        println("  backward")
        bw(back)  #warmup
        @btime bw($back)

        println("  forw and back")
        fwbw(model, ps, x) # warmup
        @btime fwbw($model, $ps, $x)
    end
end

fw_recur(m, X) = [m(x_t) for x_t in X]
bw_recur(back) = back(1f0)
fwbw_recur(m, ps, x) = gradient(() -> sum(sum(fw_recur(m, x))), ps)

function run_benchmark_recur(model, x; cuda=true, show=false)
    
    if cuda 
        model = model |> gpu
        x = x |> gpu
    end

    ps = Flux.params(model)
    y, back = pullback(() -> sum(sum(fw_recur(model, x))), ps)


    t_fw, t_bw, t_fwbw = if cuda
        CUDA.allowscalar(false)
        # CUDA.device!(3)
        fw_recur(model, x); GC.gc(); CUDA.reclaim(); #warmup
        t_fw = @benchmark CUDA.@sync(fw_recur($model, $x)) teardown=(GC.gc(); CUDA.reclaim())

        bw_recur(back); GC.gc(); CUDA.reclaim(); #warmup
        t_bw = @benchmark CUDA.@sync(bw_recur($back)) teardown=(GC.gc(); CUDA.reclaim())
        
        fwbw_recur(model, ps, x); GC.gc(); CUDA.reclaim(); #warmup
        t_fwbw = @benchmark CUDA.@sync(fwbw_recur($model, $ps, $x)) teardown=(GC.gc(); CUDA.reclaim())
        
        t_fw, t_bw, t_fwbw
    else
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

function run_benchmark_recur_3d(model, x; cuda=true, show=false)
    
    if cuda 
        model = model |> gpu
        x = x |> gpu
    end

    ps = Flux.params(model)
    y, back = pullback(() -> sum(sum(fw_recur(model, x))), ps)


    t_fw, t_bw, t_fwbw = if cuda
        CUDA.allowscalar(false)
        # CUDA.device!(3)
        fw(model, x); GC.gc(); CUDA.reclaim(); #warmup
        t_fw = @benchmark CUDA.@sync(fw($model, $x)) teardown=(GC.gc(); CUDA.reclaim())

        bw(back); GC.gc(); CUDA.reclaim(); #warmup
        t_bw = @benchmark CUDA.@sync(bw($back)) teardown=(GC.gc(); CUDA.reclaim())
        
        fwbw(model, ps, x); GC.gc(); CUDA.reclaim(); #warmup
        t_fwbw = @benchmark CUDA.@sync(fwbw($model, $ps, $x)) teardown=(GC.gc(); CUDA.reclaim())
        
        t_fw, t_bw, t_fwbw
    else
        Flux.reset!(model)
        fw(model, x)  #warmup
        t_fw = @benchmark fw($model, $x)

        Flux.reset!(model)
        bw(back)  #warmup
        t_bw = @benchmark bw($back)

        Flux.reset!(model)
        fwbw(model, ps, x) # warmup
        t_fwbw = @benchmark fwbw($model, $ps, $x)

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
