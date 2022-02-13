module FluxRNNs

using Flux
import Flux: Zygote, @functor, OneHotArray, @adjoint, zeros32, glorot_uniform
import Functors
import CUDA: CUDA, CUDNN, CuArray
import CUDA.CUDNN: CUDNN_RNN_TANH, CUDNN_RNN_RELU, CUDNN_RNN_ALGO_STANDARD, CUDNN_RNN_SINGLE_INP_BIAS, CUDNN_UNIDIRECTIONAL, CUDNN_LINEAR_INPUT
export RNN#, LSTM, GRU, GRUv3, Recur

gate(h, n) = (1:h) .+ h*(n-1)
gate(x::AbstractVector, h, n) = @view x[gate(h,n)]
gate(x::AbstractMatrix, h, n) = x[gate(h,n),:]

# Stateful recurrence

"""
    Recur(cell)
`Recur` takes a recurrent cell and makes it stateful, managing the hidden state
in the background. `cell` should be a model of the form:
    h, y = cell(h, x...)
For example, here's a recurrent network that keeps a running total of its inputs:
```julia
accum(h, x) = (h + x, x)
rnn = Flux.Recur(accum, 0)
rnn(2)      # 2
rnn(3)      # 3
rnn.state   # 5
rnn.(1:10)  # apply to a sequence
rnn.state   # 60
```
Folding over a 3d Array of dimensions `(features, batch, time)` is also supported:
```julia
accum(h, x) = (h .+ x, x)
rnn = Flux.Recur(accum, zeros(Int, 1, 1))
rnn([2])                    # 2
rnn([3])                    # 3
rnn.state                   # 5
rnn(reshape(1:10, 1, 1, :)) # apply to a sequence of (features, batch, time)
rnn.state                   # 60
```
"""
mutable struct Recur{T,S}
  cell::T
  state::S
end

struct DenseInput end
struct ConvInput end

input_size(cell) = DenseInput()

function forward!(::DenseInput, m::Recur, x::Union{AbstractVector, AbstractMatrix})
    m.state, y = m.cell(m.state, x)
    return y
end

function forward!(input_type, m::Recur, x)
    h = [forward!(input_type, m, slc) for slc in eachslice(x, dims=ndims(x))]
    sze = size(h[1])
    reshape(reduce(hcat, h), sze[1], sze[2], length(h))
end

(m::Recur)(x) = forward!(input_size(m.cell), m, x)

@functor Recur
trainable(a::Recur) = (a.cell,)

Base.show(io::IO, m::Recur) = print(io, "Recur(", m.cell, ")")

"""
    reset!(rnn)
Reset the hidden state of a recurrent layer back to its original value.
Assuming you have a `Recur` layer `rnn`, this is roughly equivalent to:
```julia
rnn.state = hidden(rnn.cell)
```
"""
reset!(m::Recur) = (m.state = m.cell.state0)
reset!(m) = foreach(reset!, functor(m)[1])


# TODO remove in v0.13
function Base.getproperty(m::Recur, sym::Symbol)
  if sym === :init
    Zygote.ignore() do
      @warn "Recur field :init has been deprecated. To access initial state weights, use m::Recur.cell.state0 instead."
    end
    return getfield(m.cell, :state0)
  else
    return getfield(m, sym)
  end
end

flip(f, xs) = reverse(f.(reverse(xs)))

# Vanilla RNN


struct RNNCell{F,VA, VV, A, S}
    σ::F
    Wi::VA
    Wh::VA
    b::VV
    W::A
    state0::S
end

function RNNCell(σ, W, state0, in, out)
    RNNCell(σ,
              view(W, :, 1:in),
              view(W, :, (in+1):(in+out)),
              view(W, :, size(W, 2)),
              W,
              state0)
end

RNNCell(in::Integer, out::Integer, σ=tanh; init=Flux.glorot_uniform, initb=zeros32, init_state=zeros32) =
    RNNCell(σ, hcat(init(out, in), init(out, out), initb(out)), init_state(out,1), in, out)

function Functors.functor(::Type{<:RNNCell}, m)
    function reconstruct_RNN(ms)
        W = ms.W
        out = size(W, 1)
        in = size(W, 2) - out - 1
        return RNNCell(m.σ, ms.W, ms.state0, in, out)
    end
    return (W = m.W, state0=m.state0), reconstruct_RNN
end

function (m::RNNCell)(h, x)
    σ, Wi, Wh, b = m.σ, m.Wi, m.Wh, m.b
    h = σ.(Wi*x .+ Wh*h .+ b)
    sz = size(x)
    return h, reshape(h, :, sz[2:end]...)
end


function cudnn_compat(m::Recur{<:RNNCell})
    m.cell.σ isa typeof(tanh) || m.cell.σ isa typeof(Flux.relu)
end

# function forward!(input_type, m::Recur{RNNCell}, x::CuArray{Float32, 3})
function (m::Recur{<:RNNCell})(x::CuArray{Float32, 3}) 

    # Check if compatable with cudnn and return description if it is
    compat = cudnn_compat(m)
    if compat
        cudnn_forward!(m, x)
    else
        forward!(input_size(m), m, x)
    end
end

function cudnn_forward!(m::Recur{<:RNNCell}, x)

    cellMode = if m.cell.σ isa typeof(tanh)
        CUDNN.CUDNN_RNN_TANH
    else
        CUDNN.CUDNN_RNN_RELU
    end

    state = if size(m.state, 2) == 1
        state = zeros(Float32, size(m.state, 1), size(x, 2), 1) |> Flux.gpu
        for i in 1:size(x, 2)
            state[:, i, 1] .= m.state
        end
        state
    elseif size(m.state, 2) == size(x, 2)
        reshape(m.state, size(m.state)..., 1)
    end
    
    ret = CUDNN.cudnnRNNForward(m.cell.W, x;
                                hx = state, # TODO: Not sure why this isn't working.
                                hiddenSize=size(m.cell.W, 1),
                                inputSize=size(x, 1), # handled inside cudnnRNNForward
                                algo = CUDNN.CUDNN_RNN_ALGO_STANDARD,
                                cellMode=cellMode,
                                biasMode=CUDNN.CUDNN_RNN_SINGLE_INP_BIAS,
                                dirMode = CUDNN_UNIDIRECTIONAL,
                                inputMode = CUDNN_LINEAR_INPUT,
                                numLayers = 1,
                                dropout = 0);

    m.state = ret[:, :, end]
    ret
end


function Base.show(io::IO, l::RNNCell)
  print(io, "RNNCell(", size(l.Wi, 2), ", ", size(l.Wi, 1))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end

"""
    RNNCell
"""
RNN(a...; ka...) = Recur(RNNCell(a...; ka...))
Recur(m::RNNCell) = Recur(m, m.state0)



@adjoint function Broadcast.broadcasted(f::Recur, args...)
  Zygote.∇map(__context__, f, args...)
end


include("cudnn.jl")

end
