


using CUDA
import Flux: Flux
import Functors
# import Adapt: Adapt, adapt_storage, adapt
# import CUDA: CUDA, CUDNN

# CuRNN{T} = Flux.RNNCell{<:Union{typeof(tanh),typeof(relu)},<:CuArray{T,2},<:CuArray{T,1}}
# CuGRU{T} = Flux.GRUCell{<:CuArray{T,2},<:CuArray{T,1}}
# CuLSTM{T} = Flux.LSTMCell{<:CuArray{T,2},<:CuArray{T,1}}
# CuRNNs{T} = Union{CuRNN{T},CuGRU{T},CuLSTM{T}}

struct CuRNNCell{F,VA, VV, A, S}
    σ::F
    Wi::VA
    Wh::VA
    b::VV
    W::A
    state0::S
end

function CuRNNCell(σ, W, state0, in, out)
    CuRNNCell(σ,
              view(W, :, 1:in),
              view(W, :, (in+1):(in+out)),
              view(W, :, size(W, 2)),
              W,
              state0)
end

CuRNNCell(in::Integer, out::Integer, σ=tanh; init=Flux.glorot_uniform, initb=zeros32, init_state=zeros32) =
    CuRNNCell(σ, hcat(init(out, in), init(out, out), initb(out)), init_state(out,1), in, out)

function Functors.functor(::Type{<:CuRNNCell}, m)
    function reconstruct_RNN(ms)
        W = ms.W
        out = size(W, 1)
        in = size(W, 2) - out - 1
        return CuRNNCell(m.σ, ms.W, ms.state0, in, out)
    end
    return (W = m.W, state0=m.state0), reconstruct_RNN
end

function (m::CuRNNCell)(h, x)
    σ, Wi, Wh, b = m.σ, m.Wi, m.Wh, m.b
    h = σ.(Wi*x .+ Wh*h .+ b)
    sz = size(x)
    return h, reshape(h, :, sz[2:end]...)
end

# function forward!(input_type, m::Recur{CuRNNCell}, x::CuArray{Float32, 3})
function (m::Recur{<:CuRNNCell})(x::CuArray{Float32, 3}) 

    # Check if compatable with cudnn and return description if it is
    compat, descriptor = true, nothing #check_cudnn_rnn_compat(m)
    if compat
        cudnn_forward!(m, x)
    else
        forward!(input_size(m), m, x)
    end
end

function cudnn_forward!(m, x)

    cellMode = if m.cell.σ isa typeof(tanh)
        CUDNN.CUDNN_RNN_TANH
    else
        CUDNN.CUDNN_RNN_RELU
    end

    ret = CUDNN.cudnnRNNForward(m.cell.W, x;
                                # hx=m.state, # TODO: Not sure why this isn't working.
                                hiddenSize=size(m.cell.W, 1),
                                inputSize=size(x, 1),
                                cellMode=cellMode,
                                biasMode=CUDNN.CUDNN_RNN_SINGLE_INP_BIAS);

    m.state = ret[:, :, end]
    ret
end


function Base.show(io::IO, l::CuRNNCell)
  print(io, "CuRNNCell(", size(l.Wi, 2), ", ", size(l.Wi, 1))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end

"""
    CuRNNCell
"""
CuRNN(a...; ka...) = Recur(CuRNNCell(a...; ka...))
Recur(m::CuRNNCell) = Recur(m, m.state0)



