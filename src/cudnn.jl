


using CUDA
# import CUDA: CUDA, CUDNN

# CuRNN{T} = Flux.RNNCell{<:Union{typeof(tanh),typeof(relu)},<:CuArray{T,2},<:CuArray{T,1}}
# CuGRU{T} = Flux.GRUCell{<:CuArray{T,2},<:CuArray{T,1}}
# CuLSTM{T} = Flux.LSTMCell{<:CuArray{T,2},<:CuArray{T,1}}
# CuRNNs{T} = Union{CuRNN{T},CuGRU{T},CuLSTM{T}}


struct CuRNNCell{F,A,S}
  σ::F
  W::A
  state0::S
end

CuRNNCell(in::Integer, out::Integer, σ=tanh; init=Flux.glorot_uniform, initb=zeros32, init_state=zeros32) = 
  CuRNNCell(σ, hcat(init(out, in+out), initb(out)), init_state(out,1))

function (m::CuRNNCell)(h, x)
    W = m.W
    h = σ.(view(W, :, 1:size(x, 1))*x .+ view(W, :, (size(x,1)+1):(size(W, 2)-1))*h .+ view(W, :, size(W,2)))
    sz = size(x)
    return h, reshape(h, :, sz[2:end]...)
end

@functor CuRNNCell

function Base.show(io::IO, l::RNNCell)
  # print(io, "CuRNNCell(", size(l.Wi, 2), ", ", size(l.Wi, 1))
  # l.σ == identity || print(io, ", ", l.σ)
  # print(io, ")")
end

"""
    CuRNNCell
"""
CuRNN(a...; ka...) = Recur(CuRNNCell(a...; ka...))
Recur(m::CuRNNCell) = Recur(m, m.state0)

# function CUDNN.RNNDesc(m::CuRNN{T})
# end



function forward!(input_type, m::Recur{CuRNNCell}, x::CuArray{Float32, 3})

    # Specialize for cudnn.
    ret = CUDNN.cudnnRNNForward(m.cell.W, x;
                                hx=m.state,
                                hiddenSize=size(m.cell.W, 1),
                                inputSize=size(x, 1),
                                cellMode=CUDNN.CUDNN_RNN_TANH,
                                biasMode=CUDNN.CUDNN_RNN_SINGLE_INP_BIAS);
    m.state = ret[:, :, end]
    ret
end



