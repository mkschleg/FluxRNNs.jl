


function forward!(input_type, m::Recur{RNN}, x::CuArray)
    h = [forward!(input_type, m, slc) for slc in eachslice(x, dims=ndims(x))]
    sze = size(h[1])
    reshape(reduce(hcat, h), sze[1], sze[2], length(h))
end

