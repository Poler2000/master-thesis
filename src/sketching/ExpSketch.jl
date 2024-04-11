function expsketch(stream::Vector{Number}, m::Number, h::Function)
    M = Vector{Number}(Inf, m)

    for (i, element) in enumerate(stream)
        for k in 1:m
            binI = bitstring(i)
            binK = bitstring(k)

            hashValue = h(binI * binK)
            sampleValue = -log(hashValue) / element

            M[k] = min(sampleValue, M[k])
        end
    end
    
    return M
end