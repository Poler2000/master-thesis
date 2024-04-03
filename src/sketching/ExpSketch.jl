function expsketch(stream::Vector{Number}, m::Number, h::Function)
    M = Vector{Number}(10000, m)

    for (i, element) in enumerate(stream)
        for k in 1:m
            binI = bitstring(i)
            binK = bitstring(k)

            hashValue = h(i + k)
            sampleValue = -log(u) / element
            

        end
    end 
end