
module ExpSketch

using Random
export StreamElement, expsketch, fast_expsketch

struct StreamElement
    id::Number
    weight::Number
end

# simple implementation of ExpSketch algorithm
# stream - a stream of elements
# m - number of sketch elements
# h - hasz function
function expsketch(stream::Vector{<:StreamElement}, m::Number, h::Function)
    M = fill(Inf, m)

    for element in stream
        binI = bitstring(element.id)
        for k in 1:m
            binK = bitstring(k)

            hashValue = h(binI * binK)
            sampleValue = -log(hashValue) / element.weight

            M[k] = min(sampleValue, M[k])
        end
    end
    
    return M
end

# FastExpSketch algorithm
# stream - a stream of elements
# m - number of sketch elements
# h - hash function
function fast_expsketch(stream::Vector{<:StreamElement}, m::Number, h::Function)
    permInit = collect(1:m)
    M = fill(Inf, m)
    maxValue = Inf

    for element in stream
        S = 0
        updateMax = false
        P = copy(permInit)
        binI = bitstring(element.id)
        for k in 1:m
            binK = bitstring(k)

            hashValue = h(binI * binK)
            sampleValue = -log(hashValue) / element.weight

            S += sampleValue / (m - k + 1)
            if S > maxValue
                break
            end

            Random.seed!(element.id)
            r = rand(k:m)

            tmp = P[k];
            P[k] = P[r]
            P[r] = tmp;

            j = P[k]

            if M[k] == maxValue
                updateMax = true
            end

            M[j] = min(M[j], S)
        end

        if updateMax
            maxValue = maximum(M)
        end
    end
    
    return M
end

end