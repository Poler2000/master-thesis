
module ExpSketch

using Random
export StreamElement, expsketch, fast_expsketch

struct StreamElement
    id1::Number
    id2::Number
    weight::Number
end

# simple implementation of ExpSketch algorithm
# stream - a stream of elements. Each element is a pair: (id, weight)
# m - number of sketch elements
# h - hash function
function expsketch(stream::Vector{<:StreamElement}, m::Number, h::Function)
    M = fill(Inf, m)

    for element in stream
        binNode1 = bitstring(element.id1)
        binNode2 = bitstring(element.id2)
        binI = element.id1 > element.id2 ? binNode2 * binNode1 : binNode1 * binNode2
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
# stream - a stream of elements. Each element is a pair: (id, weight)
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
        binNode1 = bitstring(element.id1)
        binNode2 = bitstring(element.id2)
        binI = element.id1 > element.id2 ? binNode2 * binNode1 : binNode1 * binNode2
        for k in 1:m
            binK = bitstring(k)

            hashValue = h(binI * binK)
            sampleValue = -log(hashValue) / element.weight

            S += sampleValue / (m - k + 1)
            if S > maxValue
                break
            end

            Random.seed!(hash(binI))
            r = rand(k:m)

            tmp = P[k];
            P[k] = P[r]
            P[r] = tmp;

            j = P[k]

            if M[k] == maxValue
                updateMax = true
            end

            if S < M[j]
                M[j] = S
            end
        end

        if updateMax
            maxValue = maximum(M)
        end
    end
    
    return M
end

end