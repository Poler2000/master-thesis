include("SLA.jl")

struct Sketch
    embeddings::Matrix{Float64}
end

struct Config
    L::Integer
    alpha::Number
    hashes::Vector{Function}
end

function calculate_single_sample(hash::Function, element::Number, i::Number)
    return -log(hash(i)) / element
end

function generate_sample(row, config::Config)
    return argmin(i -> calculate_single_sample(config.hashes[j], row[i], i), 1:length(row))
end

function node_sketch(A::SparseMatrixCSC{Number, Integer}, k::Int64, config::Config)::Sketch
    sketch = Sketch(Matrix(10, 10))
    
    if k > 2
        sketch = node_sketch(A, k - 1, alpha)
        for (i, row) in enumerate(eachrow(A))
            # TODO: get k-th order SLA
            for j in 1:config.L
                sketch.embeddings[j, i] = generate_sample(row, config)
            end
            println(row)
        end
    elseif k == 2
        for row in eachrow(A)
            println(row)
        end
    end

    return sketch
end