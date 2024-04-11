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

function generate_sample(row, config::Config, j)
    return argmin(i -> calculate_single_sample(config.hashes[j], row[i], i), 1:length(row))
end

function generate_sample_precalculated(row, hash_matrix, j)
    return argmin(i -> (hash_matrix[i, j] / row[i]), 1:length(row))
end

function generate_sample_precalculated(row, hash_matrix, j, neighbours)
    return argmin(i -> (hash_matrix[i, j] / row[i]), neighbours)
end

function node_sketch(A::SparseMatrixCSC{<:Number, <:Integer}, k::Integer, config::Config)::Sketch
    row_count = size(A, 1)
    sketch = Sketch(zeros(config.L, row_count))
    
    if k > 2
        sketch = node_sketch(A, k - 1, config)
        for (i, row) in enumerate(eachrow(A))
            println(i)
            v = Vector{Float64}(undef,row_count)
            neighbours = findall(x -> x != 0, row)

            for l in 1:config.L
                s = 0
                for n in neighbours
                    # change to dictionary
                    for j in 1:config.L
                        if sketch.embeddings[j, n] == l
                            s += 1
                        end
                    end
                end
                s *= config.alpha / config.L
                v[l] = s
            end

            # TODO: get k-th order SLA
            for j in 1:config.L
                sketch.embeddings[j, i] = generate_sample(v, config, j)
            end
            #println(typeof(row))
        end
    elseif k == 2
        for (i, row) in enumerate(eachrow(A))
            println(i)

            # TODO: get k-th order SLA
            for j in 1:config.L
                sketch.embeddings[j, i] = generate_sample(row, config, j)
            end
            #println(row)
        end
    end

    return sketch
end

function node_sketch_precalculated(A::SparseMatrixCSC{<:Number, <:Integer}, k::Integer, config::Config, hash_matrix)::Sketch
    row_count = size(A, 1)
    sketch = Sketch(zeros(config.L, row_count))

    #sla = zeros(size(A, 1), size(A, 2))
    
    if k > 2
        sketch = node_sketch_precalculated(A, k - 1, config, hash_matrix)
        for (i, row) in enumerate(eachrow(A))
            println("$i, $k")

            v = Vector{Float64}(undef,row_count)
            neighbours = findall(x -> x != 0, row)

            mydict = Dict(i => 0 for i in 0:row_count)

            for n in neighbours
                for j in 1:config.L
                    mydict[sketch.embeddings[j, n]] += 1
                end
            end

            for l in 1:row_count
                s = mydict[l]
                s *= config.alpha / config.L
                v[l] = s + row[l]
            end

            # TODO: get k-th order SLA
            for j in 1:config.L
                sketch.embeddings[j, i] = generate_sample_precalculated(v, hash_matrix, j, neighbours)
            end
            #println(typeof(row))
        end
    elseif k == 2
        for (i, row) in enumerate(eachrow(A))
            println("$i, $k")
            neighbours = findall(x -> x != 0, row)

            # TODO: get k-th order SLA
            for j in 1:config.L
                sketch.embeddings[j, i] = generate_sample_precalculated(row, hash_matrix, j, neighbours)
            end
            #println(row)
        end
    end

    return sketch
end