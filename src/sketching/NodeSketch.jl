module NodeSketch

include("ExpSketch.jl")
include("Logger.jl")

using DataStructures
using SparseArrays
using .ExpSketch
using .Logger
using Base.Threads
using ConcurrentCollections

export SLA, Sketch
export nodesketch, edgesketch
export nodesketch_pure, edgesketch_pure

global calculated_hashes = 0

struct Sketch
    embeddings::Matrix{Float64}
    similarity_matrix::Matrix{Float64}
    calculated_hashes::Int
end

function nodesketch(
    A::Matrix{Float64}, 
    order::Integer, 
    sketch_dimensions::Integer, 
    alpha::Number)::Sketch

    global calculated_hashes = 0
    ## Although it's more intuitive to iterate row-wise, 
    ## column-wise processing is slightly more efficient due to sparse matrix implementation
    sla = to_sla(Matrix(A'))
    ##hash_matrix = construct_hash_matrix(size(sla, 1), sketch_dimensions)
    hash_functions = [x -> (hash((x, seed)) % Int64(1e9)) / 1e9 for seed in rand(Int64, sketch_dimensions)]
    col_count = size(A, 2)

    embeddings = nodesketch_core(sla, order, sketch_dimensions, alpha, hash_functions)

    similarity_matrix = zeros(col_count, col_count)

    for i in 1:col_count
        for j in i:col_count
            matching_count = 0
            for k in 1:sketch_dimensions
                if embeddings[k, i] == embeddings[k, j]
                    matching_count += 1
                end
            end
            count = matching_count / sketch_dimensions

            similarity_matrix[i, j] = count
            if i != j
                similarity_matrix[j, i] = count
            end
        end
    end

    return Sketch(embeddings, similarity_matrix, calculated_hashes)
end

# Pure version of the nodesketch algorithm, without similarity matrix computation
function nodesketch_pure(
    A::Matrix{Float64}, 
    order::Integer, 
    sketch_dimensions::Integer, 
    alpha::Number)

    global calculated_hashes = 0
    # Although it's more intuitive to iterate row-wise, 
    # column-wise processing is slightly more efficient due to sparse matrix implementation
    sla = to_sla(Matrix(A'))
    col_count = size(sla, 2)
    hash_functions = [x -> (hash((x, seed)) % Int64(1e9)) / 1e9 for seed in rand(Int64, sketch_dimensions)]
    embeddings = nodesketch_core(sla, order, sketch_dimensions, alpha, hash_functions)
    similarity_matrix = zeros(col_count, col_count)

    return Sketch(embeddings, similarity_matrix, calculated_hashes)
end

function edgesketch(
    A::Matrix{Float64}, 
    order::Integer, 
    sketch_dimensions::Integer, 
    alpha::Number)::Sketch

    reset_calculated_hashes()
    # Although it's more intuitive to iterate row-wise, 
    # column-wise processing is slightly more efficient due to sparse matrix implementation
    sla = to_sla(Matrix(A'))
    h = x -> (hash((x, 123)) % Int64(1e9)) / 1e9

    sketch = edgesketch_similarity(sla, order, sketch_dimensions, alpha, h)

    return sketch
end

# Pure version of the edgesketch algorithm, without similarity matrix computation
function edgesketch_pure(
    A::Matrix{Float64}, 
    order::Integer, 
    sketch_dimensions::Integer, 
    alpha::Number)
    
    reset_calculated_hashes()

    # Although it's more intuitive to iterate row-wise, 
    # column-wise processing is slightly more efficient due to sparse matrix implementation
    sla = to_sla(Matrix(A'))
    h = x -> (hash((x, 123)) % Int64(1e9)) / 1e9

    embeddings = edgesketch_core(sla, sketch_dimensions, h)

    return Sketch(embeddings, zeros(size(sla, 2), size(sla, 2)), calculated_hashes)
end

function nodesketch_core(
    A::Matrix{Float64}, 
    order::Integer, 
    sketch_dimensions::Integer, 
    alpha::Number,
    hash_functions)::Matrix{Float64}

    col_count = size(A, 2)
    embeddings = zeros(sketch_dimensions, col_count)

    if order > 2
        embeddings = nodesketch_core(A, order - 1, sketch_dimensions, alpha, hash_functions)
        for (i, col) in enumerate(eachcol(A))
            v = fill(0.0, col_count)
            neighbours = findall(x -> x != 0, col)

            element_dict = Dict(i => 0 for i in 0:col_count)

            for n in neighbours
                for j in 1:sketch_dimensions
                    element_dict[embeddings[j, n]] += 1
                end
            end

            for l in 1:col_count
                s = element_dict[l]
                s *= alpha / sketch_dimensions
                v[l] = s + col[l]
            end

            neighbours = findall(x -> x != 0, v)
            embeddings[:, i] = Float64[generate_sample(v, hash_functions, j, neighbours) for j in 1:sketch_dimensions]
        end


    elseif order == 2
        for (i, col) in enumerate(eachcol(A))
            neighbours = findall(x -> x != 0, col)
            embeddings[:, i] = Float64[generate_sample(col, hash_functions, j, neighbours) for j in 1:sketch_dimensions]
        end
    end

    return embeddings
end

function edgesketch_core(
    A::Matrix{Float64}, 
    sketch_dimensions::Integer, 
    h::Function)::Matrix{Float64}

    col_count = size(A, 2)
    embeddings = zeros(sketch_dimensions, col_count)

    for (i, col) in enumerate(eachcol(A))
        neighbours = [StreamElement(index, i, weight) 
            for (index, weight) in enumerate(col) if weight != 0]
            
        embeddings[:, i] = fast_expsketch(
            neighbours, 
            sketch_dimensions, 
            h)
    end
    return embeddings
end

function edgesketch_similarity(
    A::Matrix{Float64}, 
    order::Integer, 
    sketch_dimensions::Integer, 
    alpha::Number,
    h::Function)::Sketch

    col_count = size(A, 2)
    if order > 2
        sketch = edgesketch_similarity(A, order - 1, sketch_dimensions, alpha, h)

        ak = A^(order - 1)
        sketches_with_neighbourhoods = Dict{Int, Vector{Float64}}()
        
        for i in 1:col_count
            k_neighbours = findall(x -> x > 0, ak[:, i])
            v = fill(typemax(Float64), sketch_dimensions)
            for n in k_neighbours
                for j in 1:sketch_dimensions
                    v[j] = min(v[j], sketch.embeddings[j, n])
                end
            end
            sketches_with_neighbourhoods[i] = v
        end

        for i in 1:col_count
            for j in i:col_count
                matching_count = 0
                for k in 1:sketch_dimensions
                    if sketches_with_neighbourhoods[i][k] == sketches_with_neighbourhoods[j][k]
                        matching_count += 1
                    end
                end
                count = matching_count / sketch_dimensions

                sketch.similarity_matrix[i, j] += (alpha ^ (order - 2)) * count
                if i != j
                    sketch.similarity_matrix[j, i] += (alpha ^ (order - 2)) * count
                end
            end
        end
    elseif order == 2
        embeddings = edgesketch_core(A, sketch_dimensions, h)
        sketch = Sketch(embeddings, zeros(col_count, col_count), get_calculated_hashes())

        for i in 1:col_count
        
            for j in i:col_count
                matching_count = 0
                for k in 1:sketch_dimensions
                    if embeddings[k, i] == embeddings[k, j]
                        matching_count += 1
                    end
                end
                count = matching_count / sketch_dimensions
                
                sketch.similarity_matrix[i, j] = count
                if i != j
                    sketch.similarity_matrix[j, i] = count
                end
            end
        end
    end
    return sketch
end

function generate_sample(row, hash_functions, j, neighbours)
    return argmin(i -> (global calculated_hashes += 1; return -log(hash_functions[j](i)) / row[i]), neighbours)
end

function to_sla(A::Matrix{Float64})
    matrix_size = min(size(A, 1), size(A, 2))
    for i in 1:matrix_size
        A[i,i] = 1
    end

    return A
end

end