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
export nodesketch, fastexp_nodesketch, nodesketch_extended, fastexp_nodesketch_extended

struct Sketch
    embeddings::Matrix{Float64}
    similarity_matrix::Matrix{Float64}
end

function generate_sample(row, hash_matrix, j, neighbours)
    return argmin(i -> (hash_matrix[i, j] / row[i]), neighbours)
end

function nodesketch(
    A::SparseMatrixCSC{<:Number, <:Integer}, 
    order::Integer, 
    sketch_dimensions::Integer, 
    alpha::Number)::Sketch
    
    # Although it's more intuitive to iterate row-wise, 
    # column-wise processing is slightly more efficient due to sparse matrix implementation
    sla = to_sla(sparse(A'))
    hash_matrix = construct_hash_matrix(size(sla, 1), sketch_dimensions)
    col_count = size(A, 2)

    embeddings = nodesketch_core(sla, order, sketch_dimensions, alpha, hash_matrix)
    return Sketch(embeddings, zeros(col_count, col_count))
end

function fastexp_nodesketch(
    A::SparseMatrixCSC{<:Number, <:Integer}, 
    order::Integer, 
    sketch_dimensions::Integer, 
    alpha::Number)::Sketch
    
    # Although it's more intuitive to iterate row-wise, 
    # column-wise processing is slightly more efficient due to sparse matrix implementation
    sla = to_sla(sparse(A'))
    h = x -> (hash((x, rand())) % Int64(1e9)) / 1e9
    col_count = size(A, 2)

    embeddings = fastexp_nodesketch_core(sla, order, sketch_dimensions, alpha, h)
    return Sketch(embeddings, zeros(col_count, col_count))
end

function nodesketch_extended(
    A::SparseMatrixCSC{<:Number, <:Integer}, 
    order::Integer, 
    sketch_dimensions::Integer, 
    alpha::Number,
    alpha_sim::Number)::Sketch
    
    # Although it's more intuitive to iterate row-wise, 
    # column-wise processing is slightly more efficient due to sparse matrix implementation
    sla = to_sla(sparse(A'))
    hash_matrix = construct_hash_matrix(size(sla, 1), sketch_dimensions)
    col_count = size(sla, 2)

    embeddings = nodesketch_core(sla, order, sketch_dimensions, alpha_sim, hash_matrix)
    similarity_matrix = zeros(col_count, col_count)

    for i in 1:col_count
        if (i % 100 == 0)
            log_debug("Computing similarity matrix, nodes: $i, order: $order")
        end
        
        for j in i:col_count
            count = sum(embeddings[:, i] .== embeddings[:, j]) / sketch_dimensions
            similarity_matrix[i, j] = count
            if i != j
                similarity_matrix[j, i] = count
            end
        end
    end

    return Sketch(embeddings, similarity_matrix)
end

function fastexp_nodesketch_extended(
    A::SparseMatrixCSC{<:Number, <:Integer}, 
    order::Integer, 
    sketch_dimensions::Integer, 
    alpha_sim::Number)::Sketch
    
    # Although it's more intuitive to iterate row-wise, 
    # column-wise processing is slightly more efficient due to sparse matrix implementation
    sla = to_sla(sparse(A'))
    h = x -> (hash(x) % Int64(1e9)) / 1e9

    return fastexp_nodesketch_core_similarity(sla, order, sketch_dimensions, alpha_sim, h)
end

function nodesketch_core(
    A::SparseMatrixCSC{<:Number, <:Integer}, 
    order::Integer, 
    sketch_dimensions::Integer, 
    alpha::Number,
    hash_matrix::AbstractMatrix{<:Number})::Matrix{Float64}

    col_count = size(A, 2)
    embeddings = zeros(sketch_dimensions, col_count)

    if order > 2
        embeddings = nodesketch_core(A, order - 1, sketch_dimensions, alpha, hash_matrix)
        for (i, col) in enumerate(eachcol(A))
            if (i % 100 == 0)
                log_debug("$i, $order")
            end

            v = Vector{Float64}(undef,col_count)
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

            embeddings[:, i] = sketch_single_node(v, hash_matrix, sketch_dimensions)
        end

    elseif order == 2
        for (i, col) in enumerate(eachcol(A))
            if (i % 100 == 0)
                log_debug("$i, $order")
            end

            embeddings[:, i] = sketch_single_node(col, hash_matrix, sketch_dimensions)
        end
    end

    return embeddings
end

function fastexp_nodesketch_core(
    A::SparseMatrixCSC{<:Number, <:Integer}, 
    order::Integer, 
    sketch_dimensions::Integer, 
    alpha::Number,
    h::Function)::Matrix{Float64}

    col_count = size(A, 2)
    embeddings = zeros(sketch_dimensions, col_count)

    if order > 2
        embeddings = fastexp_nodesketch_core(A, order - 1, sketch_dimensions, alpha, h)
        for (i, col) in enumerate(eachcol(A))
            if (i % 100 == 0)
                log_debug("$i, $order")
            end

            neighbours = findall(x -> x != 0, col)            
            v = zeros(sketch_dimensions)
            for n in neighbours
                for j in 1:sketch_dimensions
                    v[j] = min(embeddings[j, i], embeddings[j, n])
                end
            end

            embeddings[:, i] .+= (v .* alpha / sketch_dimensions)
        end
    elseif order == 2
        for (i, col) in enumerate(eachcol(A))
            if (i % 100 == 0)
                log_debug("$i, $order")
            end

            neighbours = [StreamElement(index, weight) 
                for (index, weight) in enumerate(col) if weight != 0]

            embeddings[:, i] = fast_expsketch(
                neighbours, 
                sketch_dimensions, 
                h)
        end
    end
    return embeddings
end

function fastexp_nodesketch_core_similarity(
    A::SparseMatrixCSC{<:Number, <:Integer}, 
    order::Integer, 
    sketch_dimensions::Integer, 
    alpha_sim::Number,
    h::Function)::Sketch

    col_count = size(A, 2)
    sketch = Sketch(zeros(sketch_dimensions, col_count), zeros(col_count, col_count))
    if order > 2
        sketch = fastexp_nodesketch_core_similarity(A, order - 1, sketch_dimensions, alpha_sim, h)
        #ak = A^(order - 1)
        #for (i, col) in enumerate(eachcol(ak))
        #    if (i % 100 == 0)
        #        log_debug("$i, $order")
        #    end
#
        #    neighbours = [StreamElement(index, weight) 
        #        for (index, weight) in enumerate(col) if weight != 0]
#
        #    sketch.embeddings[:, i] = fast_expsketch(
        #        neighbours, 
        #        sketch_dimensions, 
        #        h)
        #end
#
        #@threads for i in 1:col_count
        #    if (i % 100 == 0)
        #        log_debug("Computing similarity matrix, nodes: $i, order: $order")
        #    end
        #    
        #    for j in i:col_count
        #        count = sum(sketch.embeddings[:, i] .== sketch.embeddings[:, j]) / sketch_dimensions
        #        sketch.similarity_matrix[i, j] += (alpha_sim ^ (order - 2)) * count
        #        if i != j
        #            sketch.similarity_matrix[j, i] += (alpha_sim ^ (order - 2)) * count
        #        end
        #    end
        #end

        ak = A^(order - 1)
        hey = ConcurrentDict{Int, Vector{Float64}}()
        
        @threads for i in 1:col_count
            k_neighbours = findall(x -> x > 0, ak[:, i])
            v = fill(typemax(Float64), sketch_dimensions)
            for n in k_neighbours
                for j in 1:sketch_dimensions
                    v[j] = min(v[j], sketch.embeddings[j, n])
                end
            end
            hey[i] = v
        end

        @threads for i in 1:col_count
            if (i % 100 == 0)
                log_debug("Computing similarity matrix, nodes: $i, order: $order")
            end

            for j in i:col_count
                count = sum(hey[i] .== hey[j]) / sketch_dimensions
                sketch.similarity_matrix[i, j] += (alpha_sim ^ (order - 2)) * count
                if i != j
                    sketch.similarity_matrix[j, i] += (alpha_sim ^ (order - 2)) * count
                end
            end
        end
    elseif order == 2
        for (i, col) in enumerate(eachcol(A))
            if (i % 100 == 0)
                log_debug("$i, $order")
            end

            neighbours = [StreamElement(index, weight) 
                for (index, weight) in enumerate(col) if weight != 0]

            sketch.embeddings[:, i] = expsketch(
                neighbours, 
                sketch_dimensions, 
                h)
        end

        @threads for i in 1:col_count
            if (i % 100 == 0)
                log_debug("Computing similarity matrix, nodes: $i, order: $order")
            end
            
            for j in i:col_count
                count = sum(sketch.embeddings[:, i] .== sketch.embeddings[:, j]) / sketch_dimensions
                sketch.similarity_matrix[i, j] = count
                if i != j
                    sketch.similarity_matrix[j, i] = count
                end
            end
        end
    end
    return sketch
end

function sketch_single_node(
    col::AbstractArray{<:Number}, 
    hash_matrix::AbstractMatrix{<:Number}, 
    sketch_dimensions::Integer)::Array{<:Number}

    neighbours = findall(x -> x != 0, col)
    sketch = Float64[generate_sample(col, hash_matrix, j, neighbours) for j in 1:sketch_dimensions]

    return sketch
end

function to_sla(A::AbstractMatrix)
    matrix_size = min(size(A, 1), size(A, 2))
    for i in 1:matrix_size
        A[i,i] = 1
    end

    return A
end

function construct_hash_matrix(width, height)
    hash_matrix = zeros(width, height)

    hash_functions = [x -> (hash((x, seed)) % Int64(1e9)) / 1e9 for seed in rand(Int64, height)]

    for x in 1:width  
        for y in 1:height 
            hash_matrix[x, y] = -log(hash_functions[y](x))
        end
    end

    return hash_matrix
end

end