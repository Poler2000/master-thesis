module NodeSketch

include("ExpSketch.jl")

using DataStructures
using SparseArrays
using .ExpSketch

export SLA, Sketch
export nodesketch, fastexp_nodesketch

struct Sketch
    embeddings::Matrix{Float64}
end

#function calculate_single_sample(hash::Function, element::Number, i::Number)
#    return -log(hash(i)) / element
#end

#function generate_sample(row, config::Config, j)
#    return argmin(i -> calculate_single_sample(config.hashes[j], row[i], i), 1:length(row))
#end

#function generate_sample_precalculated(row, hash_matrix, j)
#    return argmin(i -> (hash_matrix[i, j] / row[i]), 1:length(row))
#end

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

    return nodesketch_core(sla, order, sketch_dimensions, alpha, hash_matrix)
end

function fastexp_nodesketch(
    A::SparseMatrixCSC{<:Number, <:Integer}, 
    order::Integer, 
    sketch_dimensions::Integer, 
    alpha::Number)::Sketch
    
    # Although it's more intuitive to iterate row-wise, 
    # column-wise processing is slightly more efficient due to sparse matrix implementation
    println("hello")
    sla = to_sla(sparse(A'))
    h = x -> (hash((x, 1)) % Int64(1e9)) / 1e9

    return fastexp_nodesketch_core(sla, order, sketch_dimensions, alpha, h)
end

function nodesketch_core(
    A::SparseMatrixCSC{<:Number, <:Integer}, 
    order::Integer, 
    sketch_dimensions::Integer, 
    alpha::Number,
    hash_matrix::AbstractMatrix{<:Number})::Sketch

    col_count = size(A, 2)
    sketch = Sketch(zeros(sketch_dimensions, col_count))

    if order > 2
        sketch = nodesketch_core(A, order - 1, sketch_dimensions, alpha, hash_matrix)
        for (i, col) in enumerate(eachcol(A))
            println(i)
            v = Vector{Float64}(undef,col_count)
            neighbours = findall(x -> x != 0, col)

            mydict = Dict(i => 0 for i in 0:col_count)

            for n in neighbours
                for j in 1:sketch_dimensions
                    mydict[sketch.embeddings[j, n]] += 1
                end
            end

            for l in 1:col_count
                s = mydict[l]
                s *= alpha / sketch_dimensions
                v[l] = s + col[l]
            end
            sketch.embeddings[:, i] = sketch_single_node(v, hash_matrix, sketch_dimensions)
        end
    elseif order == 2
        for (i, col) in enumerate(eachcol(A))
            println("$i, $order")

            sketch.embeddings[:, i] = sketch_single_node(col, hash_matrix, sketch_dimensions)
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

function fastexp_nodesketch_core(
    A::SparseMatrixCSC{<:Number, <:Integer}, 
    order::Integer, 
    sketch_dimensions::Integer, 
    alpha::Number,
    h::Function)::Sketch

    col_count = size(A, 2)
    sketch = Sketch(zeros(sketch_dimensions, col_count))
    println("hello")

    if order > 2
        sketch = fastexp_nodesketch_core(A, order - 1, sketch_dimensions, alpha, hash)
        for (i, col) in enumerate(eachcol(A))
            println(i)
            v = Vector{Float64}(undef,col_count)
            neighbours = findall(x -> x != 0, col)

            mydict = Dict(i => 0 for i in 0:col_count)

            for n in neighbours
                for j in 1:sketch_dimensions
                    mydict[sketch.embeddings[j, n]] += 1
                end
            end

            for l in 1:col_count
                s = mydict[l]
                s *= alpha / sketch_dimensions
                v[l] = s + col[l]
            end

            neighbours = [ExpSketch.StreamElement(index, weight) 
            for (index, weight) in enumerate(v) if weight != 0]

            sketch.embeddings[:, i] = ExpSketch.fast_expsketch(
                neighbours, 
                sketch_dimensions, 
                hash)
        end
    elseif order == 2
        for (i, col) in enumerate(eachcol(A))
            println("$i, $order")

            neighbours = [ExpSketch.StreamElement(index, weight) 
                for (index, weight) in enumerate(col) if weight != 0]

            sketch.embeddings[:, i] = ExpSketch.expsketch(
                neighbours, 
                sketch_dimensions,
                h)
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
            print("$(hash_matrix[x, y]) ")
        end
        println()
    end

    return hash_matrix
end

end