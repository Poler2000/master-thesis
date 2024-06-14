include("DataManager.jl")

using Random
using StatsBase

using .DataManager

const DATA_FOLDER = "../data"

export generate_erdos_renyi_matrix, generate_stochastic_block_matrix, generate_barabasi_albert_matrix, generate_all

function generate_erdos_renyi_matrix(n::Int, density::Float64, max_value::Int)
    matrix = zeros(Float64, n, n)

    for y in 1:n
        for x in (y + 1):n
            if rand() < density
                value = rand(1:max_value)
                matrix[x,y] = value
                matrix[y,x] = value
            end
        end
    end

    return matrix
end

function generate_stochastic_block_matrix(n::Int, block_count::Int, p::Float64, q::Float64)
    matrix = zeros(n, n)
    block_dict = Dict()

    for i in 1:n
        block = rand(1:block_count)
        block_dict[i] = block
    end

    for y in 1:n
        for x in (y + 1):n
            if block_dict[x] == block_dict[y]
                if rand() < p
                    matrix[x,y] = 1
                    matrix[y,x] = 1
                end
            elseif rand() < q
                matrix[x,y] = 1
                matrix[y,x] = 1
            end
        end
    end

    return matrix
end

function generate_barabasi_albert_matrix(n::Int, m::Int, m0::Int)
    matrix = zeros(n, n)

    ys = shuffle(collect(1:n))
    orders = Dict(i => 0 for i in 1:n)
    all_orders = 0

    for y in ys[1:m0]
        x = y
        
        while (x == y)
            x = ys[rand(1:m0)]
        end

        matrix[x,y] = 1
        matrix[y,x] = 1

        orders[x] += 1
        orders[y] += 1
        all_orders += 2
    end

    for i in (m0+1):n
        y = ys[i]

        for j in 1:m
            x = y
        
            while (x == y || matrix[x,y] > 0)
                weights = [orders[l] / all_orders for l in ys[1:(i-1)]]
                x = sample(ys[1:(i-1)], ProbabilityWeights(weights))
            end

            matrix[x,y] = 1
            matrix[y,x] = 1
    
            orders[x] += 1
            orders[y] += 1
            all_orders += 2
        end
    end

    return matrix
end

function generate_all()
    ns = [1000, 2500, 5000, 10000]
    densities = [0.01, 0.005, 0.001, 0.0005]
    ms = [2,8,16]
    block_counts = [2,4,8]    
    
    for n in ns
        for m in ms
            println("Barabasi-Albert: $n, $m")
            matrix = generate_barabasi_albert_matrix(n, m, m)

            for y in 1:10
                for x in 1:10
                    print("$(matrix[x,y]) ")
                end
                println()
            end
            println()

            edges = count(!iszero, matrix) / 2
            println("edges: $edges")
            save_matrix_network(matrix, "$DATA_FOLDER/baralb_$(n)_$(m).mat")
        end
    end

    for n in ns
        for block_count in block_counts
            println("Stochastic block: $n, $block_count")
            matrix = generate_stochastic_block_matrix(n, block_count, 0.5, 0.001)

            for y in 1:10
                for x in 1:10
                    print("$(matrix[x,y]) ")
                end
                println()
            end
            println()

            edges = count(!iszero, matrix) / 2
            println("edges: $edges")
            save_matrix_network(matrix, "$DATA_FOLDER/block_$(n)_$(block_count).mat")
        end
    end

    for n in ns
        for density in densities
            println("Erdos-Renyi: $n, $density")
            matrix_binary = generate_erdos_renyi_matrix(n, density, 1)
            matrix_weighted = generate_erdos_renyi_matrix(n, density, 100)

            density_str = replace(string(density), "0." => "")
            for y in 1:10
                for x in 1:10
                    print("$(matrix_binary[x,y]) ")
                end
                println()
            end
            println()
            for y in 1:10
                for x in 1:10
                    print("$(matrix_weighted[x,y]) ")
                end
                println()
            end
            save_matrix_network(matrix_binary, "$DATA_FOLDER/bin_$(n)_$(density_str).mat")
            save_matrix_network(matrix_weighted, "$DATA_FOLDER/weighted_$(n)_$(density_str).mat")
        end
    end
end


