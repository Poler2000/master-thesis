include("DataManager.jl")

using SparseArrays
using Random

using .DataManager

const DATA_FOLDER = "../data"

function generate_erods_renyi_matrix(n::Int, density::Float64, max_value::Int)
    matrix = zeros(n, n)

    for y in 1:n
        for x in (y + 1):n
            if rand() < density
                value = rand(1:max_value)
                matrix[x,y] = value
                matrix[y,x] = value
            end
        end
    end

    return sparse(matrix)
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

    return sparse(matrix)
end

ns = [1000, 2500, 5000, 10000]
densities = [0.0025, 0.001, 0.0005, 0.0001]
block_counts = [2,4,8]

for n in ns
    for block_count in block_counts
        println("stochastic block: $n, $block_count")
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

#for n in ns
#    for density in densities
#        println("erods renyi: $n, $density")
#        matrix_binary = generate_erods_renyi_matrix(n, density, 1)
#        matrix_weighted = generate_erods_renyi_matrix(n, density, 100)
#
#        density_str = replace(string(density), "0." => "")
#        for y in 1:10
#            for x in 1:10
#                print("$(matrix_binary[x,y]) ")
#            end
#            println()
#        end
#        println()
#        for y in 1:10
#            for x in 1:10
#                print("$(matrix_weighted[x,y]) ")
#            end
#            println()
#        end
#        save_matrix_network(matrix_binary, "$DATA_FOLDER/bin_$(n)_$(density_str).mat")
#        save_matrix_network(matrix_weighted, "$DATA_FOLDER/weighted_$(n)_$(density_str).mat")
#    end
#end