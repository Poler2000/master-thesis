include("DataManager.jl")

using SparseArrays
using Random

using .DataManager

const DATA_FOLDER = "../data"

function generate_random_matrix(n::Int, density::Float64, max_value::Int)
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

ns = [2500, 5000, 10000]
densities = [0.0025, 0.001, 0.0005, 0.0001]

for n in ns
    for density in densities
        println("$n, $density")
        matrix_binary = generate_random_matrix(n, density, 1)
        matrix_weighted = generate_random_matrix(n, density, 100)

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