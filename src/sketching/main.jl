using MAT
using Random

include("NodeSketch.jl")
include("SLA.jl")

file = matopen("../data/dblp.mat")
varnames = keys(file)

hello = read(file)

for v in varnames
    println(v)
end

println(typeof(hello))
println(typeof(hello["network"]))
#println(hello["group"])

for y in 1:50
    for x in 1:50
        print("$(hello["network"][x, y]) ")
    end
    println()
end

close(file)

submatrix = hello["network"][1:1000, 1:1000]
println(typeof(submatrix))

num_hash_functions = 128

hash_matrix = zeros(size(hello["network"], 1), num_hash_functions)

hash_functions = [x -> (hash((x, seed)) % Int64(1e9)) / 1e9 for seed in rand(Int64, num_hash_functions)]

for i in 1:size(hello["network"], 1)  # For each number from 1 to 100
    hello["network"][i,i] = 1
end

# Fill the matrix with hash values
for i in 1:size(hello["network"], 1)  # For each number from 1 to 100
    for j in 1:num_hash_functions  # For each hash function
        hash_matrix[i, j] = -log(hash_functions[j](i))
        print("$(hash_matrix[i, j]) ")
    end
    println()
end

# Seed for reproducibility (optional)
Random.seed!(123)

# Creating a vector of hash functions


config = Config(num_hash_functions, 0.0002, hash_functions)
dense_matrix = Matrix(submatrix)
println(typeof(dense_matrix))

sketch = node_sketch_precalculated(hello["network"], 4, config, hash_matrix).embeddings
#
for y in 1:100
    for x in 1:16
        print("$(sketch[x, y]) ")
    end
    println()
end