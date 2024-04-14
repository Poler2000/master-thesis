using MAT
using Random

include("NodeSketch.jl")

using .NodeSketch

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

# Seed for reproducibility (optional)
Random.seed!(123)

# Creating a vector of hash functions

alpha = 0.0002
order = 4
sketch_dimensions = 128

sketch = NodeSketch.nodesketch(hello["network"], order, sketch_dimensions, alpha).embeddings
#sketch = NodeSketch.fastexp_nodesketch(hello["network"], order, sketch_dimensions, alpha).embeddings
#
for y in 1:100
    for x in 1:sketch_dimensions
        print("$(sketch[x, y]) ")
    end
    println()
end

embs = sketch'
dense_matrix = Matrix(embs)

matwrite("my_output_2.mat", Dict(
	"embs" => dense_matrix
), version="v4")

