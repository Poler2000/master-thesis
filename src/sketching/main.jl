using Random

include("NodeSketch.jl")
include("DataManager.jl")

using .NodeSketch
using .DataManager


dataset = "dblp"
matrix = DataManager.load_matrix("../data/$dataset.mat")

alpha = 0.001
order = 4
sketch_dimensions = 128

#sketch = NodeSketch.nodesketch(matrix, order, sketch_dimensions, alpha).embeddings
sketch = NodeSketch.fastexp_nodesketch(matrix, order, sketch_dimensions, alpha).embeddings

for y in 1:100
    for x in 1:8
        print("$(sketch[x, y]) ")
    end
    println()
end

embs = sketch'
dense_matrix = Matrix(embs)

DataManager.save_matrix(dense_matrix, "results_$(dataset)_$order.mat")

