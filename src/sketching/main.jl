using Random

include("Experiments.jl")

using .Experiments

run_all()
#run_embeddings_var_size(["dblp", "blogcatalog", "wiki", "Homo_sapiens"], true, 3, 0.001, [16, 32, 64, 128])
#run_embeddings_var_order(["wiki"], true, [2,3,4], 0.001, 128)
#run_embeddings_var_alpha(["dblp"], true, 4, [0.0005, 0.001, 0.01, 0.1], 128)



