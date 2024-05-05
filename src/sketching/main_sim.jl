using Random

include("ExperimentsSimilarity.jl")

using .Experiments

default_alpha = 0.5
default_alpha_sim = 0.5
default_k = 4
default_size = 128

datasets = [#="bin_2500_01", "bin_5000_005", "bin_5000_001", "bin_5000_0005",=# "dblp"]

run_similarity_var_k(datasets, [2, 3, 4], default_alpha, default_alpha_sim, default_size)
#run_similarity_var_L(datasets, 2, default_alpha, default_alpha_sim, [8, 32, 128])
#run_similarity_var_alpha(datasets, 3, default_alpha, [0.25, 0.5, 1], 128)