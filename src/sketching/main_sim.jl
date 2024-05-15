using Random

include("ExperimentsSimilarity.jl")

using .Experiments

default_alpha = 0.3
default_alpha_sim = 0.7
default_k = 4
default_size = 24

#datasets = ["bin_10000_01", "bin_10000_005", "bin_10000_001", "bin_10000_0005", "dblp", "blogcatalog", "wiki"]
#datasets = ["bin_10000_01", "bin_10000_005", "bin_10000_001", "bin_10000_0005"]
datasets = ["block_1000_4", "block_1000_8", "block_2500_4"]
#datasets = ["dblp", "blogcatalog", "wiki", "Homo_sapiens"]

#run_similarity_var_k(datasets, [2,3,4], default_alpha, default_alpha_sim, default_size)
run_similarity_var_k(datasets, [2,3,4], 0.3, 0.3, default_size)
#run_similarity_var_k(datasets, [2,3,4], 0.7, 0.7, default_size)
#run_similarity_var_L(datasets, 2, default_alpha, default_alpha_sim, [8, 16, 32, 64, 128])
#run_similarity_var_alpha(datasets, 4, default_alpha, [0.0, 0.25, 0.5, 1], 64)