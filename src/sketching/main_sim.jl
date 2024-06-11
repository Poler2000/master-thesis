using Random

include("ExperimentsSimilarity.jl")

using .Experiments

datasets = ["bin_1000_0005"]
run_all_similarity(datasets)

#run_tests_all_datasets()