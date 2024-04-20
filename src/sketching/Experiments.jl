module Experiments
    include("NodeSketch.jl")
    include("DataManager.jl")
    include("Logger.jl")

    using .NodeSketch
    using .DataManager
    using .Logger

    const DATA_FOLDER = "../data/"
    const RESULTS_FOLDER = "../results/"

    export run_embeddings_var_order, run_embeddings_var_size, run_embeddings_var_alpha, run_all

    function run_all()
        default_alpha = 0.001
        default_k = 4
        default_size = 128

        run_embeddings_var_order(["dblp", "blogcatalog", "wiki", "Homo_sapiens"], true, [2,3,4], default_alpha, default_size)
        run_embeddings_var_order(["dblp", "blogcatalog", "wiki", "Homo_sapiens"], false, [2,3,4], default_alpha, default_size)

        run_embeddings_var_size(["dblp", "blogcatalog", "wiki", "Homo_sapiens"], true, default_k, default_alpha, [16, 32, 64, 128])
        run_embeddings_var_size(["dblp", "blogcatalog", "wiki", "Homo_sapiens"], false, default_k, default_alpha, [16, 32, 64, 128])

        run_embeddings_var_alpha(["dblp", "blogcatalog", "wiki", "Homo_sapiens"], true, default_k, [0.0005, 0.001, 0.01, 0.1], default_size)
        run_embeddings_var_alpha(["dblp", "blogcatalog", "wiki", "Homo_sapiens"], false, default_k, [0.0005, 0.001, 0.01, 0.1], default_size)
    end

    function run_embeddings_var_order(datasets::Array{String}, use_expsketch::Bool, orders::Vector{<:Integer}, alpha::Number, sketch_dimensions::Number)
        for dataset in datasets
            matrix = load_matrix_mat("$DATA_FOLDER$dataset.mat")
            for k in orders
                log_info("Executing $(use_expsketch ? "expsketch-based" : "basic") algorithm for dataset: $dataset")
                log_info("k = $k, alpha = $alpha, L = $sketch_dimensions")
                @time sketch = use_expsketch ? fastexp_nodesketch(matrix, k, sketch_dimensions, alpha) : nodesketch(matrix, k, sketch_dimensions, alpha)
                
                embs = sketch.embeddings'

                dense_matrix = Matrix(embs)
                matrix_str = join([join(row, "\t") for row in eachrow(dense_matrix[1:8, 1:16]')], "\n")
                log_info("Output sample:\n" * matrix_str * '\n')

                alpha_str = replace(string(alpha), "0." => "")
                save_matrix_mat(dense_matrix, "$RESULTS_FOLDER/var_k/$(dataset)/res_$(dataset)_$(use_expsketch ? "exp" : "basic")_$(k)_$(alpha_str)_$(sketch_dimensions).mat")
            end
        end
    end

    function run_embeddings_var_size(datasets::Array{String}, use_expsketch::Bool, k::Number, alpha::Number, sketch_sizes::Vector{<:Integer})
        for dataset in datasets
            matrix = load_matrix_mat("$DATA_FOLDER$dataset.mat")
            for sketch_size in sketch_sizes
                log_info("Executing $(use_expsketch ? "expsketch-based" : "basic") algorithm for dataset: $dataset")
                log_info("k = $k, alpha = $alpha, L = $sketch_size")
                @time sketch = use_expsketch ? fastexp_nodesketch(matrix, k, sketch_size, alpha) : nodesketch(matrix, k, sketch_size, alpha)
                
                embs = sketch.embeddings'

                dense_matrix = Matrix(embs)
                matrix_str = join([join(row, "\t") for row in eachrow(dense_matrix[1:8, 1:16]')], "\n")
                log_info("Output sample:\n" * matrix_str * '\n')

                alpha_str = replace(string(alpha), "0." => "")
                save_matrix_mat(dense_matrix, "$RESULTS_FOLDER/var_L/$(dataset)/res_$(dataset)_$(use_expsketch ? "exp" : "basic")_$(k)_$(alpha_str)_$(sketch_size).mat")
            end
        end
    end

    function run_embeddings_var_alpha(datasets::Array{String}, use_expsketch::Bool, k::Number, alphas::Vector{<:Number}, sketch_dimensions::Number)
        for dataset in datasets
            matrix = load_matrix_mat("$DATA_FOLDER$dataset.mat")
            for alpha in alphas
                log_info("Executing $(use_expsketch ? "expsketch-based" : "basic") algorithm for dataset: $dataset")
                log_info("k = $k, alpha = $alpha, L = $sketch_dimensions")
                @time sketch = use_expsketch ? fastexp_nodesketch(matrix, k, sketch_dimensions, alpha) : nodesketch(matrix, k, sketch_dimensions, alpha)
                
                embs = sketch.embeddings'

                dense_matrix = Matrix(embs)
                matrix_str = join([join(row, "\t") for row in eachrow(dense_matrix[1:8, 1:16]')], "\n")
                log_info("Output sample:\n" * matrix_str * '\n')

                alpha_str = replace(string(alpha), "0." => "")
                save_matrix_mat(dense_matrix, "$RESULTS_FOLDER/var_a/$(dataset)/res_$(dataset)_$(use_expsketch ? "exp" : "basic")_$(k)_$(alpha_str)_$(sketch_dimensions).mat")
            end
        end
    end
end