module Experiments
    include("NodeSketch.jl")
    include("DataManager.jl")
    include("Logger.jl")

    using .NodeSketch
    using .DataManager
    using .Logger

    const DATA_FOLDER = "../data/"
    const RESULTS_FOLDER = "../results/sim"

    export run_all_similarity, run_similarity_var_k, run_similarity_var_L, run_similarity_var_alpha

    struct SimResult
        dataset::String
        algorithm::String
        k::Integer
        L::Integer
        alpha::Number
        alpha_sim::Number
        sim_top_1000::Number
        sim_all::Number
    end

    function run_all_similarity()
        default_alpha = 0.002
        default_alpha_sim = 0.5
        default_size = 128
        
        datasets = ["dblp", "blogcatalog", "bin_5000_01", "bin_5000_005", "bin_5000_001", "bin_5000_0005"]

        run_similarity_var_k(datasets, [2, 3, 4], default_alpha, default_alpha_sim, default_size)
        run_similarity_var_L(datasets, 2, default_alpha, default_alpha_sim, [8, 32, 128])
        run_similarity_var_alpha(datasets, 4, default_alpha, [0.25, 0.5, 1], default_size)
    end

    function run_similarity_var_k(datasets::Array{String}, orders::Vector{<:Integer}, 
        alpha::Number, alpha_sim::Number, sketch_dimensions::Number)

        res_exp = run_similarity_var_k_single(datasets, true, orders, alpha, alpha_sim, sketch_dimensions)
        res_node = run_similarity_var_k_single(datasets, false, orders, alpha, alpha_sim, sketch_dimensions)

        print_summary(datasets, orders, "k", res_exp, res_node)
    end

    function run_similarity_var_L(datasets::Array{String}, order::Integer, 
        alpha::Number, alpha_sim::Number, dimensions::Vector{<:Integer})

        res_exp = run_similarity_var_L_single(datasets, true, order, alpha, alpha_sim, dimensions)
        res_node = run_similarity_var_L_single(datasets, false, order, alpha, alpha_sim, dimensions)

        print_summary(datasets, dimensions, "L", res_exp, res_node)
    end

    function run_similarity_var_alpha(datasets::Array{String}, order::Integer, 
        alpha::Number, alphas_sim::Vector{<:Number}, sketch_dimensions::Number)

        res_exp = run_similarity_var_alpha_single(datasets, true, order, alpha, alphas_sim, sketch_dimensions)
        res_node = run_similarity_var_alpha_single(datasets, false, order, alpha, alphas_sim, sketch_dimensions)

        print_summary(datasets, alphas_sim, "a", res_exp, res_node)
    end

    function print_summary(datasets::Array{String}, variables::Array{<:Number}, label::String, res_exp, res_node)
        i = 0
        path = "$RESULTS_FOLDER/summary_$(label).txt"
        for dataset in datasets
            log_info("Summary for dataset: $dataset", path)
            log_info("\t|\t\t\t NodeSketch \t\t\t\t|\t\t\tNodeExpSketch", path)
            log_info("$label\t|\tprecision - top 1000 \t|\tprecision - total\t|\tprecision - top 1000 \t|\tprecision - total\t", path)
            
            for variable in variables
                i += 1
                log_info("$variable\t|\t\t$(res_node[i].sim_top_1000) \t\t|\t$(res_node[i].sim_all)\t|\t\t$(res_exp[i].sim_top_1000) \t\t|\t$(res_exp[i].sim_all)\t", path)
            end
        end
    end

    function run_similarity_var_k_single(datasets::Array{String}, use_expsketch::Bool, 
        orders::Vector{<:Integer}, alpha::Number, alpha_sim::Number, sketch_dimensions::Number)::Vector{SimResult}
        results = Vector{SimResult}()
        for dataset in datasets
            matrix = load_matrix_mat("$DATA_FOLDER$dataset.mat")
            sample_size = 0
            n = size(matrix, 1)
            for i in 1:n-1
                for j in i+1:n
                    if matrix[i, j] != 0
                        sample_size += 1
                    end
                end
            end

            for k in orders
                log_info("Executing $(use_expsketch ? "expsketch-based" : "basic") algorithm for dataset: $dataset")
                log_info("k = $k, alpha_sim = $alpha_sim, L = $sketch_dimensions")
                @time sketch = use_expsketch ? fastexp_nodesketch_extended(matrix, k, sketch_dimensions, alpha_sim) : nodesketch_extended(matrix, k, sketch_dimensions, alpha, alpha_sim)
                
                embs = sketch.embeddings'
                similarity = sketch.similarity_matrix'
            
                dense_matrix = Matrix(embs)
                matrix_str = join([join(row, "\t") for row in eachrow(dense_matrix[1:8, 1:min(16, sketch_dimensions)]')], "\n")
                log_info("Output sample:\n" * matrix_str * '\n')

                mask = trues(size(similarity))
                for i in 1:min(size(similarity)...)
                    mask[i, i] = false
                    for j in i:min(size(similarity)...)
                        mask[j,i] = false
                    end
                end
            
                flat_A = similarity[mask]

                sorted_indices = sortperm(flat_A, rev=true)[1:sample_size]
            
                original_indices = findall(mask)
                top_coordinates = original_indices[sorted_indices]
            
                correctly_predicted = 0
                correctly_predicted_1000 = 0
    
                for (i, (x, y)) in enumerate(Tuple.(top_coordinates))
                    if matrix[x, y] >= 1
                        correctly_predicted += 1
                        if i <= 1000
                            correctly_predicted_1000 += 1
                        end
                    end

                end

                push!(results, SimResult(dataset, use_expsketch ? "NodeExpSketch" : "NodeSketch", k, sketch_dimensions, alpha, alpha_sim, (correctly_predicted_1000 / 1000), (correctly_predicted / sample_size)))
            
                log_info("correctly predicted: $correctly_predicted / $sample_size ($(correctly_predicted / sample_size))")
                log_info("correctly predicted per 1000: $correctly_predicted_1000 / 1000 ($(correctly_predicted_1000 / 1000))")
            end
        end

        return results
    end

    function run_similarity_var_L_single(datasets::Array{String}, use_expsketch::Bool, 
        k::Integer, alpha::Number, alpha_sim::Number, dimensions::Vector{<:Integer})::Vector{SimResult}
        results = Vector{SimResult}()
        for dataset in datasets
            matrix = load_matrix_mat("$DATA_FOLDER$dataset.mat")
            sample_size = 0
            n = size(matrix, 1)
            for i in 1:n-1
                for j in i+1:n
                    if matrix[i, j] != 0
                        sample_size += 1
                    end
                end
            end

            for sketch_dimensions in dimensions
                log_info("Executing $(use_expsketch ? "expsketch-based" : "basic") algorithm for dataset: $dataset")
                log_info("k = $k, alpha_sim = $alpha_sim, L = $sketch_dimensions")
                @time sketch = use_expsketch ? fastexp_nodesketch_extended(matrix, k, sketch_dimensions, alpha_sim) : nodesketch_extended(matrix, k, sketch_dimensions, alpha, alpha_sim)
                
                embs = sketch.embeddings'
                similarity = sketch.similarity_matrix'
            
                dense_matrix = Matrix(embs)
                matrix_str = join([join(row, "\t") for row in eachrow(dense_matrix[1:8, 1:min(16, sketch_dimensions)]')], "\n")
                log_info("Output sample:\n" * matrix_str * '\n')

                mask = trues(size(similarity))
                for i in 1:min(size(similarity)...)
                    mask[i, i] = false
                    for j in i:min(size(similarity)...)
                        mask[j,i] = false
                    end
                end
            
                flat_A = similarity[mask]

                sorted_indices = sortperm(flat_A, rev=true)[1:sample_size]
            
                original_indices = findall(mask)
                top_coordinates = original_indices[sorted_indices]
            
                correctly_predicted = 0
                correctly_predicted_1000 = 0
    
                for (i, (x, y)) in enumerate(Tuple.(top_coordinates))
                    if matrix[x, y] >= 1
                        correctly_predicted += 1
                        if i <= 1000
                            correctly_predicted_1000 += 1
                        end
                    end

                end

                push!(results, SimResult(dataset, use_expsketch ? "NodeExpSketch" : "NodeSketch", k, sketch_dimensions, alpha, alpha_sim, (correctly_predicted_1000 / 1000), (correctly_predicted / sample_size)))
            
                log_info("correctly predicted: $correctly_predicted / $sample_size ($(correctly_predicted / sample_size))")
                log_info("correctly predicted per 1000: $correctly_predicted_1000 / 1000 ($(correctly_predicted_1000 / 1000))")
            end
        end

        return results
    end

    function run_similarity_var_alpha_single(datasets::Array{String}, use_expsketch::Bool, 
        k::Integer, alpha::Number, alphas_sim::Vector{<:Number}, sketch_dimensions::Number)::Vector{SimResult}
        results = Vector{SimResult}()
        for dataset in datasets
            matrix = load_matrix_mat("$DATA_FOLDER$dataset.mat")
            sample_size = 0
            n = size(matrix, 1)
            for i in 1:n-1
                for j in i+1:n
                    if matrix[i, j] != 0
                        sample_size += 1
                    end
                end
            end

            for alpha_sim in alphas_sim
                log_info("Executing $(use_expsketch ? "expsketch-based" : "basic") algorithm for dataset: $dataset")
                log_info("k = $k, alpha_sim = $alpha_sim, L = $sketch_dimensions")
                @time sketch = use_expsketch ? fastexp_nodesketch_extended(matrix, k, sketch_dimensions, alpha_sim) : nodesketch_extended(matrix, k, sketch_dimensions, alpha_sim, alpha_sim)
                
                embs = sketch.embeddings'
                similarity = sketch.similarity_matrix'
            
                dense_matrix = Matrix(embs)
                matrix_str = join([join(row, "\t") for row in eachrow(dense_matrix[1:8, 1:min(16, sketch_dimensions)]')], "\n")
                log_info("Output sample:\n" * matrix_str * '\n')

                mask = trues(size(similarity))
                for i in 1:min(size(similarity)...)
                    mask[i, i] = false
                    for j in i:min(size(similarity)...)
                        mask[j,i] = false
                    end
                end
            
                flat_A = similarity[mask]

                sorted_indices = sortperm(flat_A, rev=true)[1:sample_size]
            
                original_indices = findall(mask)
                top_coordinates = original_indices[sorted_indices]
            
                correctly_predicted = 0
                correctly_predicted_1000 = 0
    
                for (i, (x, y)) in enumerate(Tuple.(top_coordinates))
                    if matrix[x, y] >= 1
                        correctly_predicted += 1
                        if i <= 1000
                            correctly_predicted_1000 += 1
                        end
                    end

                end

                push!(results, SimResult(dataset, use_expsketch ? "NodeExpSketch" : "NodeSketch", k, sketch_dimensions, alpha, alpha_sim, (correctly_predicted_1000 / 1000), (correctly_predicted / sample_size)))
            
                log_info("correctly predicted: $correctly_predicted / $sample_size ($(correctly_predicted / sample_size))")
                log_info("correctly predicted per 1000: $correctly_predicted_1000 / 1000 ($(correctly_predicted_1000 / 1000))")
            end
        end

        return results
    end
end