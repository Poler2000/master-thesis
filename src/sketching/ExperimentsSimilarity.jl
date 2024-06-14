module Experiments
    include("NodeSketch.jl")
    include("DataManager.jl")
    include("Logger.jl")

    using DataFrames

    using .NodeSketch
    using .DataManager
    using .Logger

    const DATA_FOLDER = "../data/"
    const RESULTS_FOLDER = "../results/sim"

    export run_tests_all_datasets, run_all_similarity, run_similarity_var_k, run_similarity_var_L, run_similarity_var_alpha

    mutable struct SimResult
        dataset::String
        algorithm::String
        k::Integer
        L::Integer
        alpha::Number
        sim_top_100::Number
        sim_top_1000::Number
        sim_top_10000::Number
        sim_all::Number
        calculated_hashes::Int
    end

    function run_tests_all_datasets()
        file_names = readdir(DATA_FOLDER)
        datasets = [splitext(file)[1] for file in file_names]

        run_all_similarity(datasets)
    end

    function run_all_similarity(datasets)
        default_alpha = 0.3
        default_size = 10

        run_similarity_var_k(datasets, [2, 3, 4], default_alpha, default_size)
        run_similarity_var_L(datasets, 2, default_alpha, [8, 16, 32, 64, 128, 256])
        run_similarity_var_alpha(datasets, 4, [0.0, 0.15, 0.3, 0.45], default_size)
    end

    function run_similarity_var_k(datasets::Array{String}, orders::Vector{<:Integer}, 
        alpha::Number, sketch_dimensions::Number)

        res_edge = run_similarity_var_k_single(datasets, true, orders, alpha, sketch_dimensions, false)
        res_edge_order_est = run_similarity_var_k_single(datasets, true, orders, alpha, sketch_dimensions, true)

        for res in res_edge_order_est
            res.algorithm = "EdgeSketch_order_estimation"
        end

        res_node = run_similarity_var_k_single(datasets, false, orders, alpha, sketch_dimensions, false)

        print_summary(datasets, orders, "k", res_node, res_edge, "NodeSketch", "EdgeSketch")
        print_summary(datasets, orders, "k", res_edge, res_edge_order_est, "EdgeSketch", "EdgeSketch with order estimation)")
        path = "$RESULTS_FOLDER/summary_k.csv"
        save_csv(to_df(res_node), path)
        save_csv(to_df(res_edge), path)
        save_csv(to_df(res_edge_order_est), path)
    end

    function run_similarity_var_L(datasets::Array{String}, order::Integer, 
        alpha::Number, dimensions::Vector{<:Integer})

        res_edge = run_similarity_var_L_single(datasets, true, order, alpha, dimensions)
        res_node = run_similarity_var_L_single(datasets, false, order, alpha, dimensions)

        print_summary(datasets, dimensions, "m", res_node, res_edge, "NodeSketch", "EdgeSketch")
        path = "$RESULTS_FOLDER/summary_m.csv"
        save_csv(to_df(res_node), path)
        save_csv(to_df(res_edge), path)
    end

    function run_similarity_var_alpha(datasets::Array{String}, order::Integer, 
        alphas::Vector{<:Number}, sketch_dimensions::Number)

        res_edge = run_similarity_var_alpha_single(datasets, true, order, alphas, sketch_dimensions)
        res_node = run_similarity_var_alpha_single(datasets, false, order, alphas, sketch_dimensions)

        print_summary(datasets, alphas, "alpha", res_node, res_edge, "NodeSketch", "EdgeSketch")
        path = "$RESULTS_FOLDER/summary_alpha.csv"
        save_csv(to_df(res_node), path)
        save_csv(to_df(res_edge), path)
    end

    function print_summary(datasets::Array{String}, variables::Array{<:Number}, label::String, res1, res2, alg1::String, alg2::String)
        i = 0
        path = "$RESULTS_FOLDER/summary_$(label).txt"
        for dataset in datasets
            log_info("\nSummary for dataset: $dataset, sketch size = $(res1[1].L), alpha = $(res1[1].alpha)", path)
            log_info("\t|\t\t\t\t\t $alg1 \t\t\t\t\t\t|\t\t\t\t\t$alg2", path)
            log_info("$label\t|\tt = 100 \t|\tt = 1000 \t|\tt = 10000 \t|\tt = |E| \t|\t #hashes \t|\tt = 100 \t|\tt = 1000 \t|\tt = 10000 \t|\tt = |E|\t|\t #hashes ", path)
            
            for variable in variables
                i += 1
                log_info("$variable\t|\t$(res1[i].sim_top_100) \t\t|\t$(res1[i].sim_top_1000) \t\t|\t$(res1[i].sim_top_10000) \t\t|\t$(res1[i].sim_all)\t\t|\t $(res1[i].calculated_hashes) \t|\t$(res2[i].sim_top_100) \t\t|\t$(res2[i].sim_top_1000) \t\t|\t$(res2[i].sim_top_10000) \t\t|\t$(res2[i].sim_all)\t|\t $(res2[i].calculated_hashes) \t", path)
            end
        end
    end

    function run_similarity_var_k_single(datasets::Array{String}, use_expsketch::Bool, 
        orders::Vector{<:Integer}, alpha::Number, sketch_dimensions::Number, use_order_estimations::Bool )::Vector{SimResult}
        results = Vector{SimResult}()
        for dataset in datasets
            matrix = Matrix(load_matrix_mat("$DATA_FOLDER$dataset.mat"))
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
                log_info("k = $k = $alpha, L = $sketch_dimensions")
                @time sketch = use_expsketch ? edgesketch(matrix, k, sketch_dimensions, alpha) : nodesketch(matrix, k, sketch_dimensions, alpha)
                
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

                sorted_indices = sortperm(flat_A, rev=true)
            
                original_indices = findall(mask)

                top_coordinates = original_indices[sorted_indices]
            
                correctly_predicted = 0
                correctly_predicted_100 = 0
                correctly_predicted_1000 = 0
                correctly_predicted_10000 = 0

                estimations = Dict(i => (sketch_dimensions - 1) / sum(embs[i, :]) for i in 1:n)
                guessed = 0
                guessed_dict = Dict(i => 0 for i in 1:n)
    
                for (i, (x, y)) in enumerate(Tuple.(top_coordinates))
                    if use_order_estimations && (guessed_dict[y] + 1 > estimations[y] || guessed_dict[x] + 1 > estimations[x])
                        continue
                    end

                    if matrix[x, y] >= 1
                        correctly_predicted += 1
                        if i <= 100
                            correctly_predicted_100 += 1
                        end
                        if i <= 1000
                            correctly_predicted_1000 += 1
                        end
                        if i <= 10000
                            correctly_predicted_10000 += 1
                        end
                    end
                    guessed_dict[x] += 1
                    guessed_dict[y] += 1

                    guessed += 1
                    if guessed >= sample_size
                        break
                    end
                end

                push!(results, SimResult(dataset, use_expsketch ? "EdgeSketch" : "NodeSketch", k, sketch_dimensions, alpha, (correctly_predicted_100 / 100), (correctly_predicted_1000 / 1000), (correctly_predicted_10000 / 10000), round(correctly_predicted / sample_size, digits=4), sketch.calculated_hashes))
            
                log_info("correctly predicted: $correctly_predicted / $sample_size ($(correctly_predicted / sample_size))")
                log_info("correctly predicted per 100: $correctly_predicted_100 / 100 ($(correctly_predicted_100 / 100))")
                log_info("correctly predicted per 1000: $correctly_predicted_1000 / 1000 ($(correctly_predicted_1000 / 1000))")
                log_info("correctly predicted per 10000: $correctly_predicted_10000 / 1000 ($(correctly_predicted_10000 / 10000))")
            end
        end

        return results
    end

    function run_similarity_var_L_single(datasets::Array{String}, use_expsketch::Bool, 
        k::Integer, alpha::Number, dimensions::Vector{<:Integer})::Vector{SimResult}
        results = Vector{SimResult}()
        for dataset in datasets
            matrix = Matrix(load_matrix_mat("$DATA_FOLDER$dataset.mat"))
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
                log_info("k = $k = $alpha, L = $sketch_dimensions")
                @time sketch = use_expsketch ? edgesketch(matrix, k, sketch_dimensions, alpha) : nodesketch(matrix, k, sketch_dimensions, alpha)
                
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
                correctly_predicted_100 = 0
                correctly_predicted_1000 = 0
                correctly_predicted_10000 = 0

                for (i, (x, y)) in enumerate(Tuple.(top_coordinates))
                    if matrix[x, y] >= 1
                        correctly_predicted += 1
                        if i <= 100
                            correctly_predicted_100 += 1
                        end
                        if i <= 1000
                            correctly_predicted_1000 += 1
                        end
                        if i <= 10000
                            correctly_predicted_10000 += 1
                        end
                    end
                end

                push!(results, SimResult(dataset, use_expsketch ? "EdgeSketch" : "NodeSketch", k, sketch_dimensions, alpha, (correctly_predicted_100 / 100), (correctly_predicted_1000 / 1000), (correctly_predicted_10000 / 10000), round(correctly_predicted / sample_size, digits=4), sketch.calculated_hashes))
            
                log_info("correctly predicted: $correctly_predicted / $sample_size ($(correctly_predicted / sample_size))")
                log_info("correctly predicted per 100: $correctly_predicted_100 / 100 ($(correctly_predicted_100 / 100))")
                log_info("correctly predicted per 1000: $correctly_predicted_1000 / 1000 ($(correctly_predicted_1000 / 1000))")
                log_info("correctly predicted per 10000: $correctly_predicted_10000 / 1000 ($(correctly_predicted_10000 / 10000))")
            end
        end

        return results
    end

    function run_similarity_var_alpha_single(datasets::Array{String}, use_expsketch::Bool, 
        k::Integer, alphas::Vector{<:Number}, sketch_dimensions::Number)::Vector{SimResult}
        results = Vector{SimResult}()
        for dataset in datasets
            matrix = Matrix(load_matrix_mat("$DATA_FOLDER$dataset.mat"))
            sample_size = 0
            n = size(matrix, 1)
            for i in 1:n-1
                for j in i+1:n
                    if matrix[i, j] != 0
                        sample_size += 1
                    end
                end
            end

            for alpha in alphas
                log_info("Executing $(use_expsketch ? "expsketch-based" : "basic") algorithm for dataset: $dataset")
                log_info("k = $k = $alpha, L = $sketch_dimensions")
                @time sketch = use_expsketch ? edgesketch(matrix, k, sketch_dimensions, alpha) : nodesketch(matrix, k, sketch_dimensions, alpha)
                
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
                correctly_predicted_100 = 0
                correctly_predicted_1000 = 0
                correctly_predicted_10000 = 0

                for (i, (x, y)) in enumerate(Tuple.(top_coordinates))
                    if matrix[x, y] >= 1
                        correctly_predicted += 1
                        if i <= 100
                            correctly_predicted_100 += 1
                        end
                        if i <= 1000
                            correctly_predicted_1000 += 1
                        end
                        if i <= 10000
                            correctly_predicted_10000 += 1
                        end
                    end
                end

                push!(results, SimResult(dataset, use_expsketch ? "EdgeSketch" : "NodeSketch", k, sketch_dimensions, alpha, (correctly_predicted_100 / 100), (correctly_predicted_1000 / 1000), (correctly_predicted_10000 / 10000), round(correctly_predicted / sample_size, digits=4), sketch.calculated_hashes))
            
                log_info("correctly predicted: $correctly_predicted / $sample_size ($(correctly_predicted / sample_size))")
                log_info("correctly predicted per 100: $correctly_predicted_100 / 100 ($(correctly_predicted_100 / 100))")
                log_info("correctly predicted per 1000: $correctly_predicted_1000 / 1000 ($(correctly_predicted_1000 / 1000))")
                log_info("correctly predicted per 10000: $correctly_predicted_10000 / 1000 ($(correctly_predicted_10000 / 10000))")
            end
        end

        return results
    end

    function to_df(results::Vector{SimResult})
        return DataFrame(dataset = [res.dataset for res in results],
            algorithm = [res.algorithm for res in results],
            k = [res.k for res in results],
            m = [res.L for res in results],
            alpha = [res.alpha for res in results],
            sim_top_100 = [res.sim_top_100 for res in results],
            sim_top_1000 = [res.sim_top_1000 for res in results],
            sim_top_10000 = [res.sim_top_10000 for res in results],
            sim_all = [res.sim_all for res in results])  
    end
end