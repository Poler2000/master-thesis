module Experiments
    include("NodeSketch.jl")
    include("DataManager.jl")
    include("Logger.jl")

    using .NodeSketch
    using .DataManager
    using .Logger
    using Printf

    const dataset = "blogcatalog"
    const DATA_FOLDER = "../data/"
    const RESULTS_FOLDER = "../results/"

    const use_expsketch = true

    matrix = load_matrix_mat("$DATA_FOLDER$dataset.mat")#[1:1000, 1:1000]

    alpha_sim = 0.5
    alpha = 0.002
    k = 3
    sketch_dimensions = 128
    sample_size = 1000

    @time sketch = use_expsketch ? fastexp_nodesketch_extended(matrix, k, sketch_dimensions, alpha_sim) : nodesketch_extended(matrix, k, sketch_dimensions, alpha, alpha_sim)
                
    embs = sketch.embeddings'
    similarity = sketch.similarity_matrix'

    dense_matrix = Matrix(embs)
    matrix_str = join([join([@sprintf("%.1f", element) for element in row], "\t") for row in eachrow(dense_matrix[1:8, 1:16]')], "\n")
    matrix_str_sim = join([join([@sprintf("%.4f", element) for element in row], "\t") for row in eachrow(similarity[1:8, 1:16]')], "\n")
    adj_str = join([join([@sprintf("%.1f", element) for element in row], "\t") for row in eachrow(Matrix(matrix)[1:8, 1:16]')], "\n")
    log_info("Output sample:\n" * matrix_str * '\n')

    log_info("Output sample similarirt:\n" * matrix_str_sim * '\n')

    log_info("Adjecency matrix:\n" * adj_str * '\n')

    # Create a mask to exclude diagonal elements
    mask = trues(size(similarity))
    for i in 1:min(size(similarity)...)
        mask[i, i] = false
    end

    # Flatten the matrix and apply the mask
    flat_A = similarity[mask]
    # Find the indices of the top 1000 elements (if the matrix has fewer than 1000 off-diagonal elements, adjust accordingly)
    sorted_indices = sortperm(flat_A, rev=true)[1:min(sample_size, length(flat_A))]

    # Convert these indices back to the original matrix coordinates
    original_indices = findall(mask)
    top_coordinates = original_indices[sorted_indices]

    # Print the coordinates
    #println("Coordinates of the top elements: ", top_coordinates)

    correctly_predicted = 0
    for (x, y) in Tuple.(top_coordinates)
        if matrix[x,y] == 1
            global correctly_predicted += 1
        end
        println("$x, $y $(matrix[x,y])\t$(matrix[y,x])\t$(similarity[x,y])")
    end

    println("correctly predicted: $correctly_predicted / $sample_size ($(correctly_predicted / sample_size))")

end
#143