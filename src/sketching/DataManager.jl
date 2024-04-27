module DataManager

include("Logger.jl")

using MAT
using .Logger

export load_matrix_mat, save_matrix_mat

function load_matrix_mat(path::String, key = "network")
    file = matopen(path)
    varkeys = keys(file)

    log_info("loaded data from file: $path\nvariable keys: $varkeys\n")

    data = read(file)

    return data[key]
end

function save_matrix_mat(matrix, path::String)
    directory = dirname(path)
    
    # Create the directory if it does not exist
    if !isdir(directory)
        mkpath(directory)
    end

    matwrite(path, Dict(
        "embs" => matrix
    ), version="v4")
end

end