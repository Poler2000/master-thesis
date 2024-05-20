module DataManager

include("Logger.jl")

using MAT
using CSV
using DataFrames

using .Logger

export load_matrix_mat, save_matrix_embs, save_matrix_network, save_csv

function load_matrix_mat(path::String, key = "network")
    file = matopen(path)
    varkeys = keys(file)

    log_info("loaded data from file: $path\nvariable keys: $varkeys\n")

    data = read(file)

    return data[key]
end

function save_matrix_embs(matrix, path::String)
    save_matrix_core(matrix, path, "embs")
end

function save_matrix_network(matrix, path::String)
    save_matrix_core(matrix, path, "network")
end

function save_csv(df::DataFrame, path)
    should_append = isfile(path)
    CSV.write(path, df, append=should_append)
end

function save_matrix_core(matrix, path::String, key::String)
    directory = dirname(path)
    
    # Create the directory if it does not exist
    if !isdir(directory)
        mkpath(directory)
    end

    matwrite(path, Dict(
        key => matrix
    ), version="v4")
end

end