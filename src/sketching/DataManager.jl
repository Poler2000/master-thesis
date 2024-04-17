module DataManager

include("Logger.jl")

using MAT
using .Logger

function load_matrix(path::String, key = "network")
    file = matopen(path)
    varkeys = keys(file)

    Logger.log_msg("loaded data from file: $path\n
                variable keys: $varkeys")

    data = read(file)

    return data[key]
end

function save_matrix(matrix, path::String)
    matwrite(path, Dict(
        "embs" => matrix
    ), version="v4")
end

end