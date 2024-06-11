include("DataGenerator.jl")
include("Logger.jl")
include("NodeSketch.jl")

using DataFrames

using .Logger  
using .NodeSketch

function test()
    n = 1000
    m = 10
    alpha = 0.3

    matrix = generate_stochastic_block_matrix(n, 4, 0.5, 0.001)
    edges = count(!iszero, matrix) / 2
    println("Number of edges: $edges")
    println("--------------------------------------------------")

    for k in 2:4
        nodesketch(matrix, k, m, alpha) 
        for i in 1:3
            time_result = @timed begin 
                nodesketch(matrix, k, m, alpha) 
            end
    
            println("NodeSketch with matrix, k: $k, test: $i, time: \t$(time_result.time)s")
        end
        println()
    end
    println("--------------------------------------------------")

    for k in 2:4
        edgesketch(matrix, k, m, alpha) 
        for i in 1:3
            time_result = @timed begin 
                edgesketch(matrix, k, m, alpha) 
            end
    
            println("EdgeSketch with matrix, k: $k, test: $i, time: \t$(time_result.time)s")
        end
        println()
    end
    println("--------------------------------------------------")

    for k in 2:4
        nodesketch_pure(matrix, k, m, alpha) 
        for i in 1:3
            time_result = @timed begin 
                nodesketch_pure(matrix, k, m, alpha) 
            end
    
            println("NodeSketch no matrix, k: $k, test: $i, time: \t$(time_result.time)s")
        end
        println()
    end
    println("--------------------------------------------------")

    for k in 2:4
        edgesketch_pure(matrix, k, m, alpha) 
        for i in 1:3
            time_result = @timed begin 
                edgesketch_pure(matrix, k, m, alpha) 
            end
    
            println("EdgeSketch no matrix, k: $k, test: $i, time: \t$(time_result.time)s")
        end
        println()
    end
    println("--------------------------------------------------")

end

test()