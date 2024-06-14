include("DataGenerator.jl")
include("Logger.jl")
include("NodeSketch.jl")

using DataFrames

using .Logger  
using .NodeSketch

const RESULTS_FOLDER = "../results/perf"

struct PerfResult
    n::Integer
    edges::Integer
    L::Integer
    alpha::Number
    calculated_hashes_node_2::Int
    calculated_hashes_node_3::Int
    calculated_hashes_node_4::Int
    calculated_hashes_edge::Int
    time_node_2
    time_node_3
    time_node_4
    time_edge_2
    time_edge_3
    time_edge_4
end

function test_number_of_calculated_hashes()
    ns = collect(1000:50:2000)
    rep = 5
    m = 10
    alpha = 0.3

    results = Vector{PerfResult}()
    for n in ns
        
        for i in 1:rep
            matrix = generate_erdos_renyi_matrix(n, 0.01, 1)
            edges = count(!iszero, matrix) / 2
            println("Erdos-Renyi: $n, $i")
            println("edges: $edges")
            time_edge_2 = @timed begin 
                calculated_hashes_edge = edgesketch(matrix, 2, m, alpha).calculated_hashes
            end
            time_edge_3 = @timed begin 
                res = edgesketch(matrix, 3, m, alpha) 
            end
            time_edge_4 = @timed begin 
                res = edgesketch(matrix, 4, m, alpha) 
            end
            time_node_4 = @timed begin 
                calculated_hashes_node_4 = nodesketch(matrix, 4, m, alpha).calculated_hashes
            end
            time_node_3 = @timed begin 
                calculated_hashes_node_3 = nodesketch(matrix, 3, m, alpha).calculated_hashes
            end
            time_node_2 = @timed begin 
                calculated_hashes_node_2 = nodesketch(matrix, 2, m, alpha).calculated_hashes
            end
            push!(results, PerfResult(n, edges, m, alpha, calculated_hashes_node_2, calculated_hashes_node_3, calculated_hashes_node_4, calculated_hashes_edge, time_node_2.time, time_node_3.time, time_node_4.time, time_edge_2.time, time_edge_3.time, time_edge_4.time))
        end
    end
    path = "$RESULTS_FOLDER/summary_time.csv"
    save_csv(to_df(results), path)
end

function to_df(results::Vector{PerfResult})
    return DataFrame(n = [res.n for res in results],
        edges = [res.edges for res in results],
        m = [res.L for res in results],
        alpha = [res.alpha for res in results],
        calculated_hashes_node_2 = [res.calculated_hashes_node_2 for res in results],
        calculated_hashes_node_3 = [res.calculated_hashes_node_3 for res in results],
        calculated_hashes_node_4 = [res.calculated_hashes_node_4 for res in results],
        calculated_hashes_edge = [res.calculated_hashes_edge for res in results],
        time_node_2 = [res.time_node_2 for res in results],
        time_node_3 = [res.time_node_3 for res in results],
        time_node_4 = [res.time_node_4 for res in results],
        time_edge_2 = [res.time_edge_2 for res in results],
        time_edge_3 = [res.time_edge_3 for res in results],
        time_edge_4 = [res.time_edge_4 for res in results])
end

test_number_of_calculated_hashes()