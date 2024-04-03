using MAT
include("NodeSketch.jl")
include("SLA.jl")

file = matopen("../data/dblp.mat")
varnames = keys(file)

hello = read(file)

for v in varnames
    println(v)
end

println(typeof(hello))
println(typeof(hello["network"]))
#println(hello["group"])

for y in 1:50
    for x in 1:50
        print("$(hello["network"][x, y]) ")
    end
    println()
end

close(file)

submatrix = hello["network"][1:100, 1:100]

node_sketch(submatrix, 3, 0.5)