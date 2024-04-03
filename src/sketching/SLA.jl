using SparseArrays

struct SLA
    matrix::SparseMatrixCSC{Float64, Int64}
end