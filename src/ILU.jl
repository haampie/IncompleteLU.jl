module ILU

include("linked_list.jl")
include("update_vector.jl")

struct ILUFactorization{T}
    L::SparseMatrixCSC{T}
    U::SparseMatrixCSC{T}
end

include("application.jl")
include("crout_ilu.jl")

end