module ILU

struct ILUFactorization{T}
    L::SparseMatrixCSC{T}
    U::SparseMatrixCSC{T}
end

include("sorted_set.jl")
include("linked_list.jl")
include("update_vector.jl")
include("application.jl")
include("crout_ilu.jl")

end