module ILU

export ILUFactorization

struct ILUFactorization{T}
    L::SparseMatrixCSC{T}
    U::SparseMatrixCSC{T}
end

include("sorted_set.jl")
include("linked_list.jl")
include("update_vector.jl")
include("insertion_sort_update_vector.jl")
include("application.jl")
include("crout_ilu.jl")

end