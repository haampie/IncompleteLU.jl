module IncompleteLU

using SparseArrays
using Base: @propagate_inbounds

struct ILUFactorization{T}
    L::SparseMatrixCSC{T}
    U::SparseMatrixCSC{T}
end

include("sorted_set.jl")
include("linked_list.jl")
include("sparse_vector_accumulator.jl")
include("insertion_sort_update_vector.jl")
include("application.jl")
include("crout_ilu.jl")

end