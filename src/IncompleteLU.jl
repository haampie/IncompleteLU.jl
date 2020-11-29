module IncompleteLU

using SparseArrays
using Base: @propagate_inbounds

struct ILUFactorization{Tv,Ti}
    L::SparseMatrixCSC{Tv,Ti}
    U::SparseMatrixCSC{Tv,Ti}
end

include("sorted_set.jl")
include("linked_list.jl")
include("sparse_vector_accumulator.jl")
include("insertion_sort_update_vector.jl")
include("application.jl")
include("crout_ilu.jl")

end