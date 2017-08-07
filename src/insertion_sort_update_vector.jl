import Base: getindex, setindex!, indices, convert, empty!

"""
`InsertableSparseVector` accumulates the sparse vector
result from SpMV. Initialization requires O(N) work,
therefore the data structure is reused. Insertion
requires O(nnz) at worst, as insertion sort is used.
"""
struct InsertableSparseVector{Tv}
    values::Vector{Tv}
    indices::SortedSet

    InsertableSparseVector{Tv}(n::Int) where {Tv} = new(Vector{Tv}(n), SortedSet(n))
end

@inline getindex(v::InsertableSparseVector{Tv}, idx::Int) where {Tv} = v.values[idx]
@inline setindex!(v::InsertableSparseVector{Tv}, value::Tv, idx::Int) where {Tv} = v.values[idx] = value
@inline indices(v::InsertableSparseVector) = convert(Vector, v.indices)

function convert(::Type{Vector}, v::InsertableSparseVector{Tv}) where {Tv}
    vals = zeros(Tv, v.indices.N - 1)
    for index in v.indices
        vals[index] = v.values[index]
    end
    return vals
end

"""
Sets `v[idx] += a` when `idx` is occupied, or sets `v[idx] = a`.
Complexity is O(nnz). The `prev_idx` can be used to start the linear
search at `prev_idx`, useful when multiple already sorted values
are added.
"""
function add!(v::InsertableSparseVector{Tv}, a::Tv, idx::Int, prev_idx::Int) where {Tv}
    if push!(v.indices, idx, prev_idx)
        v[idx] = a
    else
        v[idx] += a
    end

    v
end

"""
Add without providing a previous index.
"""
@inline add!(v::InsertableSparseVector{Tv}, a::Tv, idx::Int) where {Tv} = add!(v, a, idx, v.indices.N)

function axpy!(a::Tv, A::SparseMatrixCSC{Tv}, column::Int, start::Int, y::InsertableSparseVector{Tv}) where {Tv}
    prev_index = y.indices.N
    
    @inbounds for idx = start : A.colptr[column + 1] - 1
        add!(y, a * A.nzval[idx], A.rowval[idx], prev_index)
        prev_index = A.rowval[idx]
    end

    y
end

"""
Empties the InsterableSparseVector in O(1) operations.
"""
@inline empty!(v::InsertableSparseVector) = empty!(v.indices)

"""
Basically `A[:, j] = scale * drop(y)`, where drop removes
values less than `drop`.

Resets the `InsertableSparseVector`.

Note: does *not* update `A.colptr` for columns > j + 1,
as that is done during the steps.
"""
function append_col!(A::SparseMatrixCSC{Tv}, y::InsertableSparseVector{Tv}, j::Int, drop::Tv, scale::Tv = one(Tv)) where {Tv}
    
    total = 0
    
    for row = y.indices
        if abs(y[row]) ≥ drop || row == j
            push!(A.rowval, row)
            push!(A.nzval, scale * y[row])
            total += 1
        end
    end

    A.colptr[j + 1] = A.colptr[j] + total
    
    empty!(y)

    nothing
end