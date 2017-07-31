import Base: empty!

"""
SparseVectorAccumulator is a container that makes combining
columns of a SparseMatrixCSC cheap. It extends SparseVector
with one full vector and also does not keep `nzind` sorted.
This allows cheap insertions.
full = [0, 1, 0, 2, 0, 0, 0]
nzind = [2, 4]
nzval = [.1234, .435]
n = 2
"""
mutable struct SparseVectorAccumulator{T,N}
    full::Vector{Int}
    nzind::Vector{Int}
    nzval::Vector{T}
    n::Int
end

SparseVectorAccumulator{T}(N::Int) where {T} = SparseVectorAccumulator{T,N}(
    zeros(Int, N),
    Int[],
    T[],
    0
)

"""
Reset the vector to a vector of just zeros.
Don't bother shrinking or zero'ing out the previous
nonzeros or their indices.
"""
function empty!(v::SparseVectorAccumulator)
    for i = 1 : v.n
        v.full[v.nzind[i]] = 0
    end

    v.n = 0
end

"""
Make a linear combination of a SparseMatricCSC column with
a SparseVectorAccumulator.
"""
function axpy!(a::T, mat::SparseMatrixCSC, column::Int, y::SparseVectorAccumulator{T,N}) where {T,N}
    idx = mat.colptr[column]
    next_column_index = mat.colptr[column + 1]

    # Loop over the whole column of A
    while idx != next_column_index
        row = mat.rowval[idx]

        # Fill in.
        if y.full[row] == 0

            y.n += 1
            y.full[row] = y.n

            # Either push or overwrite.
            if length(y.nzval) < y.n
                push!(y.nzval, a * mat.nzval[idx])
                push!(y.nzind, row)
            else
                y.nzval[y.n] = a * mat.nzval[idx]
                y.nzind[y.n] = row
            end
        else # Just an update
            y.nzval[y.full[row]] += a * mat.nzval[idx]
        end

        idx += 1
    end

    y
end

"""
Basically `A[:, j] = y`. TODO: dropping strategy
Note: sorts the `nzind`'s of `y`, so that the column
can be added from top to bottom.
Note: does *not* update `A.colptr` for columns > j + 1,
as that is done automatically.
"""
function append_col!(A::SparseMatrixCSC, y::SparseVectorAccumulator{T,N}, j::Int) where {T,N}
    # Sort the indices so we can traverse from top to bottom
    sort!(y.nzind, 1, y.n, Base.Sort.QuickSortAlg(), Base.Order.Forward)
    
    # or `for nzind = take(y.nzind, y.n)`
    for idx = 1 : y.n
        # Filter and drop.
        push!(A.rowval, y.nzind[idx])

        # y.full[idx] gives the index of the nonzero
        # maybe remove indirection by sorting nzval as well.
        push!(A.nzval, y.nzval[y.full[y.nzind[idx]]])
    end

    A.colptr[j + 1] = A.colptr[j] + y.n

    return
end

"""
Computes B = A[:, 1 : end - 1] + A[:, 2 : end]
"""
function testing()
    A = sprand(10, 11, .1) + 2I
    B = spzeros(10, 10)
    v = SparseVectorAccumulator{Float64}(10)

    for i = 1 : 10
        axpy!(1.0, A, i, v)
        axpy!(1.0, A, i + 1, v)
        append_col!(B, v, i)
        empty!(v)
    end

    A, B
end