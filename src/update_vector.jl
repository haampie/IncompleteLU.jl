import Base: empty!, setindex!
using Base.Test

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
Add a part of a SparseMatrixCSC column to a SparseVectorAccumulator,
starting at a given index until the end.
"""
function axpy!(a::T, A::SparseMatrixCSC, column::Int, start::Int, y::SparseVectorAccumulator{T,N}) where {T,N}
    # Loop over the whole column of A
    for idx = start : A.colptr[column + 1] - 1
        row = A.rowval[idx]
        add!(y, a * A.nzval[idx], row)
    end

    y
end

"""
Does v[idx] += a.
"""
function add!(v::SparseVectorAccumulator{T,N}, a::T, idx::Int) where {T,N}
    # Fill-in
    if v.full[idx] == 0
        v.n += 1
        v.full[idx] = v.n

        if length(v.nzval) < v.n
            push!(v.nzval, a)
            push!(v.nzind, idx)
        else
            v.nzval[v.n] = a
            v.nzind[v.n] = idx
        end
    else # Update
        v.nzval[v.full[idx]] += a
    end
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


### Tests

"""
Computes B = A[:, 1 : end - 1] + A[:, 2 : end]
"""
function testing()
    A = sprand(10, 11, .1) + 2I
    B = spzeros(10, 10)
    v = SparseVectorAccumulator{Float64}(10)

    for i = 1 : 10
        axpy!(1.0, A, i, A.colptr[i], v)
        axpy!(1.0, A, i + 1, A.colptr[i + 1], v)
        append_col!(B, v, i)
        empty!(v)
    end

    @test full(A[:, 2 : end]) + full(A[:, 1 : end - 1]) == full(B)
end