import Base: empty!, setindex!, convert

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

function convert(::Type{Vector}, v::SparseVectorAccumulator{T,M}) where {T,M}
    x = zeros(T, M)
    for i = 1 : v.n
        x[v.nzind[i]] = v.nzval[i]
    end
    x
end

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
@inline function axpy!(a::T, A::SparseMatrixCSC{T,I}, column::I, start::I, y::SparseVectorAccumulator{T,N}) where {T,N,I}
    # Loop over the whole column of A
    for idx = start : A.colptr[column + 1] - 1
        add!(y, a * A.nzval[idx], A.rowval[idx])
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
Basically `A[:, j] = y`.
Note: sorts the `nzind`'s of `y`, so that the column
can be added from top to bottom.
Note: does *not* update `A.colptr` for columns > j + 1,
as that is done automatically.
"""
function append_col!(A::SparseMatrixCSC, y::SparseVectorAccumulator{T,N}, j::Int, drop = zero(T)) where {T,N}
    # Sort the indices so we can traverse from top to bottom
    sort!(y.nzind, 1, y.n, Base.Sort.QuickSortAlg(), Base.Order.Forward)
    
    total = 0
    
    # or `for nzind = take(y.nzind, y.n)`
    for idx = 1 : y.n
        row = y.nzind[idx]
        value = y.nzval[y.full[row]]

        if abs(value) ≥ drop || row == j
            # Filter and drop.
            push!(A.rowval, row)
            push!(A.nzval, value)
            total += 1
        end
    end

    A.colptr[j + 1] = A.colptr[j] + total

    return
end
