import Base: setindex!, convert

"""
SparseVectorAccumulator is a container that makes combining
columns of a SparseMatrixCSC cheap. It extends SparseVector
with one full vector and also does not keep `nzind` sorted.
This allows cheap insertions.
full = [0, 1, 0, 2, 0, 0, 0]
nzind = [2, 4]
nzval = [.1234, .435]
n = 2
length = 7
"""
mutable struct SparseVectorAccumulator{T}
    full::Vector{Int}
    nzind::Vector{Int}
    nzval::Vector{T}
    n::Int
    length::Int

    SparseVectorAccumulator{T}(N::Int) where {T} = new(
        zeros(Int, N),
        zeros(Int, N),
        zeros(T, N),
        0,
        N
    )
end

function convert(::Type{Vector}, v::SparseVectorAccumulator{T}) where {T}
    x = zeros(T, v.length)
    for i = 1 : v.n
        x[v.nzind[i]] = v.nzval[i]
    end
    x
end

"""
Add a part of a SparseMatrixCSC column to a SparseVectorAccumulator,
starting at a given index until the end.
"""
function axpy!(a::Tv, A::SparseMatrixCSC{Tv}, column::Int, start::Int, y::SparseVectorAccumulator{Tv}) where {Tv}
    # Loop over the whole column of A
    for idx = start : A.colptr[column + 1] - 1
        add!(y, a * A.nzval[idx], A.rowval[idx])
    end

    y
end

"""
Does v[idx] += a.
"""
function add!(v::SparseVectorAccumulator{Tv}, a::Tv, idx::Int) where {Tv}
    # Fill-in
    if v.full[idx] == 0
        v.n += 1
        v.full[idx] = v.n
        v.nzval[v.n] = a
        v.nzind[v.n] = idx
    else # Update
        v.nzval[v.full[idx]] += a
    end

    nothing
end

"""
Basically `A[:, j] = y`.
Note: sorts the `nzind`'s of `y`, so that the column
can be added from top to bottom.
Note: does *not* update `A.colptr` for columns > j + 1,
as that is done automatically.
"""
function append_col!(A::SparseMatrixCSC{Tv}, y::SparseVectorAccumulator{Tv}, j::Int, drop::Tv, scale::Tv = one(Tv)) where {Tv}
    # Sort the indices so we can traverse from top to bottom
    sort!(y.nzind, 1, y.n, Base.Sort.QuickSortAlg(), Base.Order.Forward)
    
    total = 0
    
    for idx = 1 : y.n
        row = y.nzind[idx]
        value = y.nzval[y.full[row]]

        if abs(value) ≥ drop || row == j
            # Filter and drop.
            push!(A.rowval, row)
            push!(A.nzval, value)
            total += 1
        end

        y.full[row] = 0
    end

    A.colptr[j + 1] = A.colptr[j] + total
    y.n = 0

    nothing
end
