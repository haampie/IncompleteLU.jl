import Base: setindex!, convert

"""
SparseVectorAccumulator is a container that makes combining
columns of a SparseMatrixCSC cheap. It extends SparseVector
with two full vectors and also does not keep `nzind` sorted.
This allows O(1) insertions.
occupied = [0, 1, 0, 1, 0, 0, 0]
nzind = [2, 4, 0, 0, 0, 0]
nzval = [0., .1234, 0., .435, 0., 0., 0.]
n = 2
length = 7
curr = 1
"""
mutable struct SparseVectorAccumulator{T}
    occupied::Vector{Int}
    nzind::Vector{Int}
    nzval::Vector{T}
    n::Int
    length::Int
    curr::Int

    SparseVectorAccumulator{T}(N::Int) where {T} = new(
        zeros(Int, N),
        Vector{Int}(N),
        Vector{T}(N),
        0,
        N,
        1
    )
end

function convert(::Type{Vector}, v::SparseVectorAccumulator{T}) where {T}
    x = zeros(T, v.length)
    x[v.nzind[1 : v.n]] = v.nzval[v.nzind[1 : v.n]]
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
    if v.occupied[idx] != v.curr
        v.n += 1
        v.occupied[idx] = v.curr
        v.nzval[idx] = a
        v.nzind[v.n] = idx
    else # Update
        v.nzval[idx] += a
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
    # Move the indices of interest up front
    total = 0

    for idx = 1 : y.n
        row = y.nzind[idx]
        value = y.nzval[row]

        if abs(value) ≥ drop || row == j
            total += 1
            y.nzind[total] = row
        end
    end

    sort!(y.nzind, 1, total, Base.Sort.QuickSortAlg(), Base.Order.Forward)
    
    for idx = 1 : total
        row = y.nzind[idx]
        push!(A.rowval, row)
        push!(A.nzval, scale * y.nzval[row])
    end

    A.colptr[j + 1] = A.colptr[j] + total
    y.curr += 1
    y.n = 0

    nothing
end
