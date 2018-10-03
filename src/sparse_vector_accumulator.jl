import Base: setindex!, empty!, Vector
import LinearAlgebra: axpy!

"""
`SparseVectorAccumulator` accumulates the sparse vector
resulting from SpMV. Initialization requires O(N) work,
therefore the data structure is reused. Insertion is O(1).
Note that `nzind` is unordered. Also note that there is
wasted space: `nzind` could be a growing list. Pre-allocation
seems faster though.

SparseVectorAccumulator incorporates the multiple switch technique
by Gustavson (1976), which makes resetting an O(1) operation rather
than O(nnz): the `curr` value is used to flag the occupied indices,
and `curr` is increased at each reset.

occupied = [0, 1, 0, 1, 0, 0, 0]
nzind = [2, 4, 0, 0, 0, 0]
nzval = [0., .1234, 0., .435, 0., 0., 0.]
nnz = 2
length = 7
curr = 1
"""
mutable struct SparseVectorAccumulator{T}
    occupied::Vector{Int}
    nzind::Vector{Int}
    nzval::Vector{T}
    nnz::Int
    length::Int
    curr::Int

    return SparseVectorAccumulator{T}(N::Int) where {T} = new(
        zeros(Int, N),
        Vector{Int}(undef, N),
        Vector{T}(undef, N),
        0,
        N,
        1
    )
end

function Vector(v::SparseVectorAccumulator{T}) where {T}
    x = zeros(T, v.length)
    @inbounds x[v.nzind[1 : v.nnz]] = v.nzval[v.nzind[1 : v.nnz]]
    return x
end

"""
Add a part of a SparseMatrixCSC column to a SparseVectorAccumulator,
starting at a given index until the end.
"""
function axpy!(a::Tv, A::SparseMatrixCSC{Tv}, column::Int, start::Int, y::SparseVectorAccumulator{Tv}) where {Tv}
    # Loop over the whole column of A
    @inbounds for idx = start : A.colptr[column + 1] - 1
        add!(y, a * A.nzval[idx], A.rowval[idx])
    end

    return y
end

"""
Sets `v[idx] += a` when `idx` is occupied, or sets `v[idx] = a`.
Complexity is O(1).
"""
function add!(v::SparseVectorAccumulator{Tv}, a::Tv, idx::Int) where {Tv}
    @inbounds begin
        if isoccupied(v, idx)
            v.nzval[idx] += a
        else
            v.nnz += 1
            v.occupied[idx] = v.curr
            v.nzval[idx] = a
            v.nzind[v.nnz] = idx
        end
    end

    return nothing
end

"""
Check whether `idx` is nonzero.
"""
@propagate_inbounds isoccupied(v::SparseVectorAccumulator, idx::Int) = v.occupied[idx] == v.curr

"""
Empty the SparseVectorAccumulator in O(1) operations.
"""
@inline function empty!(v::SparseVectorAccumulator)
    v.curr += 1
    v.nnz = 0
end

"""
Basically `A[:, j] = scale * drop(y)`, where drop removes
values less than `drop`. Note: sorts the `nzind`'s of `y`, 
so that the column can be appended to a SparseMatrixCSC.

Resets the `SparseVectorAccumulator`.

Note: does *not* update `A.colptr` for columns > j + 1,
as that is done during the steps.
"""
function append_col!(A::SparseMatrixCSC{Tv}, y::SparseVectorAccumulator{Tv}, j::Int, drop::Tv, scale::Tv = one(Tv)) where {Tv}
    # Move the indices of interest up front
    total = 0

    @inbounds for idx = 1 : y.nnz
        row = y.nzind[idx]
        value = y.nzval[row]

        if abs(value) ≥ drop || row == j
            total += 1
            y.nzind[total] = row
        end
    end

    # Sort the retained values.
    sort!(y.nzind, 1, total, Base.Sort.QuickSortAlg(), Base.Order.Forward)
    
    @inbounds for idx = 1 : total
        row = y.nzind[idx]
        push!(A.rowval, row)
        push!(A.nzval, scale * y.nzval[row])
    end

    @inbounds A.colptr[j + 1] = A.colptr[j] + total
    
    empty!(y)

    return nothing
end
