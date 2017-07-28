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
