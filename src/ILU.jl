module ILU

import Base.Order.Ordering
import Base.Order.lt

using DataStructures

export lu!, csc_by_row

"""
Dense indexing: reference example of Crout ILU
"""
function crout(A::AbstractMatrix{T}; τ = 1e-3) where {T}
    n = size(A, 1)
    L = zeros(A)
    U = zeros(A)

    for k = 1 : n

        # Initialize the row & col
        z = zeros(1, n)
        z[k : n] = A[k, k : n]
        w = zeros(n, 1)
        w[k + 1 : n] = A[k + 1 : n, k]

        for i = 1 : k - 1
            if L[k, i] != zero(T)
                z[k : n] -= L[k, i] * U[i, k : n]
            end
        end

        # Initialize the column
        for i = 1 : k - 1
            if U[i, k] != zero(T)
                w[k + 1 : n] -= U[i, k] * L[k + 1 : n, i]
            end
        end

        # Apply dropping rule to row z and column w
        map!(x -> abs(x) < τ ? zero(T) : x, z, z)
        map!(x -> abs(x) < τ ? zero(T) : x, w, w)

        # Update the final columns in L and U
        U[k, :] = z
        L[:, k] = w / U[k, k]
        L[k, k] = one(T)
    end

    L, U
end

struct RowOrdering{matT <: SparseMatrixCSC} <: Ordering
    A::matT
end

lt(wrap::RowOrdering, left::Int, right::Int) = @inbounds return wrap.A.rowval[left] < wrap.A.rowval[right]

function csc_by_row_pq(A::SparseMatrixCSC{T}) where {T}
    n = size(A, 1)
    
    # Map from column -> current index in row, ordered by A.rowval[index]
    pq = PriorityQueue(collect(1 : n), A.colptr[1 : end - 1], RowOrdering(A))
    
    row = 1
    
    @inbounds while !isempty(pq)
        column, idx = peek(pq)

        next_row = A.rowval[idx]

        if next_row > row
            row = next_row
            println()
        end

        print(A.nzval[idx], " (", column ,") ")

        # If the next element is in the next column, remove it
        # otherwise update
        if idx + 1 == A.colptr[column + 1]
            dequeue!(pq)
        else
            pq[column] = idx + 1
        end
    end
end
end
