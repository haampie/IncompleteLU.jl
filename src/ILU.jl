module ILU

import Base.Order.Ordering
import Base.Order.lt

using DataStructures

export crout_ilu

include("update_vector.jl")

struct RowOrdering{matT <: SparseMatrixCSC} <: Ordering
    A::matT
end

lt(wrap::RowOrdering, left::Int, right::Int) = @inbounds return wrap.A.rowval[left] < wrap.A.rowval[right]

function crout_ilu(A::SparseMatrixCSC{T}; τ = 1e-3) where {T}
    n = size(A, 1)

    pq = PriorityQueue(collect(1 : n), A.colptr[1 : end - 1], RowOrdering(A))

    L = spzeros(T, n, n)
    U = spzeros(T, n, n)
    
    U_row = SparseVectorAccumulator{T}(n)
    L_col = SparseVectorAccumulator{T}(n)

    # U_first[i] is the index of the first nonzero of U in row i after the diagonal
    # L_first[i] is the index of the first nonzero of L in column i after the diagonal
    U_first = zeros(Int, n)
    L_first = zeros(Int, n)

    # U_nonzero_col[i] is a vector of indices of nonzero elements in the i'th column of U before the diagonal
    # L_nonzero_row[i] is a vector of indices of nonzero elements in the i'th row of L before the diagonal
    U_nonzero_col = Vector{Vector{Int}}(n)
    L_nonzero_row = Vector{Vector{Int}}(n)

    # Initialization
    for i = 1 : n
        U_nonzero_col[i] = Vector{Int}()
        L_nonzero_row[i] = Vector{Int}()
    end

    for k = 1 : n

        ##
        ## Copy the new row into U_row and the new column into L_col
        ##

        # Copy all row elements into U_row
        while !isempty(pq)
            column, idx = peek(pq)

            # Quit when we're past the current row.
            if A.rowval[idx] != k
                break
            end

            # Just remove columns before the diagonal
            if column < k
                dequeue!(pq)
                continue
            end

            # Copy the row value into U
            add!(U_row, A.nzval[idx], column)

            # Remove if this was the last column entry,
            # otherwise update the column index.
            if idx == A.colptr[column + 1] - 1
                dequeue!(pq)
            else
                pq[column] = idx + 1
            end
        end

        # Copy the remaining part of the column into L_col
        axpy!(one(T), A, k, get(pq, k, A.colptr[k + 1]), L_col)

        ##
        ## Combine the vectors:
        ##
        
        # L_col[k+1:n] -= U[i,k] * L[i,k+1:n] for i = 1 : k - 1
        # Immediately update L_first and L_nonzero_row
        for col = L_nonzero_row[k]
            axpy!(-L.nzval[L_first[col]], U, col, U_first[col], U_row)

            L_first[col] += 1

            # If there is still another value in column `col` of L
            # then add it to L_nonzero_row as well.
            if L_first[col] != L.colptr[col + 1]
                push!(L_nonzero_row[L.rowval[L_first[col]]], col)
            end
        end

        # U_row[k:n] -= L[k,i] * U[i,k:n] for i = 1 : k - 1
        for row = U_nonzero_col[k]
            axpy!(-U.nzval[U_first[row]], L, row, L_first[row], L_col)

            U_first[row] += 1

            # Check if there is still another value in row `row` of U
            # if so, inform U_nonzero_col as well.
            if U_first[row] != U.colptr[row + 1]
                push!(U_nonzero_col[U.rowval[U_first[row]]], row)
            end
        end

        U_diag_element = U_row.nzval[U_row.full[k]]

        for i = 1 : L_col.n
            L_col.nzval[i] /= U_diag_element
        end

        # Add a one to the diagonal.
        add!(L_col, one(T), k)
    
        ## 
        ## Apply a drop rule
        ##

        # Append the columns
        append_col!(U, U_row, k, τ)
        append_col!(L, L_col, k, τ)

        # Add the new row and column to U_nonzero_col, L_nonzero_row, U_first, L_first
        # (First index *after* the diagonal)
        U_first[k] = U.colptr[k] + 1
        if U.colptr[k] != U.colptr[k + 1] - 1
            push!(U_nonzero_col[U.rowval[U_first[k]]], k)
        end

        L_first[k] = L.colptr[k] + 1
        if L.colptr[k] != L.colptr[k + 1] - 1
            push!(L_nonzero_row[L.rowval[L_first[k]]], k)
        end

        ##
        ## Clean up for next step
        ##

        empty!(L_col)
        empty!(U_row)
        empty!(U_nonzero_col[k])
        empty!(L_nonzero_row[k])
    end

    L, U
end

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
    end

    L, U
end
end
