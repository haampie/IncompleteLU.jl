import Base.Order.Ordering
import Base.Order.lt

export crout_ilu

function crout_ilu(A::SparseMatrixCSC{T,I}; τ = 1e-3) where {T,I}
    n = size(A, 1)

    A_reader = RowReader(A)

    L = spzeros(T, n, n)
    U = spzeros(T, n, n)
    
    U_row = SparseVectorAccumulator{T}(n)
    L_col = SparseVectorAccumulator{T}(n)

    L_reader = RowReader(L, Val{false})

    # U_first[i] is the index of the first nonzero of U in row i after the diagonal
    # L_first[i] is the index of the first nonzero of L in column i after the diagonal
    U_first = zeros(Int, n)

    # U_nonzero_col[i] is a vector of indices of nonzero elements in the i'th column of U before the diagonal
    # L_nonzero_row[i] is a vector of indices of nonzero elements in the i'th row of L before the diagonal
    U_nonzero_col = Vector{Vector{Int}}(n)

    # Initialization
    for i = 1 : n
        U_nonzero_col[i] = Vector{Int}()
    end

    for k = 1 : n

        ##
        ## Copy the new row into U_row and the new column into L_col
        ##

        col = first_in_row(A_reader, k)

        while is_column(col)
            add!(U_row, nzval(A_reader, col), col)
            next_col = next_column(A_reader, col)
            next_row!(A_reader, col)

            # Check if the next nonzero in this column
            # is still above the diagonal
            if has_next_nonzero(A_reader, col) && nzrow(A_reader, col) ≤ col
                enqueue_next_nonzero!(A_reader, col)
            end

            col = next_col
        end

        # Copy the remaining part of the column into L_col
        axpy!(one(T), A, k, nzidx(A_reader, k), L_col)

        ##
        ## Combine the vectors:
        ##
        
        # U_row[k:n] -= L[k,i] * U[i,k:n] for i = 1 : k - 1
        col = first_in_row(L_reader, k)
        
        while is_column(col)
            axpy!(-nzval(L_reader, col), U, col, U_first[col], U_row)

            next_col = next_column(L_reader, col)
            next_row!(L_reader, col)

            if has_next_nonzero(L_reader, col)
                enqueue_next_nonzero!(L_reader, col)
            end

            col = next_col
        end
        
        # Nothing is happening here when k = n, maybe remove?
        # L_col[k+1:n] -= U[i,k] * L[i,k+1:n] for i = 1 : k - 1
        if k < n
            for row = U_nonzero_col[k]
                axpy!(-U.nzval[U_first[row]], L, row, nzidx(L_reader, row), L_col)

                U_first[row] += 1

                # Check if there is still another value in row `row` of U
                # if so, inform U_nonzero_col as well.
                if U_first[row] != U.colptr[row + 1]
                    push!(U_nonzero_col[U.rowval[U_first[row]]], row)
                end
            end
        end
    
        ## 
        ## Apply a drop rule
        ##

        # Append the columns
        append_col!(U, U_row, k, τ)
        append_col!(L, L_col, k, τ)


        U_diag_element = U_row.nzval[U_row.full[k]]

        for i = L.colptr[k] : L.colptr[k + 1] - 1
            L.nzval[i] /= U_diag_element
        end

        # Add the new row and column to U_nonzero_col, L_nonzero_row, U_first, L_first
        # (First index *after* the diagonal)
        U_first[k] = U.colptr[k] + 1
        if U.colptr[k] < U.colptr[k + 1] - 1
            push!(U_nonzero_col[U.rowval[U_first[k]]], k)
        end

        L_reader.next_in_column[k] = L.colptr[k]
        if L.colptr[k] < L.colptr[k + 1]
            enqueue_next_nonzero!(L_reader, k)
        end

        ##
        ## Clean up for next step
        ##

        empty!(L_col)
        empty!(U_row)
        empty!(U_nonzero_col[k])
    end

    ILUFactorization(L, U)
end
