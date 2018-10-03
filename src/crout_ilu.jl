export ilu

function ilu(A::SparseMatrixCSC{T,I}; τ = 1e-3) where {T,I}
    n = size(A, 1)

    L = spzeros(T, n, n)
    U = spzeros(T, n, n)
    
    U_row = SparseVectorAccumulator{T}(n)
    L_col = SparseVectorAccumulator{T}(n)

    A_reader = RowReader(A)
    L_reader = RowReader(L, Val{false})
    U_reader = RowReader(U, Val{false})

    @inbounds for k = 1 : n

        ##
        ## Copy the new row into U_row and the new column into L_col
        ##
 
        col::Int = first_in_row(A_reader, k)

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
            axpy!(-nzval(L_reader, col), U, col, nzidx(U_reader, col), U_row)

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
            col = first_in_row(U_reader, k)

            while is_column(col)
                axpy!(-nzval(U_reader, col), L, col, nzidx(L_reader, col), L_col)

                next_col = next_column(U_reader, col)
                next_row!(U_reader, col)

                if has_next_nonzero(U_reader, col)
                    enqueue_next_nonzero!(U_reader, col)
                end

                col = next_col
            end
        end

        ## 
        ## Apply a drop rule
        ##

        U_diag_element = U_row.nzval[k]
        # U_diag_element = U_row.values[k]

        # Append the columns        
        append_col!(U, U_row, k, τ)
        append_col!(L, L_col, k, τ, inv(U_diag_element))

        # Add the new row and column to U_nonzero_col, L_nonzero_row, U_first, L_first
        # (First index *after* the diagonal)
        U_reader.next_in_column[k] = U.colptr[k] + 1
        if U.colptr[k] < U.colptr[k + 1] - 1
            enqueue_next_nonzero!(U_reader, k)
        end

        L_reader.next_in_column[k] = L.colptr[k]
        if L.colptr[k] < L.colptr[k + 1]
            enqueue_next_nonzero!(L_reader, k)
        end
    end

    return ILUFactorization(L, U)
end
