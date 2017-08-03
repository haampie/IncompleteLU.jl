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

        column = first_in_row(A_reader, k)

        while is_column(column)
            if column ≥ k
                add!(U_row, nzval(A_reader, column), column)
            end

            column = next_column!(A_reader, column)
        end

        # Copy the remaining part of the column into L_col
        axpy!(one(T), A, k, nzidx(A_reader, k), L_col)

        ##
        ## Combine the vectors:
        ##
        
        
        # U_row[k:n] -= L[k,i] * U[i,k:n] for i = 1 : k - 1
        for col = L_nonzero_row[k]
            axpy!(-L.nzval[L_first[col]], U, col, U_first[col], U_row)

            L_first[col] += 1

            # If there is still another value in column `col` of L
            # then add it to L_nonzero_row as well.
            if L_first[col] != L.colptr[col + 1]
                push!(L_nonzero_row[L.rowval[L_first[col]]], col)
            end
        end
        
        # Nothing is happening here when k = n, maybe remove?
        # L_col[k+1:n] -= U[i,k] * L[i,k+1:n] for i = 1 : k - 1
        if k < n
            for row = U_nonzero_col[k]
                axpy!(-U.nzval[U_first[row]], L, row, L_first[row], L_col)

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

        L_first[k] = L.colptr[k]
        if L.colptr[k] < L.colptr[k + 1]
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

    ILUFactorization(L, U)
end
