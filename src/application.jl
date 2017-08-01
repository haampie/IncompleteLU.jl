function A_ldiv_B!(F::ILUFactorization, y::AbstractVector)
    transposed_backward_substitution!(F.U, y)
    forward_substitution_without_diag!(F.L, y)
end

function transposed_backward_substitution!(A::SparseMatrixCSC, y::AbstractVector)
    @inbounds for col = A.n : -1 : 1
        for idx = A.colptr[col + 1] - 1 : -1 : A.colptr[col] + 1
            y[col] -= A.nzval[idx] * y[A.rowval[idx]]
        end

        y[col] /= A.nzval[A.colptr[col]]
    end

    y
end

function forward_substitution_without_diag!(A::SparseMatrixCSC, y::AbstractVector)
    @inbounds for col = 1 : A.n - 1
        for idx = A.colptr[col] : A.colptr[col + 1] - 1
            y[A.rowval[idx]] -= A.nzval[idx] * y[col]
        end
    end

    y
end