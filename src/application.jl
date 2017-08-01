import Base: A_ldiv_B!

function A_ldiv_B!(F::ILUFactorization, y::AbstractVector)
    transposed_backward_substitution!(F.U, y)
    forward_substitution_without_diag!(F.L, y)
end

"""
Applies in-place backward substitution with the U factor, under the assumptions: 

1. U is stored transposed / row-wise
2. U has no lower-triangular elements stored
3. U has (nonzero) diagonal elements stored.
"""
function transposed_backward_substitution!(U::SparseMatrixCSC, y::AbstractVector)
    @inbounds for col = U.n : -1 : 1

        # Substitutions
        for idx = U.colptr[col + 1] - 1 : -1 : U.colptr[col] + 1
            y[col] -= U.nzval[idx] * y[U.rowval[idx]]
        end

        # Final answer for y[col]
        y[col] /= U.nzval[U.colptr[col]]
    end

    y
end

"""
Applies in-place forward substitution with the L factor, under the assumptions:

1. L is stored column-wise (unlike U)
2. L has no upper triangular elements
3. L has *no* diagonal elements
"""
function forward_substitution_without_diag!(L::SparseMatrixCSC, y::AbstractVector)
    @inbounds for col = 1 : L.n - 1
        for idx = L.colptr[col] : L.colptr[col + 1] - 1
            y[L.rowval[idx]] -= L.nzval[idx] * y[col]
        end
    end

    y
end