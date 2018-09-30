import SparseArrays: nnz
import LinearAlgebra: ldiv!
import Base.\


"""
Returns the number of nonzeros of the `L` and `U`
factor combined.

Excludes the unit diagonal of the `L` factor,
which is not stored.
"""
nnz(F::ILUFactorization) = nnz(F.L) + nnz(F.U)

function ldiv!(F::ILUFactorization, y::AbstractVector)
    forward_substitution_without_diag!(F.L, y)
    transposed_backward_substitution!(F.U, y)
end

(\)(F::ILUFactorization, y::AbstractVector) = ldiv!(F, copy(y))

function ldiv!(y::AbstractVector, F::ILUFactorization, x::AbstractVector)
    copyto!(y, x)
    ldiv!(F, y)
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