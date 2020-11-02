import SparseArrays: nnz
import LinearAlgebra: ldiv!
import Base.\

export forward_substitution!, backward_substitution!
"""
Returns the number of nonzeros of the `L` and `U`
factor combined.

Excludes the unit diagonal of the `L` factor,
which is not stored.
"""
nnz(F::ILUFactorization) = nnz(F.L) + nnz(F.U)

function ldiv!(F::ILUFactorization, y::AbstractVector)
    forward_substitution!(F, y)
    backward_substitution!(F, y)
end

function ldiv!(y::AbstractVector, F::ILUFactorization, x::AbstractVector)
    y .= x
    ldiv!(F, y)
end

(\)(F::ILUFactorization, y::AbstractVector) = ldiv!(F, copy(y))

"""
Applies in-place backward substitution with the U factor of F, under the assumptions:

1. U is stored transposed / row-wise
2. U has no lower-triangular elements stored
3. U has (nonzero) diagonal elements stored.
"""
function backward_substitution!(F::ILUFactorization, y::AbstractVector)
    U = F.U
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

function backward_substitution!(v::AbstractVector, F::ILUFactorization, y::AbstractVector)
    v .= y
    backward_substitution!(F, v)
end

"""
Applies in-place forward substitution with the L factor of F, under the assumptions:

1. L is stored column-wise (unlike U)
2. L has no upper triangular elements
3. L has *no* diagonal elements
"""
function forward_substitution!(F::ILUFactorization, y::AbstractVector)
    L = F.L
    @inbounds for col = 1 : L.n - 1
        for idx = L.colptr[col] : L.colptr[col + 1] - 1
            y[L.rowval[idx]] -= L.nzval[idx] * y[col]
        end
    end

    y
end

function forward_substitution!(v::AbstractVector, F::ILUFactorization, y::AbstractVector)
    v .= y
    forward_substitution!(F, v)
end
