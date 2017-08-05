[![Build Status](https://travis-ci.org/haampie/ILU.jl.svg?branch=master)](https://travis-ci.org/haampie/ILU.jl) [![codecov](https://codecov.io/gh/haampie/ILU.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/haampie/ILU.jl)

# SparseMatrixCSC → (Crout) ILU

The Crout version of ILU loops roughly as follows:

```
for k = 1 : n
  row = zeros(n); row[k:n] = A[k,k:n]
  col = zeros(n); col[k+1:n] = A[k+1:n,k]

  for i = 1 : k - 1 where L[k,i] != 0
    row -= L[k,i] * U[i,k:n]
  end

  for i = 1 : k - 1 where U[i,k] != 0
    col -= U[i,k] * L[k+1:n,i]
  end

  # Apply a dropping rule in row and col

  U[k,:] = row
  L[:,k] = col / U[k,k]
  L[k,k] = 1
end
```

```
          k
+---+---+---+---+---+---+---+---+
| \ |   | x | x | x | x | x | x |
+---+---+---+---+---+---+---+---+
|   | \ | x | x | x | x | x | x |
+---+---+---+---+---+---+---+---+
|   |   | . | . | . | . | . | . | k
+---+---+---+---+---+---+---+---+
| x | x | . | \ |   |   |   |   |
+---+---+---+---+---+---+---+---+
| x | x | . |   | \ |   |   |   |
+---+---+---+---+---+---+---+---+
| x | x | . |   |   | \ |   |   |
+---+---+---+---+---+---+---+---+
| x | x | . |   |   |   | \ |   |
+---+---+---+---+---+---+---+---+
| x | x | . |   |   |   |   | \ |
+---+---+---+---+---+---+---+---+

col and row are the .'s, updated by the x's.
```

At step `k` we load (part of) a row and column of the matrix `A`, and subtract the previous rows and columns. The problem is that our matrix is column-major, so that loading a row is not cheap. Secondly, it makes sense to store the `L` factor column-wise and the `U` factor row-wise, yet we need access to a row of `L` and a column of `U`.

The latter problem can be worked around without expensive searches. It's basically smart bookkeeping: going from step `k` to `k+1` requires updating indices to the next nonzero of each row of `U` after column `k`. If you now store for each column of `U` a list of nonzero indices, this is the moment you can update it. Similarly for the `L` factor.

The matrix `A` can be read row by row as well with the same trick.

## Accumulating a new sparse row or column
At each step a temporary column or row of the `L` and `U` factors is created as a linear combination of previous columns and rows. We don't get these values in sorted order. At this point the fastest structure for keeping track of the indices of the temporary vector is just insertion sort using `InsertableSparseVector`. Another possibility is `SparseVectorAccumulator` which delays sorting until the temporary vector is copied to the `L` or `U` factor.

## Example

Using a drop tolerance of `0.01`, we get a reasonable preconditioner with a bit of fill-in.

```julia
> using ILU
> A = sprand(1000, 1000, 5 / 1000) + 10I
> @time fact = crout_ilu(A, τ = 0.001)
  0.005182 seconds (100 allocations: 1.167 MiB)
> vecnorm((fact.L + I) * fact.U.' - A)
0.05610746209883846
> (nnz(fact.L) + nnz(fact.U)) / nnz(A)
3.670773780187284
```

Full LU is obtained when the drop tolerance is `0.0`.

```julia
>  @time fact = crout_ilu(A, τ = 0.)
  0.400229 seconds (116 allocations: 12.167 MiB, 0.41% gc time)
> vecnorm((fact.L + I) * fact.U.' - A)
1.532520861565543e-13
> (nnz(fact.L) + nnz(fact.U)) / nnz(A)
61.66009528503368
```