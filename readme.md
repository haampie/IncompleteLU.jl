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

However, there is still the problem of loading a row of `A`. The current solution is to maintain a priority queue which stores the next unvisited element of each column. The key is its column, the value is its `idx` and the ordering is its `rowval`. The priority queue is read until an element of the next row is reached. Rather than popping the min element, its value is incremented, which means that it is replaced by the next non-zero in the same column. Once there are no elements left in the column, it is dequeued.

## Example

Using a drop tolerance of `0.01`, we get a reasonable preconditioner with a bit of fill-in.

```
> using ILU
> A = sprand(1000, 1000, 5 / 1000) + 10I
> L, U = crout_ilu(A, τ = 0.001)
> vecnorm(L * U.' - A)
0.2753999717863117
> (nnz(L - I) + nnz(U)) / nnz(A)
2.7285592497868714
```

Full LU is obtained when the drop tolerance is `0.0`.

```
> L, U = crout_ilu(A, τ = 0.)
> vecnorm(L * U.' - A)
1.484079220395129e-13
> (nnz(L - I) + nnz(U)) / nnz(A)
66.51270247229327
```