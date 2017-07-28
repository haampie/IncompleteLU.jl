# SparseMatrixCSC → (Crout) ILU

ILUC loops as follows:

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

  # drop stuff in row & col
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
| x | x | . | . | . | . | . | . | k
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

In the decomposition phase it makes sense to store U row-wise and L column-wise. In application `y = U \ (L \ x)` it makes more sense to have L row-wise as well.

There are multiple issues at each step `k`:

1. Looping over the non-zero entries in the `k`th row of L and the `k`th column of U.
2. Updating `row` means having access to non-zeros of `U` at column-index `k` and larger at each row `< k`; updating `col` mean having access to non-zeros of `L` at row-index `k` and larger at each column `< k`.
3. Computing `col` and `row` should go (more or less) sparse, yet they are linear combinations of vectors with a different sparsity patterns.
4. Loading row `k` of `A`, while `A` is stored column-wise.

Issue (3) is a problem in general ILU decompositions and solved in Saad's book.

Issue (1) and (2) seem to be solved using a priority queue:

1. We need to keep track of the first non-zero in each row of U after column `k`, in order to update `row`.
2. Set up a min-PQ with `key = row` and `priority = column`
3. At step `k`, when updating `col`, peek/pop until `priority` becomes > column `k`. This gives you the non-zeros in the `k`'th column of U.
4. But rather than actually *popping* we need to *update* the priority to the next non-zero in the row (next non-zero is available because U is stored row-wise)
5. Equivalently, set up a min-PQ with `key = column` and `priority = row` for the L-factor.

**Subtle issue**: the U-factor updates the diagonal element, the L-factor does not. Popping in the U-prio-queue makes it impossible to update the diagonal element. Therefore pop the L-prio-queue first to compute the new row, then pop the U-prio-queue to compute the new column.

## Prio-queue example

```julia
A = sprand(5, 5, .3) + I
full(A)
ILU.csc_by_row_pq(A)
```

Outputs

```
5×5 Array{Float64,2}:
 1.15505   0.0  0.0  0.0       0.0
 0.0       1.0  0.0  0.362133  0.0
 0.0       0.0  1.0  0.0       0.0
 0.0       0.0  0.0  1.0       0.0
 0.476451  0.0  0.0  0.0       1.0

1.1550452695900866 (1)
1.0 (2) 0.36213344765906164 (4)
1.0 (3)
1.0 (4)
0.47645076828880906 (1) 1.0 (5)
```