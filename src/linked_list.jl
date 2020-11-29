import Base: push!

"""
The factor L is stored column-wise, but we need
all nonzeros in row `row`. We already keep track of 
the first nonzero in each column (at most `n` indices).
Take `l = LinkedLists(n)`. Let `l.head[row]` be the column
of some nonzero in row `row`. Then we can store the column
of the next nonzero of row `row` in `l.next[l.head[row]]`, etc.
That "spot" is empty and there will never be a conflict
because as long as we only store the first nonzero per column: 
the column is then a unique identifier.
"""
struct LinkedLists{Ti}
    head::Vector{Ti}
    next::Vector{Ti}
end

LinkedLists{Ti}(n::Integer) where {Ti} = LinkedLists(zeros(Ti, n), zeros(Ti, n))

"""
For the L-factor: insert in row `head` column `value`
For the U-factor: insert in column `head` row `value`
"""
@propagate_inbounds function push!(l::LinkedLists, head::Integer, value::Integer)
    l.head[head], l.next[value] = value, l.head[head]
    return l
end

struct RowReader{Tv,Ti}
    A::SparseMatrixCSC{Tv,Ti}
    next_in_column::Vector{Ti}
    rows::LinkedLists{Ti}
end

function RowReader(A::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    n = size(A, 2)
    @inbounds next_in_column = [A.colptr[i] for i = 1 : n]
    rows = LinkedLists{Ti}(n)
    @inbounds for i = Ti(1) : Ti(n)
        push!(rows, A.rowval[A.colptr[i]], i)
    end
    return RowReader(A, next_in_column, rows)
end

function RowReader(A::SparseMatrixCSC{Tv,Ti}, initialize::Type{Val{false}}) where {Tv,Ti}
    n = size(A, 2)
    return RowReader(A, zeros(Ti, n), LinkedLists{Ti}(n))
end

@propagate_inbounds nzidx(r::RowReader, column::Integer) = r.next_in_column[column]
@propagate_inbounds nzrow(r::RowReader, column::Integer) = r.A.rowval[nzidx(r, column)]
@propagate_inbounds nzval(r::RowReader, column::Integer) = r.A.nzval[nzidx(r, column)]

@propagate_inbounds has_next_nonzero(r::RowReader, column::Integer) = nzidx(r, column) < r.A.colptr[column + 1]
@propagate_inbounds enqueue_next_nonzero!(r::RowReader, column::Integer) = push!(r.rows, nzrow(r, column), column)
@propagate_inbounds next_column(r::RowReader, column::Integer) = r.rows.next[column]
@propagate_inbounds first_in_row(r::RowReader, row::Integer) = r.rows.head[row]
@propagate_inbounds is_column(column::Integer) = column != 0
@propagate_inbounds next_row!(r::RowReader, column::Integer) = r.next_in_column[column] += 1

