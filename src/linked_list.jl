import Base: push!, empty!, start, next, done, getindex

export LinkedLists, RowReader, 
       nzval, next_in_row, 
       first_in_row, next_column!, is_column

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
struct LinkedLists
    head::Vector{Int}
    next::Vector{Int}
end

LinkedLists(n::Int) = LinkedLists(zeros(n), zeros(n))

"""
For the L-factor: insert in row `head` column `value`
For the U-factor: insert in column `head` row `value`
"""
function push!(l::LinkedLists, head::Int, value::Int)
    l.head[head], l.next[value] = value, l.head[head]
    return l
end

struct RowReader{matT <: SparseMatrixCSC}
    A::matT
    next_in_column::Vector{Int}
    rows::LinkedLists
end

function RowReader(A::SparseMatrixCSC{T,I}) where {T,I}
    n = size(A, 2)
    next_in_column = [A.colptr[i] for i = 1 : n]
    rows = LinkedLists(n)
    for i = 1 : n
        push!(rows, A.rowval[A.colptr[i]], i)
    end
    return RowReader(A, next_in_column, rows)
end

"""
Once the nonzero in column `column` is visited,
go find the next one.
"""
@inline function set_next_nonzero_in_column!(r::RowReader, column::Int)
    r.next_in_column[column] += 1

    # Push the next nonzero in the linked list in its corresponding row.
    # Only if there is a nonzero left in the column of course.
    if r.next_in_column[column] < r.A.colptr[column + 1]
        push!(r.rows, r.A.rowval[r.next_in_column[column]], column)
    end

    return
end

@inline nzval(r::RowReader, column::Int) = r.A.nzval[r.next_in_column[column]]
@inline next_in_row(r::RowReader, column::Int) = r.rows.next[column]
@inline first_in_row(r::RowReader, row::Int) = r.rows.head[row]
@inline is_column(column::Int) = column != 0

function next_column!(r::RowReader, column::Int)
    next_column = next_in_row(r, column)
    set_next_nonzero_in_column!(r, column)
    next_column
end
