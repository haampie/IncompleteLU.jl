using Test
using IncompleteLU: LinkedLists, RowReader, first_in_row, is_column, nzval, next_column, 
                    next_row!, has_next_nonzero, enqueue_next_nonzero!
using SparseArrays

@testset "Linked List" begin
    n = 5
    let
        lists = LinkedLists(n)

        # head[2] -> 5 -> nil
        # head[5] -> 4 -> 3 -> nil
        push!(lists, 5, 3)
        push!(lists, 5, 4)
        push!(lists, 2, 5)

        @test lists.head[5] == 4
        @test lists.next[4] == 3
        @test lists.next[3] == 0

        @test lists.head[2] == 5
        @test lists.next[5] == 0
    end
end

@testset "Read SparseMatrixCSC row by row" begin
    # Read a sparse matrix row by row.
    n = 10
    A = sprand(n, n, .5)
    reader = RowReader(A)

    for row = 1 : n
        column = first_in_row(reader, row)

        while is_column(column)
            @test nzval(reader, column) == A[row, column]

            next_col = next_column(reader, column)
            next_row!(reader, column)

            if has_next_nonzero(reader, column)
                enqueue_next_nonzero!(reader, column)
            end

            column = next_col
        end
    end
end