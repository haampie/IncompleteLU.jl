module BenchInsertion

using IncompleteLU
using BenchmarkTools

function do_stuff(A)
    n = size(A, 1)
    U_row = IncompleteLU.InsertableSparseVector{Float64}(n);
    A_reader = IncompleteLU.RowReader(A)

    for k = 1 : n
        col = first_in_row(A_reader, k)

        while is_column(col)
            IncompleteLU.add!(U_row, nzval(A_reader, col), col, n + 1)
            next_col = next_column(A_reader, col)
            next_row!(A_reader, col)

            if has_next_nonzero(A_reader, col) && nzrow(A_reader, col) ≤ col
                enqueue_next_nonzero!(A_reader, col)
            end

            col = next_col
        end

        U_row.indices[n + 1] = n + 1
    end

    U_row
end

function wut(n = 1_000)
    A = sprand(n, n, 10 / n) + 15I
    @benchmark BenchInsertion.do_stuff($A)
end

function check_allocs(n = 100_000)
    srand(1)
    A = sprand(n, n, 10 / n) + 15I
    do_stuff(A)
    Profile.clear()
    Profile.clear_malloc_data()
    @profile do_stuff(A)
end

end

# BenchInsertion.check_allocs()