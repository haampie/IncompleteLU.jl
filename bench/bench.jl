module Bench

using ILU
using BenchmarkTools

function go()
    A = sprand(1_000, 1_000, 10 / 1_000) + 15I
    LU = crout_ilu(A)
    Profile.clear_malloc_data()
    @inbounds LU = crout_ilu(A)
end

function sum_values_row_wise(A::SparseMatrixCSC)
    n = size(A, 1)
    reader = RowReader(A)
    sum = 0.0

    for row = 1 : n
        column = first_in_row(reader, row)

        while is_column(column)
            sum += nzval(reader, column)
            column = next_column!(reader, column)
        end
    end

    sum
end

function sum_values_column_wise(A::SparseMatrixCSC)
    n = size(A, 1)
    sum = 0.0

    for col = 1 : n
        for idx = A.colptr[col] : A.colptr[col + 1] - 1
            sum += A.nzval[idx]
        end
    end

    sum
end

function bench_alloc()
    A = sprand(1_000, 1_000, 10 / 1_000) + 15I
    sum_values_row_wise(A)
    Profile.clear_malloc_data()
    sum_values_row_wise(A)
end

function bench_perf()
    A = sprand(10_000, 10_000, 10 / 100_000) + 15I

    @show sum_values_row_wise(A)
    @show sum_values_column_wise(A)

    fst = @benchmark Bench.sum_values_row_wise($A)
    snd = @benchmark Bench.sum_values_column_wise($A)

    fst, snd
end

end