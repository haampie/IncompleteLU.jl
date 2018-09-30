module Bench

using IncompleteLU
using BenchmarkTools
using SparseArrays
using LinearAlgebra: axpy!, I
using Random: seed!
using Profile

function go()
    seed!(1)
    A = sprand(10_000, 10_000, 10 / 10_000) + 15I
    LU = ilu(A)
    Profile.clear_malloc_data()
    @profile ilu(A)
end

function axpy_perf()
    A = sprand(1_000, 1_000, 10 / 1_000) + 15I
    y = SparseVectorAccumulator{Float64}(1_000)

    axpy!(1.0, A, 1, A.colptr[1], y)
    axpy!(1.0, A, 2, A.colptr[2], y)
    axpy!(1.0, A, 3, A.colptr[3], y)

    Profile.clear_malloc_data()

    axpy!(1.0, A, 1, A.colptr[1], y)
    axpy!(1.0, A, 2, A.colptr[2], y)
    axpy!(1.0, A, 3, A.colptr[3], y)
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
    A = sprand(10_000, 10_000, 10 / 10_000) + 15I

    @show sum_values_row_wise(A)
    @show sum_values_column_wise(A)

    fst = @benchmark Bench.sum_values_row_wise($A)
    snd = @benchmark Bench.sum_values_column_wise($A)

    fst, snd
end

function bench_ILU()
    seed!(1)
    A = sprand(10_000, 10_000, 10 / 10_000) + 15I
    LU = ilu(A, τ = 0.1)
    
    @show nnz(LU.L) nnz(LU.U)
    # nnz(LU.L) = 44836
    # nnz(LU.U) = 54827

    result = @benchmark ilu($A, τ = 0.1)
    # BenchmarkTools.Trial:
    #   memory estimate:  16.24 MiB
    #   allocs estimate:  545238
    #   --------------
    #   minimum time:     116.923 ms (0.00% GC)
    #   median time:      127.514 ms (2.18% GC)
    #   mean time:        130.932 ms (1.75% GC)
    #   maximum time:     166.202 ms (3.05% GC)
    #   --------------
    #   samples:          39
    #   evals/sample:     1

    # After switching to row reader.
    #     BenchmarkTools.Trial:
    #   memory estimate:  15.96 MiB
    #   allocs estimate:  545222
    #   --------------
    #   minimum time:     55.264 ms (0.00% GC)
    #   median time:      61.872 ms (4.73% GC)
    #   mean time:        61.906 ms (3.72% GC)
    #   maximum time:     74.615 ms (4.12% GC)
    #   --------------
    #   samples:          81
    #   evals/sample:     1

    # After skipping off-diagonal elements in A
    #     BenchmarkTools.Trial:
    #   memory estimate:  15.96 MiB
    #   allocs estimate:  545222
    #   --------------
    #   minimum time:     51.187 ms (0.00% GC)
    #   median time:      55.767 ms (4.27% GC)
    #   mean time:        56.586 ms (3.50% GC)
    #   maximum time:     72.987 ms (7.53% GC)
    #   --------------
    #   samples:          89
    #   evals/sample:     1

    # After moving L and U to Row Reader structs
    #     BenchmarkTools.Trial:
    #   memory estimate:  13.03 MiB
    #   allocs estimate:  495823
    #   --------------
    #   minimum time:     43.062 ms (0.00% GC)
    #   median time:      46.205 ms (2.83% GC)
    #   mean time:        47.076 ms (1.76% GC)
    #   maximum time:     65.956 ms (1.96% GC)
    #   --------------
    #   samples:          107
    #   evals/sample:     1

    # After emptying the fill-in vecs during copy.
    #     BenchmarkTools.Trial:
    #   memory estimate:  13.03 MiB
    #   allocs estimate:  495823
    #   --------------
    #   minimum time:     41.930 ms (0.00% GC)
    #   median time:      44.583 ms (2.25% GC)
    #   mean time:        45.785 ms (1.38% GC)
    #   maximum time:     66.683 ms (1.59% GC)
    #   --------------
    #   samples:          110
    #   evals/sample:     1
end

end

# Bench.go()