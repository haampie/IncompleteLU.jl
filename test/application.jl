using Base.Test
using ILU
import ILU: forward_substitution_without_diag!, transposed_backward_substitution!

@testset "Backward substitution" begin
    function test_bw_substitution(A::SparseMatrixCSC)
        x = rand(size(A, 1))
        y = copy(x)

        forward_substitution_without_diag!(A, x)
        A_ldiv_B!(LowerTriangular(A + I), y)

        @test x ≈ y
    end

    test_bw_substitution(sparse(tril(rand(10, 10), -1)))
    test_bw_substitution(tril(sprand(10, 10, .5), -1))
    test_bw_substitution(spzeros(10, 10))
end

@testset "Forward substitution" begin
    function test_fw_substitution(A::SparseMatrixCSC)
        x = rand(size(A, 1))
        y = copy(x)

        transposed_backward_substitution!(A, x)
        A_ldiv_B!(UpperTriangular(A.'), y)

        @test x ≈ y
    end

    test_fw_substitution(sparse(tril(rand(10, 10)) + 10I))
    test_fw_substitution(tril(sprand(10, 10, .5) + 10I))
    test_fw_substitution(spzeros(10, 10) + 10I)
end

@testset "A_ldiv_B!" begin
    function test_A_ldiv_B!(L, U)
        LU = ILUFactorization(L, U)
        x = rand(size(LU.L, 1))
        y = copy(x)

        A_ldiv_B!(LU, x)

        A_ldiv_B!(UpperTriangular(LU.U.'), y)
        A_ldiv_B!(LowerTriangular(LU.L + I), y)

        @test x ≈ y
    end

    test_A_ldiv_B!(tril(sprand(10, 10, .5), -1), tril(sprand(10, 10, .5) + 10I))
end