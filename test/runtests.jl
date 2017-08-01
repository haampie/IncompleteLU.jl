using Base.Test
using ILU

@testset "Backward substitution" begin
    function test_bw_substitution(A::SparseMatrixCSC)
        x = rand(size(A, 1))
        y = copy(x)

        ILU.forward_substitution_without_diag!(A, x)
        A_ldiv_B!(LowerTriangular(full(A) + I), y)

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

        ILU.transposed_backward_substitution!(A, x)
        A_ldiv_B!(UpperTriangular(full(A.')), y)

        @test x ≈ y
    end

    test_fw_substitution(sparse(tril(rand(10, 10)) + 10I))
    test_fw_substitution(tril(sprand(10, 10, .5) + 10I))
    test_fw_substitution(spzeros(10, 10) + 10I)
end