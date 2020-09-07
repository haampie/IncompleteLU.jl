using Test
using IncompleteLU: ILUFactorization, forward_substitution_without_diag!, transposed_backward_substitution!
using LinearAlgebra

@testset "Backward substitution" begin
    function test_bw_substitution(A::SparseMatrixCSC)
        x = rand(eltype(A),size(A, 1))
        y = copy(x)

        forward_substitution_without_diag!(A, x)
        ldiv!(LowerTriangular(A + I), y)

        @test x ≈ y
    end

    test_bw_substitution(sparse(tril(rand(10, 10), -1)))
    test_bw_substitution(tril(sprand(10, 10, .5), -1))
    test_bw_substitution(spzeros(10, 10))

    test_bw_substitution(sparse(tril(rand(ComplexF64,10, 10), -1)))
    test_bw_substitution(tril(sprand(ComplexF64,10, 10, .5), -1))
    test_bw_substitution(spzeros(ComplexF64,10, 10))
end

@testset "Forward substitution" begin
    function test_fw_substitution(A::SparseMatrixCSC)
        x = rand(eltype(A),size(A, 1))
        y = copy(x)

        transposed_backward_substitution!(A, x)
        ldiv!(UpperTriangular(transpose(A)), y)

        @test x ≈ y
    end

    test_fw_substitution(sparse(tril(rand(10, 10)) + 10I))
    test_fw_substitution(tril(sprand(10, 10, .5) + 10I))
    test_fw_substitution(spzeros(10, 10) + 10I)


    test_fw_substitution(sparse(tril(rand(ComplexF64,10, 10)) + 10I))
    test_fw_substitution(tril(sprand(ComplexF64,10, 10, .5) + 10I))
    test_fw_substitution(spzeros(ComplexF64,10, 10) + 10I)
end

@testset "ldiv!" begin
    function test_ldiv!(L, U)
        LU = ILUFactorization(L, U)
        x = rand(eltype(L),size(LU.L, 1))
        y = copy(x)
        z = copy(x)
        w = copy(x)

        ldiv!(LU, x)
        ldiv!(LowerTriangular(LU.L + I), y)
        ldiv!(UpperTriangular(transpose(LU.U)), y)

        @test x ≈ y
        @test LU \ z == x

        ldiv!(w, LU, z)

        @test w == x
    end

    test_ldiv!(tril(sprand(10, 10, .5), -1), tril(sprand(10, 10, .5) + 10I))
    test_ldiv!(tril(sprand(ComplexF64,10, 10, .5), -1), tril(sprand(ComplexF64,10, 10, .5) + 10I))
end

@testset "nnz" begin
    let
        L = tril(sprand(10, 10, .5), -1)
        U = tril(sprand(10, 10, .5)) + 10I
        LU = ILUFactorization(L, U)
        @test nnz(LU) == nnz(L) + nnz(U)
    end
    let
        L = tril(sprand(ComplexF64,10, 10, .5), -1)
        U = tril(sprand(ComplexF64,10, 10, .5)) + 10I
        LU = ILUFactorization(L, U)
        @test nnz(LU) == nnz(L) + nnz(U)
    end
end