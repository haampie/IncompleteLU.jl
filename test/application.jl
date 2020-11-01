using Test
using IncompleteLU: ILUFactorization, forward_substitution!, backward_substitution!
using LinearAlgebra

@testset "Forward and backward substitutions" begin
    function test_fw_substitution(F::ILUFactorization)
        A = F.L
        n = size(A, 1)
        x = rand(n)
        y = copy(x)
        v = zeros(n)

        forward_substitution!(v, F, x)
        forward_substitution!(F, x)
        ldiv!(LowerTriangular(A + I), y)

        @test v ≈ y
        @test x ≈ y
    end

    function test_bw_substitution(F::ILUFactorization)
        A = F.U
        n = size(A, 1)
        x = rand(n)
        y = copy(x)
        v = zeros(n)

        backward_substitution!(v, F, x)
        backward_substitution!(F, x)
        ldiv!(UpperTriangular(transpose(A)), y)

        @test v ≈ y
        @test x ≈ y
    end

    L = sparse(tril(rand(10, 10), -1))
    U = sparse(tril(rand(10, 10)) + 10I)
    F = ILUFactorization(L, U)
    test_fw_substitution(F)
    test_bw_substitution(F)

    L = sparse(tril(tril(sprand(10, 10, .5), -1)))
    U = sparse(tril(sprand(10, 10, .5) + 10I))
    F = ILUFactorization(L, U)
    test_fw_substitution(F)
    test_bw_substitution(F)

    L = spzeros(10, 10)
    U = spzeros(10, 10) + 10I
    F = ILUFactorization(L, U)
    test_fw_substitution(F)
    test_bw_substitution(F)
end

@testset "ldiv!" begin
    function test_ldiv!(L, U)
        LU = ILUFactorization(L, U)
        x = rand(size(LU.L, 1))
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
end

@testset "nnz" begin
    L = tril(sprand(10, 10, .5), -1)
    U = tril(sprand(10, 10, .5)) + 10I
    LU = ILUFactorization(L, U)
    @test nnz(LU) == nnz(L) + nnz(U)
end