using Test
using IncompleteLU
using SparseArrays
using LinearAlgebra

@testset "Crout ILU" for Tv in (Float64, Float32, ComplexF64, ComplexF32), Ti in (Int64, Int32)
    let
        # Test if it performs full LU if droptol is zero
        A = convert(SparseMatrixCSC{Tv, Ti}, sprand(Tv, 10, 10, .5) + 10I)
        ilu = IncompleteLU.ilu(A, τ = 0)
        flu = lu(Matrix(A), Val(false))

        @test typeof(ilu) == IncompleteLU.ILUFactorization{Tv,Ti}
        @test Matrix(ilu.L + I) ≈ flu.L
        @test Matrix(transpose(ilu.U)) ≈ flu.U
    end

    let
        # Test if L = I and U = diag(A) when the droptol is large.
        A = convert(SparseMatrixCSC{Tv, Ti}, sprand(10, 10, .5) + 10I)
        ilu = IncompleteLU.ilu(A, τ = 1.0)

        @test nnz(ilu.L) == 0
        @test nnz(ilu.U) == 10
        @test diag(ilu.U) == diag(A)
    end
end

@testset "Crout ILU with integer matrix" begin
    A = sparse(Int32(1):Int32(10), Int32(1):Int32(10), 1)
    ilu = IncompleteLU.ilu(A, τ = 0)

    @test typeof(ilu) == IncompleteLU.ILUFactorization{Float64,Int32}
    @test nnz(ilu.L) == 0
    @test diag(ilu.U) == diag(A)
end