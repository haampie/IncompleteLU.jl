using Test
using IncompleteLU
using SparseArrays
using LinearAlgebra

@testset "Crout ILU" begin
    let
        # Test if it performs full LU if droptol is zero
        A = sprand(10, 10, .5) + 10I
        ilu = IncompleteLU.ilu(A, τ = 0.0)
        flu = lu(Matrix(A), Val(false))

        @test Matrix(ilu.L + I) ≈ flu.L
        @test Matrix(transpose(ilu.U)) ≈ flu.U
    end

    let
        # Test if L = I and U = diag(A) when the droptol is large.
        A = sprand(10, 10, .5) + 10I
        ilu = IncompleteLU.ilu(A, τ = 1.0)

        @test nnz(ilu.L) == 0
        @test nnz(ilu.U) == 10
        @test diag(ilu.U) == diag(A)
    end

end