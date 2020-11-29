using IncompleteLU: SparseVectorAccumulator, add!, append_col!, isoccupied
using LinearAlgebra

@testset "SparseVectorAccumulator" for Ti in (Int32, Int64), Tv in (Float64, Float32)
    @testset "Initialization" begin
        v = SparseVectorAccumulator{Tv,Ti}(10)
        @test iszero(v.nnz)
        @test iszero(v.occupied)
    end

    @testset "Add to SparseVectorAccumulator" begin
        v = SparseVectorAccumulator{Tv,Ti}(3)
        add!(v, Tv(1.0), Ti(3))
        add!(v, Tv(1.0), Ti(3))
        add!(v, Tv(3.0), Ti(2))
        @test v.nnz == 2
        @test isoccupied(v, 1) == false
        @test isoccupied(v, 2)
        @test isoccupied(v, 3)
        @test Vector(v) == Tv[0.; 3.0; 2.0]
    end

    @testset "Add column of SparseMatrixCSC" begin
        # Copy all columns of a 
        v = SparseVectorAccumulator{Tv,Ti}(5)
        A = convert(SparseMatrixCSC{Tv,Ti}, sprand(Tv, 5, 5, 1.0))
        axpy!(Tv(2), A, Ti(3), A.colptr[3], v)
        axpy!(Tv(3), A, Ti(4), A.colptr[4], v)
        @test Vector(v) == 2 * A[:, 3] + 3 * A[:, 4]
    end

    @testset "Append column to SparseMatrixCSC" begin
        A = spzeros(Tv, Ti, 5, 5)
        v = SparseVectorAccumulator{Tv,Ti}(5)

        add!(v, Tv(0.3), Ti(1))
        add!(v, Tv(0.009), Ti(3))
        add!(v, Tv(0.12), Ti(4))
        add!(v, Tv(0.007), Ti(5))
        append_col!(A, v, Ti(1), Tv(0.1))

        # Test whether the column is copied correctly
        # and the dropping rule is applied
        @test A[1, 1] == Tv(0.3)
        @test A[2, 1] == Tv(0.0) # zero
        @test A[3, 1] == Tv(0.0) # dropped
        @test A[4, 1] == Tv(0.12)
        @test A[5, 1] == Tv(0.0) # dropped

        # Test whether the InsertableSparseVector is reset
        # when reusing it for the second column. Also do
        # scaling with a factor of 10.
        add!(v, Tv(0.5), Ti(2))
        add!(v, Tv(0.009), Ti(3))
        add!(v, Tv(0.5), Ti(4))
        add!(v, Tv(0.007), Ti(5))
        append_col!(A, v, Ti(2), Tv(0.1), Tv(10.0))

        @test A[1, 2] == Tv(0.0) # zero
        @test A[2, 2] == Tv(5.0) # scaled
        @test A[3, 2] == Tv(0.0) # dropped
        @test A[4, 2] == Tv(5.0) # scaled
        @test A[5, 2] == Tv(0.0) # dropped
    end
end