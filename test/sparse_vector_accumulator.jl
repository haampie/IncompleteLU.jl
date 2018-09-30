using IncompleteLU: SparseVectorAccumulator, add!, append_col!, isoccupied
using LinearAlgebra

@testset "SparseVectorAccumulator" begin
    @testset "Initialization" begin
        v = SparseVectorAccumulator{Float64}(10)
        @test iszero(v.nnz)
        @test iszero(v.occupied)
    end

    @testset "Add to SparseVectorAccumulator" begin
        v = SparseVectorAccumulator{Float64}(3)
        add!(v, 1.0, 3)
        add!(v, 1.0, 3)
        add!(v, 3.0, 2)
        @test v.nnz == 2
        @test isoccupied(v, 1) == false
        @test isoccupied(v, 2)
        @test isoccupied(v, 3)
        @test Vector(v) == [0.; 3.0; 2.0]
    end

    @testset "Add column of SparseMatrixCSC" begin
        # Copy all columns of a 
        v = SparseVectorAccumulator{Float64}(5)
        A = sprand(5, 5, 1.0)
        axpy!(2., A, 3, A.colptr[3], v)
        axpy!(3., A, 4, A.colptr[4], v)
        @test Vector(v) == 2 * A[:, 3] + 3 * A[:, 4]
    end

    @testset "Append column to SparseMatrixCSC" begin
        A = spzeros(5, 5)
        v = SparseVectorAccumulator{Float64}(5)

        add!(v, 0.3, 1)
        add!(v, 0.009, 3)
        add!(v, 0.12, 4)
        add!(v, 0.007, 5)
        append_col!(A, v, 1, 0.1)

        # Test whether the column is copied correctly
        # and the dropping rule is applied
        @test A[1, 1] == 0.3
        @test A[2, 1] == 0.0 # zero
        @test A[3, 1] == 0.0 # dropped
        @test A[4, 1] == 0.12
        @test A[5, 1] == 0.0 # dropped

        # Test whether the InsertableSparseVector is reset
        # when reusing it for the second column. Also do
        # scaling with a factor of 10.
        add!(v, 0.5, 2)
        add!(v, 0.009, 3)
        add!(v, 0.5, 4)
        add!(v, 0.007, 5)
        append_col!(A, v, 2, 0.1, 10.0)

        @test A[1, 2] == 0.0 # zero
        @test A[2, 2] == 5.0 # scaled
        @test A[3, 2] == 0.0 # dropped
        @test A[4, 2] == 5.0 # scaled
        @test A[5, 2] == 0.0 # dropped
    end
end