using ILU
import ILU: SparseVectorAccumulator, add!, axpy!

@testset "SparseVectorAccumulator" begin
    let
        v = SparseVectorAccumulator{Float64}(10)
        @test iszero(v.n)
        @test iszero(v.full)
    end

    @testset "Add to SparseVectorAccumulator" begin
        v = SparseVectorAccumulator{Float64}(3)
        add!(v, 1.0, 3)
        add!(v, 1.0, 3)
        add!(v, 3.0, 2)
        @test v.n == 2
        @test v.full[3] == 1
        @test v.full[2] == 2
        @test v.nzind[1] == 3
        @test v.nzind[2] == 2
        @test v.nzval[1] == 2.0
        @test v.nzval[2] == 3.0
        @test convert(Vector, v) == [0; 3.0; 2.0]
    end

    @testset "Add column of SparseMatrixCSC" begin
        # Copy all columns of a 
        v = SparseVectorAccumulator{Float64}(5)
        A = sprand(5, 5, 1.0)
        axpy!(2., A, 3, A.colptr[3], v)
        axpy!(3., A, 4, A.colptr[4], v)
        @test convert(Vector, v) == 2 * A[:, 3] + 3 * A[:, 4]
    end

    @testset "Append column to SparseMatrixCSC" begin
        v = SparseVectorAccumulator{Float64}(5)
        add!(v, 0.3, 1)
        add!(v, 0.009, 3)
        add!(v, 0.12, 4)
        add!(v, 0.007, 5)

        A = spzeros(5, 5)
        append_col!(A, v, 1, 0.1)

        # Test whether the column is copied correctly
        # and the dropping rule is applied
        @test A[1, 1] == 0.3
        @test A[2, 1] == 0.0
        @test A[3, 1] == 0.0
        @test A[4, 1] == 0.12
        @test A[5, 1] == 0.0

        # Test whether the InsertableSparseVector is reset
        @test iszero(convert(Vector, v))
    end
end