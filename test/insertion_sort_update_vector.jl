using Base.Test
import ILU: InsertableSparseVector, add!, axpy!, append_col!

@testset "Insertion sorted sparse vector" begin
    v = InsertableSparseVector{Float64}(10)

    add!(v, 3.0, 6, 11)
    add!(v, 3.0, 3, 11)
    add!(v, 3.0, 3, 11)

    @test v[6] == 3.0
    @test v[3] == 6.0
    @test indices(v) == [3, 6]
end

@testset "Add column of SparseMatrixCSC" begin
    v = InsertableSparseVector{Float64}(5)
    A = sprand(5, 5, 1.0)
    axpy!(2., A, 3, A.colptr[3], v)
    axpy!(3., A, 4, A.colptr[4], v)
    @test convert(Vector, v) == 2 * A[:, 3] + 3 * A[:, 4]
end

@testset "Append column to SparseMatrixCSC" begin
    v = InsertableSparseVector{Float64}(5)
    add!(v, 0.3, 1, 6)
    add!(v, 0.009, 3, 6)
    add!(v, 0.12, 4, 6)
    add!(v, 0.007, 5, 6)

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