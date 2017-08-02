using ILU
import ILU: SparseVectorAccumulator, add!, axpy!

@testset "SparseVectorAccumulator" begin
    let
        v = SparseVectorAccumulator{Float64}(10)
        @test iszero(v.n)
        @test iszero(v.full)
        @test isempty(v.nzval)
        @test isempty(v.nzind)
    end

    let 
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

    let 
        # Copy all columns of a 
        v = SparseVectorAccumulator{Float64}(5)
        A = sprand(5, 5, 1.0)
        axpy!(2., A, 3, A.colptr[3], v)
        axpy!(3., A, 4, A.colptr[4], v)
        @test convert(Vector, v) == 2 * A[:, 3] + 3 * A[:, 4]
    end

    let # Test emptying
        v = SparseVectorAccumulator{Float64}(3)
        add!(v, 1.0, 3)
        empty!(v)
        @test v.n == 0
        @test iszero(v.full)
        
        add!(v, 1.0, 2)
        @test v.n == 1
        @test convert(Vector, v) == [0.; 1.; 0.0]
    end
end