using Base.Test
import ILU: InsertableSparseVector, add!

@testset "Insertion sorted sparse vector" begin
    v = InsertableSparseVector{Float64}(10)

    add!(v, 3.0, 6)
    add!(v, 3.0, 3)
    add!(v, 3.0, 3)

    @test v[6] == 3.0
    @test v[3] == 6.0
    @test indices(v) == [3, 6]
end