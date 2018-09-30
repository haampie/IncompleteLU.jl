using Test

import IncompleteLU: SortedSet, push!

@testset "Sorted indices" begin
    @testset "New values" begin
        indices = SortedSet(10)
        @test push!(indices, 5)
        @test push!(indices, 7)
        @test push!(indices, 4)
        @test push!(indices, 6)
        @test push!(indices, 8)

        as_vec = Vector(indices)
        @test as_vec == [4, 5, 6, 7, 8]
    end

    @testset "Duplicate values" begin
        indices = SortedSet(10)
        @test push!(indices, 3)
        @test push!(indices, 3) == false
        @test push!(indices, 8)
        @test push!(indices, 8) == false
        @test Vector(indices) == [3, 8]
    end

    @testset "Quick insertion with known previous index" begin
        indices = SortedSet(10)
        @test push!(indices, 3)
        @test push!(indices, 4, 3)
        @test push!(indices, 8, 4)
        @test Vector(indices) == [3, 4, 8]
    end

    @testset "Pretty printing" begin
        indices = SortedSet(10)
        push!(indices, 3)
        push!(indices, 2)
        @test occursin("with values", sprint(show, indices))
    end
end