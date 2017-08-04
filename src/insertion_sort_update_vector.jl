import Base: getindex, setindex!, indices

struct InsertableSparseVector{Tv,Ti <: Integer}
    values::Vector{Tv}
    indices::SortedSet{Ti}

    InsertableSparseVector{Tv,Ti}(n::Ti) where {Ti <: Integer,Tv} = new(Vector{Tv}(n), SortedSet(n))
end

InsertableSparseVector{Tv}(n::Ti) where {Ti <: Integer,Tv} = InsertableSparseVector{Tv,Ti}(n)

@inline getindex(v::InsertableSparseVector{Tv,Ti}, idx::Ti) where {Tv,Ti} = v.values[idx]
@inline setindex!(v::InsertableSparseVector{Tv,Ti}, value::Tv, idx::Ti) where {Tv,Ti} = v.values[idx] = value
@inline indices(v::InsertableSparseVector) = convert(Vector, v.indices)

@inline function add!(v::InsertableSparseVector{Tv,Ti}, a::Tv, idx::Ti) where {Tv,Ti}
    if push!(v.indices, idx)
        v[idx] = a
    else
        v[idx] += a
    end

    v
end