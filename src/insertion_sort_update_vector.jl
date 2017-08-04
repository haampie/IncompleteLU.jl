import Base: getindex, setindex!, indices, convert

struct InsertableSparseVector{Tv}
    values::Vector{Tv}
    indices::SortedSet
    n::Int

    InsertableSparseVector{Tv}(n::Int) where {Tv} = new(Vector{Tv}(n), SortedSet(n), n)
end

@inline getindex(v::InsertableSparseVector{Tv}, idx::Int) where {Tv} = v.values[idx]
@inline setindex!(v::InsertableSparseVector{Tv}, value::Tv, idx::Int) where {Tv} = v.values[idx] = value
@inline indices(v::InsertableSparseVector) = convert(Vector, v.indices)

function convert(::Type{Vector}, v::InsertableSparseVector{Tv}) where {Tv}
    vals = zeros(Tv, v.n)
    for index in v.indices
        vals[index] = v.values[index]
    end
    return vals
end

function add!(v::InsertableSparseVector{Tv}, a::Tv, idx::Int, prev_idx::Int) where {Tv}
    if push!(v.indices, idx, prev_idx)
        v[idx] = a
    else
        v[idx] += a
    end

    v
end

function axpy!(a::Tv, A::SparseMatrixCSC{Tv}, column::Int, start::Int, y::InsertableSparseVector{Tv}) where {Tv}
    prev_index = y.n + 1
    
    @inbounds for idx = start : A.colptr[column + 1] - 1
        add!(y, a * A.nzval[idx], A.rowval[idx], prev_index)
        prev_index = A.rowval[idx]
    end

    y
end

function append_col!(A::SparseMatrixCSC{Tv}, y::InsertableSparseVector{Tv}, j::Int, drop = zero(real(Tv))) where {Tv}
    
    total = 0
    
    for row = y.indices
        if abs(y[row]) ≥ drop || row == j
            push!(A.rowval, row)
            push!(A.nzval, y[row])
            total += 1
        end
    end

    A.colptr[j + 1] = A.colptr[j] + total
    
    # Reset
    y.indices[y.n + 1] = y.n + 1

    nothing
end