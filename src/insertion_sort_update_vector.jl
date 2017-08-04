import Base: getindex, setindex!, indices, convert

struct InsertableSparseVector{Tv,Ti<:Integer,N}
    values::Vector{Tv}
    indices::SortedSet{Ti}

    InsertableSparseVector{Tv,Ti,N}() where {Ti <: Integer,Tv,N} = new(Vector{Tv}(N), SortedSet(N))
end

InsertableSparseVector{Tv}(n::Ti) where {Ti <: Integer,Tv} = InsertableSparseVector{Tv,Ti,n}()

@inline getindex(v::InsertableSparseVector{Tv,Ti}, idx::Ti) where {Tv,Ti} = v.values[idx]
@inline setindex!(v::InsertableSparseVector{Tv,Ti}, value::Tv, idx::Ti) where {Tv,Ti} = v.values[idx] = value
@inline indices(v::InsertableSparseVector) = convert(Vector, v.indices)

function convert(::Type{Vector}, v::InsertableSparseVector{Tv,Ti,N}) where {Tv,Ti,N}
    vals = zeros(Tv, N)
    for index in v.indices
        vals[index] = v.values[index]
    end
    return vals
end

@inline function add!(v::InsertableSparseVector{Tv,Ti}, a::Tv, idx::Ti, prev_idx::Ti) where {Tv,Ti}
    if push!(v.indices, idx, prev_idx)
        v[idx] = a
    else
        v[idx] += a
    end

    v
end

function axpy!(a::Tv, A::SparseMatrixCSC{Tv,Ti}, column::Ti, start::Ti, y::InsertableSparseVector{Tv,Ti,N}) where {Tv,Ti,N}
    prev_index::Ti = N + one(Ti)
    
    @inbounds for idx = start : A.colptr[column + 1] - 1
        add!(y, a * A.nzval[idx], A.rowval[idx], prev_index)
        prev_index = A.rowval[idx]
    end

    y
end

function append_col!(A::SparseMatrixCSC{Tv}, y::InsertableSparseVector{Tv,Ti,N}, j::Ti, drop = zero(real(Tv))) where {Tv,Ti,N}
    
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
    y.indices[N + 1] = N + 1

    nothing
end