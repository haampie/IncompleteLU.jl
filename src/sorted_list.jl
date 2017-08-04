import Base: start, next, done, push!, convert, getindex, setindex!

export SortedIndices, init!

"""
SortedIndices keeps track of a set of indices ≤ N
in sorted order using insertion sort. Requires O(N + 1)
memory.
"""
struct SortedIndices{N}
    next::Vector{Int}
end

SortedIndices(n::Int) = SortedIndices{n}(Vector{Int}(n + 1))

getindex(s::SortedIndices, i::Int) = s.next[i]
setindex!(s::SortedIndices, value::Int, i::Int) = s.next[i] = value

start(s::SortedIndices{N}) where {N} = N + 1
next(s::SortedIndices, p::Int) = s[p], s[p]
done(s::SortedIndices{N}, p::Int) where {N} = s[p] == N + 1

"""
For debugging and testing
"""
function convert(Vector, s::SortedIndices)
    v = Int[]
    for index in s
        push!(v, index)
    end
    return v
end

"""
Insert the first value 
"""
function init!(s::SortedIndices{N}, i::Int) where {N}
    s[i], s[N + 1] = N + 1, i
    return s
end

"""
Insert `index` after a known value `after`
"""
function push!(s::SortedIndices, index::Int, after::Int)    
    while s[after] < index
        after = s[after]
    end

    if s[after] == index
        return false
    end
    
    s[after], s[index] = index, s[after]
    
    return true
end

push!(s::SortedIndices{N}, index::Int) where {N} = push!(s, index, N + 1)
