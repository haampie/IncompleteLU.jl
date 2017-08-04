import Base: start, next, done, push!, convert, getindex, setindex!

export SortedIndices, init!

"""
SortedIndices keeps track of a sorted set of indices ≤ N
using insertion sort with a linked list structure in a pre-allocated 
vector. Requires O(N + 1) memory. Insertion goes via a linear scan in O(n)
where `n` is the number of stored elements, but can be accelerated 
by passing along a known value in the set (which is useful when pushing
in an already sorted list). The insertion itself requires O(1) operations
due to the linked list structure.
"""
struct SortedIndices{Ti <: Integer, N}
    next::Vector{Ti}
    SortedIndices{Ti, N}() where {Ti <: Integer, N} = new(Vector{Ti}(N + one(Ti)))
end

SortedIndices(n::Ti) where {Ti <: Integer} = SortedIndices{Ti, n}()

getindex(s::SortedIndices{Ti}, i::Ti) where {Ti} = s.next[i]
setindex!(s::SortedIndices{Ti}, value::Ti, i::Ti) where {Ti} = s.next[i] = value

start(s::SortedIndices{Ti,N}) where {Ti,N} = N + one(Ti)
next(s::SortedIndices{Ti}, p::Ti) where {Ti} = s[p], s[p]
done(s::SortedIndices{Ti,N}, p::Ti) where {Ti,N} = s[p] == N + one(Ti)

"""
For debugging and testing
"""
function convert(Vector, s::SortedIndices{Ti}) where {Ti}
    v = Ti[]
    for index in s
        push!(v, index)
    end
    return v
end

"""
Insert the first value 
"""
function init!(s::SortedIndices{Ti,N}, i::Ti) where {Ti,N}
    s[i], s[N + one(Ti)] = N + one(Ti), i
    return s
end

"""
Insert `index` after a known value `after`
"""
function push!(s::SortedIndices{Ti,N}, index::Ti, after::Ti) where {Ti,N}
    while s[after] < index
        after = s[after]
    end

    if s[after] == index
        return false
    end
    
    s[after], s[index] = index, s[after]
    
    return true
end

push!(s::SortedIndices{Ti,N}, index::Ti) where {Ti,N} = push!(s, index, N + one(Ti))
