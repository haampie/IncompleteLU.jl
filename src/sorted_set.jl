import Base: start, next, done, push!, convert, getindex, setindex!

export SortedSet, init!

"""
SortedSet keeps track of a sorted set of integers < N
using insertion sort with a linked list structure in a pre-allocated 
vector. Requires O(N) memory. Insertion goes via a linear scan in O(n)
where `n` is the number of stored elements, but can be accelerated 
by passing along a known value in the set (which is useful when pushing
in an already sorted list). The insertion itself requires O(1) operations
due to the linked list structure. Provides iterators:
```julia
ints = SortedSet(10)
push!(ints, 5)
push!(ints, 3)
for value in ints
    println(value)
end
```
"""
struct SortedSet{Ti <: Integer, N}
    next::Vector{Ti}
    SortedSet{Ti, N}() where {Ti <: Integer, N} = new(Vector{Ti}(N + one(Ti)))
end

SortedSet(n::Ti) where {Ti <: Integer} = SortedSet{Ti, n + one(Ti)}()

# Convenience wrappers for indexing
getindex(s::SortedSet{Ti}, i::Ti) where {Ti} = s.next[i]
setindex!(s::SortedSet{Ti}, value::Ti, i::Ti) where {Ti} = s.next[i] = value

# Iterate in 
start(s::SortedSet{Ti,N}) where {Ti,N} = N
next(s::SortedSet{Ti}, p::Ti) where {Ti} = s[p], s[p]
done(s::SortedSet{Ti,N}, p::Ti) where {Ti,N} = s[p] == N

"""
For debugging and testing
"""
function convert(Vector, s::SortedSet{Ti}) where {Ti}
    v = Ti[]
    for index in s
        push!(v, index)
    end
    return v
end

"""
Insert the first value 
"""
function init!(s::SortedSet{Ti,N}, value::Ti) where {Ti,N}
    s[value], s[N] = N, value
    return s
end

"""
Insert `index` after a known value `after`
"""
function push!(s::SortedSet{Ti}, value::Ti, after::Ti) where {Ti}
    while s[after] < value
        after = s[after]
    end

    if s[after] == value
        return false
    end
    
    s[after], s[value] = value, s[after]
    
    return true
end

push!(s::SortedSet{Ti,N}, index::Ti) where {Ti,N} = push!(s, index, N)
