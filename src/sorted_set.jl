import Base: start, next, done, push!, convert, getindex, setindex!, show

export SortedSet, init!

"""
SortedSet keeps track of a sorted set of integers ≤ N
using insertion sort with a linked list structure in a pre-allocated 
vector. Requires O(N + 1) memory. Insertion goes via a linear scan in O(n)
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
struct SortedSet
    next::Vector{Int}
    N::Int

    function SortedSet(N::Int)
        next = Vector{Int}(N + 1)
        next[N + 1] = N + 1
        new(next, N + 1)
    end
end

# Convenience wrappers for indexing
@inline getindex(s::SortedSet, i::Int) = s.next[i]
@inline setindex!(s::SortedSet, value::Int, i::Int) = s.next[i] = value

# Iterate in 
@inline start(s::SortedSet) = s.N
@inline next(s::SortedSet, p::Int) = s[p], s[p]
@inline done(s::SortedSet, p::Int) = s[p] == s.N

show(io::IO, ::MIME"text/plain", s::SortedSet) = print(io, typeof(s), " with values ", convert(Vector, s))

"""
For debugging and testing
"""
function convert(::Type{Vector}, s::SortedSet)
    v = Int[]
    for index in s
        push!(v, index)
    end
    return v
end

# """
# Insert the first value 
# """
# function init!(s::SortedSet{Ti,N}, value::Ti) where {Ti,N}
#     s[value], s[N] = N, value
#     return s
# end

"""
Insert `index` after a known value `after`
"""
function push!(s::SortedSet, value::Int, after::Int)
    while s[after] < value
        after = s[after]
    end

    if s[after] == value
        return false
    end
    
    s[after], s[value] = value, s[after]
    
    return true
end

@inline push!(s::SortedSet, index::Int) = push!(s, index, s.N)
