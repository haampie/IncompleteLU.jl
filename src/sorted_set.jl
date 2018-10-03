import Base: iterate, push!, Vector, getindex, setindex!, show, empty!

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
        next = Vector{Int}(undef, N + 1)
        @inbounds next[N + 1] = N + 1
        new(next, N + 1)
    end
end

# Convenience wrappers for indexing
@propagate_inbounds getindex(s::SortedSet, i::Int) = s.next[i]
@propagate_inbounds setindex!(s::SortedSet, value::Int, i::Int) = s.next[i] = value

# Iterate in 
@inline function iterate(s::SortedSet, p::Int = s.N) 
    @inbounds nxt = s[p]
    return nxt == s.N ? nothing : (nxt, nxt)
end

show(io::IO, s::SortedSet) = print(io, typeof(s), " with values ", Vector(s))

"""
For debugging and testing
"""
function Vector(s::SortedSet)
    v = Int[]
    for index in s
        push!(v, index)
    end
    return v
end

"""
Insert `index` after a known value `after`
"""
function push!(s::SortedSet, value::Int, after::Int)
    @inbounds begin
        while s[after] < value
            after = s[after]
        end

        if s[after] == value
            return false
        end
        
        s[after], s[value] = value, s[after]
        
        return true
    end
end

"""
Make the head pointer do a self-loop.
"""
@inline empty!(s::SortedSet) = s[s.N] = s.N

@inline push!(s::SortedSet, index::Int) = push!(s, index, s.N)
