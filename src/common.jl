module Common
using ACEhamiltonians
using JuLIP: Atoms
export parse_key, with_cache, number_of_orbitals, species_pairs, shell_pairs

# Converts strings into tuples of integers or integers as appropriate. This function
# should be refactored and moved to a more appropriate location. It is mostly a hack
# at the moment.
function parse_key(key)
    if key isa Integer || key isa Tuple
        return key
    elseif '(' in key
        return Tuple(parse.(Int, split(strip(key, ['(', ')', ' ']), ", ")))
    else
        return parse(Int, key)
    end
end


"""Nᵒ of orbitals present in system `atoms` based on a given basis definition."""
function number_of_orbitals(atoms::Atoms, basis_definition::BasisDef)
    # Work out the number of orbitals on each species
    n_orbs = Dict(k=>sum(v * 2 .+ 1) for (k, v) in basis_definition)
    # Use this to get the sum of the number of orbitals on each atom
    return sum(getindex.(Ref(n_orbs), getfield.(atoms.Z, :z)))
end

"""Nᵒ of orbitals present on a specific element, `z`, based on a given basis definition."""
number_of_orbitals(z::I, basis_definition::BasisDef) where I<:Integer = sum(basis_definition[z] * 2 .+ 1)



"""
Todo:
    - Document this function correctly.

Returns a cached guarded version of a function that stores known argument-result pairs.
This reduces the overhead associated with making repeated calls to expensive functions.
It is important to note that results for identical inputs will be the same object.

# Warnings
Do no use this function, it only supports a very specific use-case.

Although similar to the Memoize packages this function's use case is quite different and
is not supported by Memoize; hence the reimplementation here. This is mostly a stop-gap
measure and will be refactored at a later data.
"""
function with_cache(func::Function)::Function
    cache = Dict()
    function cached_function(args...)
        if !haskey(cache, args)
            cache[args] = func(args...)
        end
        return cache[args]
    end
    return cached_function
end


_triangular_number(n::I) where I<:Integer = n*(n + 1)÷2 

function species_pairs(atoms::Atoms)
    species = sort(unique(getfield.(atoms.Z, :z)))
    n = length(species)

    pairs = Vector{NTuple{2, valtype(species)}}(undef, _triangular_number(n))

    c = 0
    for i=1:n, j=i:n
        c += 1
        pairs[c] = (species[i], species[j])
    end

    return pairs
end


function shell_pairs(species_1, species_2, basis_def)
    n₁, n₂ = length(basis_def[species_1]), length(basis_def[species_2])
    c = 0

    if species_1 ≡ species_2
        pairs = Vector{NTuple{2, Int}}(undef, _triangular_number(n₁))

        for i=1:n₁, j=i:n₁
            c += 1
            pairs[c] = (i, j)
        end

        return pairs
    else
        pairs = Vector{NTuple{2, Int}}(undef, n₁ * n₂)

        for i=1:n₁, j=1:n₂
            c += 1
            pairs[c] = (i, j)
        end
        
        return pairs

    end
end


end