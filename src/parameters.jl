module Parameters
using Base, ACEbase
export NewParams, GlobalParams, AtomicParams, AzimuthalParams, ShellParams, ParaSet, OnSiteParaSet, OffSiteParaSet, ison

# The `Params` structure has been temporarily renamed to `NewParams` to avoid conflicts
# with the old code. However, this will be rectified when the old code is overridden. 

# ╔════════════╗
# ║ Parameters ║
# ╚════════════╝
# Parameter related code.


#
# Todo:
#  - Parameters:
#    - Need to enforce limits on key values, shells must be larger than zero and
#      azimuthal numbers must be non-negative.
#    - All Params should be combinable, with compound classes generated when combining
#      different Params types. Compound types should always check the more refined
#      struct first (i.e. ShellParams<AzimuthalParams<AtomicParams).
#    - Constructor "build" macro
#      - Use macro to warn when redundant keys are found.
#      - Reduce code duplication.
#      - Add descriptive comments.
#    - Implement additional methods:
#      - `in` to check if there are any valid super-keys present which can yield a result.
#      - `haskey` to check directly if the exact key is present, this will not check for
#        sub-types, equivalent types and will not convert from shell to azimuthal number.
#  - ParaSet:
#    - Implement error checking:
#      - `Params` instance keys are correct for the expected type of interaction, i.e. a key
#        should contain two atomic numbers for off-site parameters.
#    - Ensure the outer cutoff is greater than the inner cutoff.
#    - Enforce parameter type consistency in a more elegant manner.
#  - to_dict method:
#    - Label: Not required
#    - GlobalParams: 
#    - AtomicParams: 
#    - AzimuthalParams: 
#    - ShellParams: 


# ╔═══════╗
# ║ Label ║
# ╚═══════╝
# This structure is allows for order agnostic interactions representations; e.g. the
# interactions (z₁, z₂) and (z₁, z₂) are not only equivalent but are fundamentally the
# same interaction. For an atomic number pair (z₁, z₂) or a shell number pair (s₁, s₂)
# the necessary representation could be achieved using a Set. However, this starts to
# fail when describing interactions by both atomic and shell number (z₁, z₂, s₁, s₂).
# Furthermore, it is useful to be able indicate that some interactions are sub-types
# of others; i.e. all the following interactions (z₁, z₂, 1, 1), (z₁, z₂, 1, 2) and
# (z₁, z₂, 2, 2) are sub-interactions of the (z₁, z₂) interaction type. This property
# is useful when dealing with groups of interactions specifying parameters.
#
# In general it is not intended for the user to interact with `Label` entities directly.
# Such structures are primarily used in the background. IT should be noted that these
# structures are designed for ease of use rather than performance; however this should not
# be an issue given that this is never used in performance critical parts of the code.
struct Label{N, I}
    id::NTuple{N, I}
    
    # Label(i, j, k...)
    Label(id::Vararg{I, N}) where {I<:Integer, N} = new{N, I}(_process_tuple(id))
    Label{N, I}(id::Vararg{I, N}) where {I<:Integer, N} = new{N, I}(_process_tuple(id))

    # Label((i, j, k, ...))
    Label(id::NTuple{N, I}) where {I<:Integer, N} = new{N, I}(_process_tuple(id))
    Label{N, I}(id::NTuple{N, I}) where {I<:Integer, N} = new{N, I}(_process_tuple(id))

    # Label(), Label((,)) () Special cases for an empty Label; used for global interactions
    Label(id::Tuple{}) = new{0, Int}(id)
    Label() = new{0, Int}(NTuple{0, Int}())
    Label{N, I}(id::Tuple{}) where {N, I<:Integer} = new{N, I}(id)
    Label{N, I}() where {N, I<:Integer} = new{N, I}(NTuple{N, I})
    
    # Label(Label)
    Label(id::Label) = id

    # Construct from string representation
    function Label(id::AbstractString)
        
        # Strip parentheses if present along with any trailing commas
        id = rstrip(startswith(id, "(") ? id[2:end-1] : id, ',')
        # Deal with special empty/"Any" label case 
        if length(id) == 0 || id == "Any"
            Label()
        # Normal label conversion
        else
            Label(Tuple(parse.(Int, split(id, ','))))#
        end
    end

end

# ╭───────┬───────────────────╮
# │ Label │ Equality Operator │
# ╰───────┴───────────────────╯
Base.:(==)(x::T₁, y::T₂) where {T₁<:Label, T₂<:Label} = x.id == y.id
Base.:(==)(x::T₁, y::T₂) where {T₁<:Label, T₂<:NTuple} = x == Label(y)
Base.:(==)(x::T₁, y::T₂) where {T₁<:Label, T₂<:Integer} = x == Label(y)
Base.:(==)(x::T₁, y::T₂) where {T₁<:NTuple, T₂<:Label} = Label(x) == y
Base.:(==)(x::T₁, y::T₂) where {T₁<:Integer, T₂<:Label} = Label(x) == y


# ╭───────┬─────────────────╮
# │ Label │ Subset Operator │
# ╰───────┴─────────────────╯
# These are used to check if an interaction represents a sub-interaction of another. For
# example CₛNₛ, CₛNₚ and CₚNₚ are all considered subtypes of the CN interaction; that is to
# say (6, 7, 0, 0), (6, 7, 0, 1) & (7, 6, 1, 1) are sub-sets of (6, 7). Note that all
# interactions are considered to be subtypes of Label((,)); i.e. an empty label is taken
# to mean "all interactions".

"""
    A ⊆ B

Checks if the label `A` is a subset of `B`.  
"""
function Base.:⊆(x::Label{N₁, I}, y::Label{N₂, I}) where {N₁, N₂, I}

    # If labels have the same length then check if they are equivalent
    if N₁≡N₂; return x==y

    # If N₁<N₂ then x cannot be equal to or a subset of y
    elseif N₁<N₂; false

    # All interactions are a subsets of the global label Label((,)); i.e. N₂≡0
    elseif N₂≡0; true

    # One and Two atom labels cannot be equal to or be subsets of one another. 
    elseif isodd(N₁)⊻isodd(N₂); false

    # Check if the x is a subset of y (one atom) 
    elseif isodd(N₁); x[1] == y[1]

    # Check if the x is a subset of y (two atom) 
    elseif iseven(N₁); x[1:2] == y[1:2]
    
    else; error("Unknown subset ⊆(Label,Label) call")
    end
end

# Routines to handle comparisons with tuples and the reverse subset operator `⊇`. 
Base.:⊆(x::T₁, y::T₂) where {T₁<:Label, T₂} = x ⊆ Label(y)
Base.:⊆(x::T₁, y::T₂) where {T₁, T₂<:Label} = Label(x) ⊆ y
Base.:⊇(x::T₁, y::T₂) where {T₁<:Label, T₂} = y ⊆ x
Base.:⊇(x::T₁, y::T₂) where {T₁, T₂<:Label} = y ⊆ x

# ╭───────┬───────────────────────╮
# │ Label │ General Functionality │
# ╰───────┴───────────────────────╯
# Length of a label, i.e. how many elements are present in its tuple
Base.length(::Label{N, I}) where {N, I} = N
Base.length(::Type{Label{N, I}}) where {N, I} = N

# Indexing
Base.getindex(x::Label, idx) = x.id[idx]
Base.lastindex(x::Label) = lastindex(x.id)

# Tuple/integer to Label conversion
Base.convert(t::Type{Label{N, I}}, x::NTuple{N, I}) where {N, I<:Integer} = t(x)
Base.convert(t::Type{Label{1, I}}, x::I) where I<:Integer = t(x)

# ╭───────┬──────────────────╮
# │ Label │ IO Functionality │
# ╰───────┴──────────────────╯
Base.show(io::IO, x::Label) = print(io, string(x.id))
Base.show(io::IO, ::Label{0, I}) where I = print(io, "Any")



# ╔════════════╗
# ║ Parameters ║
# ╚════════════╝
# These structures provide a means to easily specify and retrieve basis parameters. `Params`
# entities are created and accessed like dictionaries. Internally these are just dictionaries
# keyed by `Label` instances. However, the user should never need to access the internal
# dictionary. These instances are designed to be user friendly first and performant second.
# The lack of performance is not of concern as this will only ever be accessed once by the
# code when it gathers the parameters to construct the basis functions.   
   

# ╭────────┬───────╮
# │ Params │ Setup │
# ╰────────┴───────╯
# This macro just abstracts the highly repetitive constructor code used by `Params` sub-structures.
macro build(name, N, with_basis)
    T1 = :(Vararg{Pair{K, V}, N})

    # Function for Tuple->Label conversion 
    t2l(val::Pair{NTuple{N, I}, V}) where {N, I<:Integer, V} = convert(Pair{Label{N, I}, V}, val)

    if with_basis
        return quote
            function $(esc(name))(b_def, arg::$T1) where {K<:Label{$N, I}, V, N} where I<:Integer
                $(Expr(:call, Expr(:curly, esc(:new), :K, :V), Expr(:call, Dict, :arg), :b_def))
            end

            function $(esc(name))(b_def, arg::$T1) where {K<:NTuple{$N, I}, V, N} where I<:Integer
                $(Expr(:call, esc(name), :b_def, Expr(:(...), Expr(:call, :map, esc(t2l), :arg))))
            end
        end
    else
        return quote
            function $(esc(name))(arg::$T1) where {K<:Label{$N, I}, V, N} where I<:Integer
                $(Expr(:call, Expr(:curly, esc(:new), :K, :V), Expr(:call, Dict, :arg)))
            end

            function $(esc(name))(arg::$T1) where {K<:NTuple{$N, I}, V, N} where I<:Integer
                $(Expr(:call, esc(name), Expr(:(...), Expr(:call, :map, esc(t2l), :arg))))
            end
        end
    end
end


# ╭────────┬────────────╮
# │ Params │ Definition │
# ╰────────┴────────────╯
"""
Dictionary-like structures for specifying model parameters.

These are used to provide the parameters needed when constructing models within the
`ACEhamiltonians` framework. There are currently four `Params` type structures, namely
`GlobalParams`, `AtomicParams`, `AzimuthalParams`, and `ShellParams`, each offering
varying levels of specificity.

Each parameter, correlation order, maximum polynomial degree, environmental cutoff
distance, etc. may be specified using any of the available `Params` based structures.
However, i) each `Params` instance may represent one, and only one, parameter, and ii)
on/off-site parameters must not be mixed.
"""
abstract type NewParams{K, V} end

"""
    GlobalParams(val)

A `GlobalParams` instance indicates that a single value should be used for all relevant
interactions. Querying such instances will always return the value `val`; so long as the
query is valid. For example:
```
julia> p = GlobalParams(10.)
GlobalParams{Float64} with 1 entries:
  () => 10.0

julia> p[1] # <- query parameter associated with H
10.
julia> p[(1, 6)] # <- query parameter associated with H-C interaction
10.
julia> p[(1, 6, 1, 2)] # <- interaction between 1ˢᵗ shell on H and 2ⁿᵈ shell on C
10.
```
As can be seen the specified value `10.` will always be returned so long as the query is
valid. These instances are useful when specifying parameters that are constant across all
bases, such as the internal cutoff distance, as it avoids having to repeatedly specify it
for each and every interaction.

# Arguments
 - `val::Any`: value of the parameter

"""
struct GlobalParams{K, V} <: NewParams{K, V}
    _vals::Dict{K, V}
    
    @build GlobalParams 0 false
    # Catch for special case where a single value passed
    GlobalParams(arg) = GlobalParams(Label()=>arg)
end


"""
    AtomicParams(k₁=>v₁, k₂=>v₂, ..., kₙ=>vₙ)

These instances allow for parameters to be specified on a species by species basis. This
equates to one parameter per species for on-site interactions and one parameter per species
pair for off-site interactions. This will then result in all associated bases associated
with a specific species/species-pair all using a common value, like so: 
```
julia> p_on  = AtomicParams(1=>9., 6=>11.)
AtomicParams{Float64} with 2 entries:
  6 => 11.0
  1 => 9.0

julia> p_off = AtomicParams((1, 1)=>9., (1, 6)=>10., (6, 6)=>11.)
AtomicParams{Float64} with 3 entries:
  (6, 6) => 11.0
  (1, 6) => 10.0
  (1, 1) => 9.0

# The value 11. is returned for all on-site C interaction queries 
julia> p_on[(6, 1, 1)] == p_on[(6, 1, 2)] == p_on[(6, 2, 2)] == 11.
true
# The value 10. is returned for all off-site H-C interaction queries 
julia> p_off[(1, 6, 1, 1)] == p_off[(6, 1, 2, 1)] == p_off[(6, 1, 2, 2)] == 10.
true
```
These instances are instantiated in a similar manner to dictionaries and offer a finer
degree of control over the parameters than `GlobalParams` structures but are not as
granular as `AzimuthalParams` structures.

# Arguments
- `pairs::Pair`: a sequence of pair arguments specifying the parameters for each species
  or species-pair. Valid parameter forms are:

    - on-site: `z₁=>v` or `(z,)=>v` for on-sites
    - off-site: `(z₁, z₂)=>v` 

  where `zᵢ` represents the atomic number of species `i` and `v` the parameter valued
  associated with this species or specie pair.
  

# Notes
It is important to note that atom pair keys are permutationally invariant, i.e. the keys
`(1, 6)` and `(6, 1)` are redundant and will overwrite one another like so:
```
julia> test = AtomicParams((1, 6)=>10., (6, 1)=>1000.)
AtomicParams{Float64} with 1 entries:
  (1, 6) => 1000.0

julia> test[(1, 6)] == test[(6, 1)] == 1000.0
true
```
Finally atomic numbers will be sorted so that the lowest atomic number comes first. However,
this is only a superficial visual change and queries will still be invariant to permutation.
"""
struct AtomicParams{K, V} <: NewParams{K, V}
    _vals::Dict{K, V}

    @build AtomicParams 1 false
    @build AtomicParams 2 false
    # Catch for special case where keys are integers rather than tuples
    AtomicParams(arg::Vararg{Pair{I, V}, N}) where {I<:Integer, V, N} = AtomicParams(
        map(i->((first(i),)=>last(i)), arg)...)

end


"""
    AzimuthalParams(basis_definition, k₁=>v₁, k₂=>v₂, ..., kₙ=>vₙ)

Parameters specified for each azimuthal quantum number of each species. This allows for a
finer degree of control and is a logical extension of the `AtomicParams` structure. It is
important to note that `AzimuthalParams` instances must be supplied with a basis definition.
This allows it to work out the azimuthal quantum number associated with each shell during
lookup.

```
# Basis definition describing a H_1s C_2s1p basis set
julia> basis_def = Dict(1=>[0], 6=>[0, 0, 1])
julia> p_on = AzimuthalParams(
    basis_def, (1, 0, 0)=>1, (6, 0, 0)=>2, (6, 0, 1)=>3, (6, 1, 1)=>4)
AzimuthalParams{Int64} with 4 entries:
  (6, 0, 0) => 2
  (1, 0, 0) => 1
  (6, 1, 1) => 4
  (6, 0, 1) => 3

julia> p_off = AzimuthalParams(
    basis_def, (1, 1, 0, 0)=>1, (6, 6, 0, 0)=>2, (6, 6, 0, 1)=>3, (6, 6, 1, 1)=>4,
    (1, 6, 0, 0)=>6, (1, 6, 0, 1)=>6)

AzimuthalParams{Int64} with 6 entries:
    (1, 6, 0, 1) => 6
    (6, 6, 0, 1) => 3
    (1, 6, 0, 0) => 6
    (1, 1, 0, 0) => 1
    (6, 6, 1, 1) => 4
    (6, 6, 0, 0) => 2

# on-site interactions involving shells 1 % 2 will return 2 as they're both s-shells.
julia> p_on[(6, 1, 1)] == p_on[(6, 1, 2)] == p_on[(6, 2, 2)] == 2
true

```

# Arguments
- `basis_definition::BasisDef`: basis definition specifying the bases present on each
  species. This is used to work out the azimuthal quantum number associated with each
  shell when queried.
- `pairs::Pair`: a sequence of pair arguments specifying the parameters for each unique
  atomic-number/azimuthal-number pair. Valid forms are:
  
   - on-site: `(z, ℓ₁, ℓ₂)=>v`  
   - off-site: `(z₁, z₂, ℓ₁, ℓ₂)=>v`
   
  where `zᵢ` and `ℓᵢ` represents the atomic and azimuthal numbers of species `i` to which
  the parameter `v` is associated.

# Notes
While keys are agnostic to the ordering of the azimuthal numbers; the first atomic number
`z₁` will always correspond to the first azimuthal number `ℓ₁`, i.e.:
    - `(z₁, ℓ₁, ℓ₂) == (z₁, ℓ₂, ℓ₁)`
    - `(z₁, z₂, ℓ₁, ℓ₂) == (z₂, z₁, ℓ₂, ℓ₁)`
    - `(z₁, z₂, ℓ₁, ℓ₂) ≠ (z₁, z₂ ℓ₂, ℓ₁)`
    - `(z₁, z₂, ℓ₁, ℓ₂) ≠ (z₂, z₁ ℓ₁, ℓ₂)`

"""
struct AzimuthalParams{K, V} <: NewParams{K, V}
    _vals::Dict{K, V}
    _basis_def

    @build AzimuthalParams 3 true
    @build AzimuthalParams 4 true
end

"""
    ShellParams(k₁=>v₁, k₂=>v₂, ..., kₙ=>vₙ)

`ShellParams` structures allow for individual values to be provided for each and every
unique interaction. While this proved the finest degree of control it can quickly become
untenable for systems with large basis sets or multiple species due the shear number of
variable required.
```
# For H1s C2s1p basis set.
julia> p_on = ShellParams(
    (1, 1, 1)=>1, (6, 1, 1)=>2, (6, 1, 2)=>3, (6, 1, 3)=>4,
    (6, 2, 2)=>5, (6, 2, 3)=>6, (6, 3, 3)=>7)

ShellParams{Int64} with 7 entries:
  (6, 3, 3) => 7
  (1, 1, 1) => 1
  (6, 1, 3) => 4
  (6, 2, 2) => 5
  (6, 1, 1) => 2
  (6, 1, 2) => 3
  (6, 2, 3) => 6

julia> p_off = ShellParams(
    (1, 1, 1, 1)=>1, (1, 6, 1, 1)=>2, (1, 6, 1, 2)=>3, (1, 6, 1, 3)=>4,
    (6, 6, 1, 1)=>5, (6, 6, 1, 2)=>6, (6, 6, 1, 3)=>74, (6, 6, 2, 2)=>8,
    (6, 6, 2, 3)=>9, (6, 6, 3, 3)=>10)

ShellParams{Int64} with 10 entries:
  (6, 6, 2, 2) => 8
  (6, 6, 3, 3) => 10
  (6, 6, 1, 3) => 74
  (1, 1, 1, 1) => 1
  (1, 6, 1, 2) => 3
  (1, 6, 1, 1) => 2
  (1, 6, 1, 3) => 4
  (6, 6, 1, 1) => 5
  (6, 6, 1, 2) => 6
  (6, 6, 2, 3) => 9

```

# Arguments
- `pairs::Pair`: a sequence of pair arguments specifying the parameters for each unique
  shell pair:
   - on-site: `(z, s₁, s₂)=>v`, interaction between shell numbers `s₁` & `s₂` on species `z`
   - off-site: `(z₁, z₂, s₁, s₂)=>v`, interaction between shell number `s₁` on species
   `zᵢ` and shell number `s₂` on species `z₂`.

"""
struct ShellParams{K, V} <: NewParams{K, V}
    _vals::Dict{K, V}

    @build ShellParams 3 false
    @build ShellParams 4 false
end

# ╭────────┬───────────────────────╮
# │ Params │ General Functionality │
# ╰────────┴───────────────────────╯
# Return the key and value types of the internal dictionary.
Base.valtype(::NewParams{K, V}) where {K, V} = V
Base.keytype(::NewParams{K, V}) where {K, V} = K
Base.valtype(::Type{NewParams{K, V}}) where {K, V} = V
Base.keytype(::Type{NewParams{K, V}}) where {K, V} = K

# Extract keys and values from the internal dictionary (and number of elements)
Base.keys(x::NewParams) = keys(x._vals)
Base.values(x::NewParams) = values(x._vals)
Base.length(x::T) where T<:NewParams = length(x._vals)

# Equality check, mostly use during testing
function Base.:(==)(x::T₁, y::T₂) where {T₁<:NewParams, T₂<:NewParams}
    dx, dy = x._vals, y._vals
    # Different type Params are not comparable
    if T₁ ≠ T₂
        return false
    # Different key sets means x & y are different
    elseif keys(dx) ≠ keys(dy)
        return false
    # If any key yields a different value in x from x then x & y are different 
    else
        for key in keys(dx)
            if dx[key] ≠ dy[key]
                return false
            end
        end
        # Otherwise there is no difference between x and y, thus return true
        return true
    end
end



# ╭────────┬────────────────────╮
# │ Params │ Indexing Functions │
# ╰────────┴────────────────────╯
"""
    params_object[key]

This function makes `Params` structures indexable in the same way that dictionaries are.
This will not only check the `Params` object `params` for the specified key `key` but will
also check for i) permutationally equivalent matches, i.e. (1, 6)≡(6, 1), and ii) keys
that `key` is a subtype of i.e. (1, 6, 1, 1) ⊆ (1, 6).

Valid key types are:
 - z/(z,): single atomic number
 - (z₁, z₂): pair of atomic numbers
 - (z, s₁, s₂): single atomic number with pair of shell numbers
 - (z₁, z₂, s₁, s₂): pair of atomic numbers with pair of shell numbers

This is primarily intended to be used by the code internally, but is left accessible to the
user.
"""
function Base.getindex(x::T, key::K) where {T<:NewParams, K}
    # This will not only match the specified key but also any superset it is a part of;
    # i.e. the key (z₁, z₂, s₁, s₂) will match (z₁, z₂).

    # Block 1: convert shell numbers to azimuthal numbers for the AzimuthalParams case.
    if T<:AzimuthalParams && !(K<:Integer)
        if length(key) == 3 
            key = (key[1], [x._basis_def[key[1]][i] for i in key[2:3]]...)
        else
            key = (key[1:2]..., x._basis_def[key[1]][key[3]], x._basis_def[key[2]][key[4]])
        end
    end

    # Block 2: identify closest viable key.
    super_key = filter(k->(key ⊆ k), keys(x))

    # Block 3: retrieve the selected key.
    if length(super_key) ≡ 0
        throw(KeyError(key))
    else
        return x._vals[first(super_key)]
    end
end


# ╭────────┬──────────────────╮
# │ Params │ IO Functionality │
# ╰────────┴──────────────────╯
"""Full, multi-line string representation of a `Param` type objected"""
function _multi_line(x::T) where T<:NewParams
    i = length(keytype(x._vals).types[1].types) ≡ 1 ? 1 : Base.:(:)
    v_string = join(["$(k[i]) => $v" for (k, v) in x._vals], "\n  ")
    return "$(nameof(T)){$(valtype(x))} with $(length(x._vals)) entries:\n  $(v_string)"
end


function Base.show(io::O, x::T) where {T<:NewParams, O<:IO}
    # If printing an isolated Params instance, just use the standard multi-line format
    if !haskey(io.dict, :SHOWN_SET)
        print(io, _multi_line(x))
    # If the Params is being printed as part of a group then a more compact
    # representation is needed.
    else
        # Create a slicer remove braces from tuples of length 1 if needed
        s = length(keytype(x)) == 1 ? 1 : Base.:(:)
        # Sort the keys to ensure consistency
        keys_s = sort([j.id for j in keys(x._vals)])  
        # Only show first and last keys (or just the first if there is only one)
        targets = length(x) != 1 ? [[1, lastindex(keys_s)]] : [1]
        # Build the key list and print the message out
        k_string = join([k[s] for k in keys_s[targets...]], " … ")
        print(io, "$(nameof(T))($(k_string))")
    end
end

# Special show case: Needed as Base.TTY has no information dictionary 
Base.show(io::Base.TTY, x::T) where T<:NewParams = print(io, _multi_line(x))


function ACEbase.write_dict(p::T) where T<:NewParams{K, V} where {K, V}
    # Recursive and arbitrary value type storage to be implemented later
    # value_parsable = hasmethod(ACEbase.write_dict, (V))

    dict = Dict(
        "__id__"=>"NewParams",
        "vals"=>Dict(string(k)=>v for (k, v) in p._vals))
    
    if T<:AzimuthalParams
        dict["basis_def"] = p._basis_def
    end

    return dict
end

function ACEbase.read_dict(::Val{:NewParams}, dict::Dict)
    vals = Dict(Label(k)=>v for (k,v) in dict["vals"])
    n = length(keytype(vals))

    if n ≡ 0
        return GlobalParams(vals...)
    elseif n ≤ 2
        return AtomicParams(vals...)
    elseif haskey(dict, "basis_def")
        return AzimuthalParams(dict["basis_def"], vals...)
    else
        return ShellParams(vals...)
    end

end


# ╔═════════╗
# ║ ParaSet ║
# ╚═════════╝
# Containers for collections of `Params` instances. These exist mostly to ensure that
# all the required parameters are specified and provide a single location where user
# specified parameters can be collected and checked.

# ╭─────────┬────────────╮
# │ ParaSet │ Definition │
# ╰─────────┴────────────╯
"""
`ParaSet` instances are structures which collect all the required parameter definitions
for a given interaction type in once place. Once instantiated, the `OnSiteParaSet` and
`OffSiteParaSet` structures should contain all parameters required to construct all of
the desired on/off-site bases.
"""
abstract type ParaSet end


"""
    OnSiteParaSet(ν, deg, e_cut_out, r0)

This structure holds all the `Params` instances required to construct the on-site
bases.


# Arguments
- `ν::Params{K, Int}`: correlation order, for on-site interactions the body order is one
  more than the correlation order.   
- `deg::Params{K, Int}`: maximum polynomial degree.
- `e_cut_out::Parameters{K, Float}`: environment's external cutoff distance.
- `r0::Parameters{K, Float}`: scaling parameter (typically set to the nearest neighbour distances).

"""
struct OnSiteParaSet <: ParaSet
    ν
    deg
    e_cut_out
    r0

    function OnSiteParaSet(ν::T₁, deg::T₂, e_cut_out::T₃, r0::T₄
        ) where {T₁<:NewParams, T₂<:NewParams, T₃<:NewParams, T₄<:NewParams}
        ν::NewParams{<:Label, <:Integer}
        deg::NewParams{<:Label, <:Integer}
        e_cut_out::NewParams{<:Label, <:AbstractFloat}
        r0::NewParams{<:Label, <:AbstractFloat}
        new(ν, deg, e_cut_out, r0)
    end

end

"""
    OffSiteParaSet(ν, deg, b_cut, e_cut_out, r0)

This structure holds all the `Params` instances required to construct the off-site
bases.

# Arguments
- `ν::Params{K, Int}`: correlation order, for off-site interactions the body order is two
  more than the correlation order.   
- `deg::Params{K, Int}`: maximum polynomial degree.
- `b_cut::Params{K, Float}`: cutoff distance for off-site interactions.
- `e_cut_out::Params{K, Float}`: environment's external cutoff distance.
- `r0::Params{K, Float}`: scaling parameter (typically set to the nearest neighbour distances).
"""
struct OffSiteParaSet <: ParaSet
    ν
    deg
    b_cut
    e_cut_out
    r0
    
    function OffSiteParaSet(ν::T₁, deg::T₂, b_cut::T₃, e_cut_out::T₄, r0::T₅
        ) where {T₁<:NewParams, T₂<:NewParams, T₃<:NewParams, T₄<:NewParams, T₅<:NewParams}
        ν::NewParams{<:Label, <:Integer}
        deg::NewParams{<:Label, <:Integer}
        b_cut::NewParams{<:Label, <:AbstractFloat}
        e_cut_out::NewParams{<:Label, <:AbstractFloat}
        r0::NewParams{<:Label, <:AbstractFloat}
        new(ν, deg, b_cut, e_cut_out, r0)
    end

end

# ╭─────────┬───────────────────────╮
# │ ParaSet │ General Functionality │
# ╰─────────┴───────────────────────╯
function Base.:(==)(x::T, y::T) where T<:ParaSet
    # Check that all fields are equal to one another
    for field in fieldnames(T)
        # If any do not match then return false
        if getfield(x, field) ≠ getfield(y, field)
            return false
        end
    end

    # If all files match then return true
    return true
end


# ╭─────────┬────────────────────────────────╮
# │ ParaSet │ Miscellaneous Helper Functions │
# ╰─────────┴────────────────────────────────╯
# Returns true if a `ParaSet` corresponds to an on-site interaction.
ison(::OnSiteParaSet) = true
ison(::OffSiteParaSet) = false


# ╭─────────┬──────────────────╮
# │ ParaSet │ IO Functionality │
# ╰─────────┴──────────────────╯
function ACEbase.write_dict(p::T) where T<:ParaSet
    dict = Dict(
        "__id__"=>"ParaSet",
        (string(fn)=>write_dict(getfield(p, fn)) for fn in fieldnames(T))...)
    return dict
end


function ACEbase.read_dict(::Val{:ParaSet}, dict::Dict)
    if haskey(dict, "b_cut")
        return OffSiteParaSet((
            ACEbase.read_dict(dict[i]) for i in
            ["ν", "deg", "b_cut", "e_cut_out", "r0"])...)
    else
        return OnSiteParaSet((
            ACEbase.read_dict(dict[i]) for i in
            ["ν", "deg", "e_cut_out", "r0"])...)
    end
end



# ╭─────────┬────────────────────╮
# │ ParaSet │ Indexing Functions │
# ╰─────────┴────────────────────╯
"""
    on_site_para_set[key]

Indexing an `OnSiteParaSet` instance will index each of the internal fields and return
their results in a tuple, i.e. calling `res = on_site_para_set[key]` equates to calling
```
res = (
    on_site_para_set.ν[key], on_site_para_set.deg[key],
    on_site_para_set.e_cut_out[key], on_site_para_set.e_cut_in[key])
```

This is mostly intended as a convenience function.
"""
function Base.getindex(para::OnSiteParaSet, key)
    return (
        para.ν[key], para.deg[key],
        para.e_cut_out[key], para.r0[key])
end


"""
    off_site_para_set[key]

Indexing an `OffSiteParaSet` instance will index each of the internal fields and return
their results in a tuple, i.e. calling `res = off_site_para_set[key]` equates to calling
```
res = (
    off_site_para_set.ν[key], off_site_para_set.deg[key], off_site_para_set.b_cut[key],
    off_site_para_set.e_cut_out[key], off_site_para_set.e_cut_in[key])
```

This is mostly intended as a convenience function.
"""
function Base.getindex(para::OffSiteParaSet, key)
    return (
        para.ν[key], para.deg[key], para.b_cut[key],
        para.e_cut_out[key], para.r0[key])
end



# ╔═══════════════════════════╗
# ║ Internal Helper Functions ║
# ╚═══════════════════════════╝

"""
Sort `Label` tuples so that the lowest atomic-number/shell-number comes first for the
two/one atom interaction labels. If more than four integers are specified then an error
is raised.
"""

"""
    _process_ctuple(tuple)

Preprocess tuples prior to their conversion into `Label` instances. This ensures that
tuples are ordered so that:
 1. the lowest atomic number comes first, but only if multiple atomic numbers are specified.
 2. the lowest shell number comes first, but only where this does not conflict with point 1.

An error is then raised if the tuple is of an unexpected length. permitted lengths are:
 - 1/(z) single atomic number.
 - 2/(z₁, z₂) pair of atomic numbers
 - 3/(z, s₁, s₂) single atomic number and pair of shell numbers
 - 4/(z₁, z₂, s₁, s₂) pair of atomic numbers and a pair of shell numbers.

Note that in the latter case s₁ & s₂ correspond to shells on z₁ & z₂ respectively thus
if z₁ and z₂ are flipped due to z₁>z₂ then s₁ & s₂ must also be shuffled.

This os intended only to be used internally and only during the construction of `Label`
instances.
"""
function _process_tuple(x::NTuple{N, I}) where {N, I<:Integer}
    if N <= 1; x
    elseif N ≡ 2; x[1] ≤ x[2] ? x : reverse(x)
    elseif N ≡ 3; x[2] ≤ x[3] ? x : x[[1, 3, 2]]
    elseif N ≡ 4
        if x[1] > x[2] || ((x[1] ≡ x[2]) && (x[3] > x[4])); x[[2, 1, 4, 3]]
        else; x
        end
    else
        error(
            "Label may contain no more than four integers, valid formats are:\n"*
            "  ()\n  (z₁,)\n  (z₁, s₁, s₂)\n  (z₁, z₂)\n  (z₁, z₂, s₁, s₂)")
    end
end


# # Guards type conversion of dictionaries keyed with `Label` entities. This is done to
# # ensure that a meaningful message is given to the user when a key-collision occurs.
# function _guarded_convert(t::Type{Dict{Label{N, I}, V}}, x::Dict{NTuple{N, I}, V}) where {N, I<:Integer, V}
#     try
#         return convert(t, x)
#     catch e
#         if e.msg == "key collision during dictionary conversion" 
#             r_keys = _redundant_keys([k for k in keys(x)])
#             error("Redundant keys found:\n$(join(["  - $(join(i, ", "))" for i in r_keys], "\n"))")
#         else
#             rethrow(e)
#         end
#     end
# end

# # Collisions cannot occur when input dictionary is keyed by integers not tuples
# _guarded_convert(t::Type{Dict{Label{1, I}, V}}, x::Dict{I, V}) where {N, I<:Integer, V} = convert(t, x)


# function _redundant_keys(keys_in::Vector{NTuple{N, I}}) where {I<:Integer, N}
#     duplicates = []
#     while length(keys_in) ≥ 1
#         key = Label(pop!(keys_in))
#         matches = [popat!(keys_in, i) for i in findall(i -> i == key, keys_in)]
#         if length(matches) ≠ 0
#             append!(duplicates, Ref((key, matches...)))
#         end
#     end
#     return duplicates
# end

end