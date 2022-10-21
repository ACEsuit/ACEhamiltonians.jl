module States
using ACEhamiltonians, NeighbourLists, JuLIP
using ACEhamiltonians.MatrixManipulation: BlkIdx
using StaticArrays: SVector
using LinearAlgebra: norm, normalize
using ACE: AbstractState, CylindricalBondEnvelope, BondEnvelope, _evaluate_bond, _evaluate_env

import ACEhamiltonians.Parameters: ison
import ACE: _inner_evaluate

export BondState, AtomState, reflect, get_state

# ╔════════╗
# ║ States ║
# ╚════════╝
"""
    BondState(rr, rr0, bond)

State entities used when representing the environment about a bond.  

# Fields
- `rr`: environmental atom's position relative to the midpoint of the bond.
- `rr0`: vector between the two "bonding" atoms, i.e. the bond vector.
- `bond`: a boolean, which if true indicates the associated `BondState` entity represents
  the bond itself. If false, then the state is taken to represent an environmental atom
  about the bond rather than the bond itself.   

# Notes
If `bond == true` then `rr` should be set to `rr0/2`. If an environmental atom lies too
close to bond's midpoint then ACE may crash. Thus a small offset may be required in some
cases.

# Developers Notes
An additional field will be added at a later data to facilitate multi-species support. It
is possible that the `BondState` structure will have to be split into two sub-structures. 
"""
struct BondState{T<:SVector{3, <:AbstractFloat}, B<:Bool} <: AbstractState
    rr::T
    rr0::T
    bond::B
end


"""
    AtomState(rr)

State entity representing the environment about an atom.

# Fields
- `rr`: environmental atom's position relative to the host atom. 

# Developers Notes
An additional field will be added at a later data to facilitate multi-species support.

"""
struct AtomState{T<:SVector{3, <:AbstractFloat}} <: AbstractState
    rr::T
end

# ╭────────┬───────────────────────╮
# │ States │ General Functionality │
# ╰────────┴───────────────────────╯

# Display methods to help alleviate endless terminal spam.
function Base.show(io::IO, state::BondState)
    rr = string([round.(state.rr, digits=5)...])
    rr0 = string([round.(state.rr0, digits=5)...])
    print(io, "BondState(rr:$rr, rr0:$rr0, bond:$(state.bond))")
end

function Base.show(io::IO, state::AtomState)
    rr = string([round.(state.rr, digits=5)...])
    print(io, "AtomState(rr:$rr)")
end

# Allow for equality checks (will otherwise default to equivalency)  
Base.:(==)(x::T, y::T) where T<:BondState = x.rr == y.rr && y.rr0 == y.rr0 && x.bond == y.bond
Base.:(==)(x::T, y::T) where T<:AtomState = x.rr == y.rr

# The ≈ operator is commonly of more use for State entities than the equality
Base.isapprox(x::T, y::T; kwargs...) where T<:BondState = isapprox(x.rr, y.rr; kwargs...) && isapprox(x.rr0, y.rr0; kwargs...) && x.bond == y.bond
Base.isapprox(x::T, y::T; kwargs...) where T<:AtomState = isapprox(x.rr, y.rr; kwargs...)
Base.isapprox(x::T, y::T; kwargs...) where T<:AbstractVector{<:BondState} = all(x .≈ y)
Base.isapprox(x::T, y::T; kwargs...) where T<:AbstractVector{<:AtomState} = all(x .≈ y)

# ACE requires the `zero` method to be defined for states.
Base.zero(::Type{BondState{T, S}}) where {T, S} = BondState{T, S}(zero(T), zero(T), false)
Base.zero(::Type{AtomState{T}}) where T = AtomState{T}(zero(T))
Base.zero(::B) where B<:BondState = zero(B)
Base.zero(::B) where B<:AtomState = zero(B)
# Todo:
#   - Identify why `zero(BondState)` is being called and if the location where it is used
#     is effected by always choosing `bond` to be false. [PRIORITY:LOW]

"""
    ison(state)    

Return a boolean indicating whether the state entity is associated with either an on-site
or off-site interaction.
"""
ison(::T) where T<:AtomState = true
ison(::T) where T<:BondState = false


"""
    reflect(state)

Reflect `BondState` across the bond's midpoint. Calling on a state representing the bond A→B
will return the symmetrically B→A state. For states where `bond=true` this will flip the
sign on `rr` & `rr0`; whereas only `rr0` is flipped for the `bond=false` case.

# Arguments
- `state::BondState`: the state to be reflected.

# Returns
- `reflected_state::BondState`: a view of `state` reflected across the midpoint.

# Warnings
This is only valid for bond states whose atomic positions are given relative to the midpoint
of the bond; i.e. `envelope.λ≡0`.
"""
function reflect(state::T) where T<:BondState
    if state.bond
        return T(-state.rr, -state.rr0, true)
    else
        return T(state.rr, -state.rr0, false)
    end
end

# `reflect` is just an identify function for `AtomState` instances. This is included to
# alleviate the need for branching elsewhere in the code.
reflect(state::AtomState) = state

# ╭───────┬───────────╮
# │ State │ Factories │
# ╰───────┴───────────╯
"""
    get_state(i, atoms[; r=16.0])

Construct a state representing the environment about atom `i`.

# Arguments
- `i::Integer`: index of the atom whose state is to be constructed.
- `atoms::Atoms`: the `Atoms` object in which atom `i` resides.
- `r::AbstractFloat`: maximum distance up to which neighbouring atoms
  should be considered.

# Returns
- `state::Vector{::AtomState}`: state objects representing the environment about atom `i`.

"""
function get_state(i::Integer, atoms::Atoms; r::AbstractFloat=16.0)
    # Construct the neighbour list (this is cached so speed is not an issue)
    pair_list = JuLIP.neighbourlist(atoms, r; fixcell=false)

    # Extract environment about each relevant atom from the pair list. These will be tuples
    # of the form: (atomic-index, relative-position)
    idxs, vecs = NeighbourLists.neigs(pair_list, i) 

    # Neighbour list caching results in atoms outside the cutoff being present in `vecs`.
    filtered_vecs = filter(k -> norm(k) <= r, vecs)

    # Once multi-species support is added the atomic numbers will need to be retrieved.

    # Construct & return the `AtomState`` vector
    return AtomState.(filtered_vecs)
end


"""
    get_state(i, j, atoms, envelope[, image])

Construct a state representing the environment about the "bond" between atoms `i` & `j`.

# Arguments
- `i::Int`: atomic index of the first bonding atom.
- `j::Int`: atomic index of the second bonding atom.
- `atoms::Atoms`: the `Atoms` object in which atoms `i` and `j` reside.
- `envelope::CylindricalBondEnvelope:` an envelope specifying the volume to consider when
  constructing the state. This must be centred at the bond's midpoint; i.e. `envelope.λ≡0`.
- `image::Optional{Vector}`: a vector specifying the image in which atom `j`
  should reside; i.e. the cell translation vector. This defaults to `nothing` which will
  result in the closets periodic image of `j` being used. 

# Returns
- `state::Vector{::BondState}`: state objects representing the environment about the bond
  between atoms `i` and `j`.

# Notes
It is worth noting that a state will be constructed for the ij bond even when the distance
between them exceeds the bond-cutoff specified by the `envelope`. The maximum cutoff
distance for neighbour list construction is handled automatically.

# Warnings
- The neighbour list for the bond is constructed by applying an offset to the first atom's
  neighbour list. As such spurious states will be encountered when the ij distance exceeds
  the bond cutoff value `envelope.r0cut`. Do not ignore this warning!
- It is vital to ensure that when an `image` is supplied that all atomic coordinates are
  correctly wrapped into the unit cell. If fractional coordinates lie outside of the range
  [0, 1] then the results of this function will not be correct.

"""
function get_state(
    i::I, j::I, atoms::Atoms, envelope::CylindricalBondEnvelope,
    image::Union{AbstractVector{I}, Nothing}=nothing) where {I<:Integer}

    # Todo:
    #   - Combine the neighbour lists of atom i and j rather than just the former. This
    #     will reduce the probably of spurious state construction. But will increase run
    #     time as culling of duplicate states and bond states will be required.

    # Guard against instances where a non-zero envelope.λ value is used. When λ is set to
    # zero it means that all positions are relative to the mid-point of the bond. If this
    # is not the case then must more work is required elsewhere in the code. 
    @assert envelope.λ == 0.0 "Non-zero envelope λ values are not supported" 

    # Neighbour list cutoff distance; accounting for distances being relative to atom `i`
    # rather than the bond's mid-point.
    r = sqrt((envelope.r0cut + envelope.zcut)^2 + envelope.rcut^2)

    # Neighbours list construction (about atom `i`)
    idxs, vecs, cells = _neighbours(i, atoms, r)

    # Get the bond vector between atoms i & j; where i is in the origin cell & j resides
    # in either i) closest periodic image, or ii) that specified by `image` if provided.
    if isnothing(image)
        # Identify the shortest i→j vector account for PBC.
        idx = _locate_minimum_image(j, idxs, vecs)
        rr0 = vecs[idx]
    else
        @assert length(image) == 3 "image must be a vector of length three"
        # Find the vector between atom i in the origin cell and atom j in cell `image`.
        idx = _locate_target_image(j, idxs, cells, image)
        if idx != 0
            rr0 = vecs[idx]
        else  # Special case where the cutoff was too short to catch the desired i→j bond.
            # In this case we must calculate rr0 manually. 
            rr0 = atoms.X[j] - atoms.X[i] + (adjoint(image .* atoms.pbc) * atoms.cell).parent
        end
    end

    # The i→j bond vector must be removed from `vecs` so that it does not get treated as
    # an environmental atom in the for loop later on. This operation is done even if the
    # `idx==0` to maintain type stability.
    @views vecs_no_bond = vecs[1:end .!= idx]
    
    # As the mid-point of the bond is used as the origin an offset is needed to shift
    # vectors so they're relative to the bond's midpoint and not atom `i`.
    offset =  rr0 / 2.0

    # `BondState` entity vector 
    states = Vector{BondState{typeof(rr0), Bool}}(undef, length(vecs_no_bond) + 1)

    # Construct the bond vector state; i.e where `bond=true`
    states[1] = BondState(rr0/2.0, rr0, true)

    # Construct the environmental atom states; i.e. where `bond=false`.
    for k=1:length(vecs_no_bond)
        # Offset the vectors as needed and guard against erroneous positions.
        vec = _guard_position(vecs_no_bond[k] - offset, rr0, i, j)
        states[k+1] = BondState{typeof(rr0), Bool}(vec, rr0, false)
    end
    
    # Cull states outside of the bond envelope using the envelope's filter operator. This
    # task is performed manually here in an effort to reduce run time and memory usage.
    @views mask = _inner_evaluate.(Ref(envelope), states[2:end]) .!= 0.0
    @views n = sum(mask) + 1
    @views states[2:n] = states[2:end][mask]
    
    return states[1:n]

end



# Commonly one will need to collect multiple states rather than single states on their
# own. Hence the `get_state[s]` functions. These functions have been tagged for internal
# use only until they can be polished up. 

"""
    _get_states(block_idxs, atoms, envelope[, images])

Get the states describing the environments about a collection of bonds as defined by the
block index list `block_idxs`. This is effectively just a fancy wrapper for `get_state'.

# Arguments
- `block_idxs`: atomic index matrix in which the first & second rows specify the indices of the
  two "bonding" atoms. The third row, if present, is used to index `images` to collect
  cell in which the second atom lies.
- `atoms`: the `Atoms` object in which that atom pair resides.
- `envelope`: an envelope specifying the volume to consider when constructing the states.
- `images`: Cell translation index lookup list, this is only relevant when `block_idxs`
  supplies and cell index value. The cell translation index for the iᵗʰ state will be
  taken to be `images[block_indxs[i, 3]]`.

# Returns
- `bond_states::Vector{::Vector{::BondState}}`: a vector providing the requested bond states.

# Developers Notes
This is currently set to private until it is cleaned up.

"""
function _get_states(block_idxs::BlkIdx, atoms::Atoms{T}, envelope::CylindricalBondEnvelope,
    images::Union{AbstractMatrix{I}, Nothing}=nothing) where {I, T}
    if isnothing(images)
        if size(block_idxs, 1) == 3 && any block_idxs[3, :] != 1
            throw(ArgumentError("`idxs` provides non-origin cell indices but no 
            `images` argument was given!"))
        end
        return get_state.(block_idxs[1, :], block_idxs[2, :], Ref(atoms), Ref(envelope))::Vector{Vector{BondState{SVector{3, T}, Bool}}}
    else
        # If size(block_idxs,1) == 2, i.e. no cell index is supplied then this will error out.
        # Thus not manual error handling is required. If images are supplied then block_idxs
        # must contain the image index.
        # println("$(size(block_idxs[1, :])), $(size(block_idxs[2, :])), $(size())")
        return get_state.(
            block_idxs[1, :], block_idxs[2, :], Ref(atoms),
            Ref(envelope), eachcol(images[:, block_idxs[3, :]]))::Vector{Vector{BondState{SVector{3, T}, Bool}}}
    end
end


"""
    _get_states(block_idxs, atoms[; r=16.0])

Get states describing the environments around each atom block specified in `block_idxs`.
Note that `block_idxs` is assumed to contain only on-site blocks. This is just a wrapper
for `get_state'.

# Developers Notes
This is currently set to private until it is cleaned up.

"""
function _get_states(block_idxs::BlkIdx, atoms::Atoms{T}; r=16.0) where T
    if @views block_idxs[1, :] != block_idxs[2, :]
        throw(ArgumentError(
            "The supplied `block_idxs` represent a hetroatomic interaction. But the function 
            called is for retrieving homoatomic states."))
    end
    # Type ambiguities in the JuLIP.Atoms structure means that Julia cannot determine the
    # function's return type; specifically the value type of the static vector. Thus some
    # pseudo type hard coding must be done here.
    return get_state.(block_idxs[1, :], (atoms,); r=r)::Vector{Vector{AtomState{SVector{3, T}}}}
end



# ╭────────┬──────────────────────────╮
# │ States │ Factory Helper Functions │
# ╰────────┴──────────────────────────╯


"""
    _neighbours(i, atoms, r)

Identify and return information about atoms neighbouring atom `i` in system `atoms`.  

# Arguments
- `i::Int`: index of the atom for which the neighbour list is to be constructed.
- `atoms::Atoms`: system in which atom `i` is located.
- `r::AbstractFloat`: cutoff distance to for the neighbour list. Due to the effects off
  cacheing this should be treated as if it were a lower bounds for the cutoff rather than
  the cutoff itself. 

# Returns
- `idxs`: atomic index of each neighbour.
- `vecs`: distance vector to each neighbour.
- `cells`: index specifying the cell in which the neighbouring atom resides.

# Warnings
Due to the effects of caching there is a high probably that the returned neighbour list
will contain neighbours at distances greater than `r`.

"""
function _neighbours(i::Integer, atoms::Atoms, r::AbstractFloat)
    pair_list = JuLIP.neighbourlist(atoms, r; fixcell=false)
    return NeighbourLists.neigss(pair_list, i)
end


"""
    _locate_minimum_image(j, idxs, vecs)

Index of the closest `j` neighbour accounting for periodic boundary conditions.   

# Arguments
- `j::Integer`: Index of the atom for for whom the minimum image is to be identified.
- `idxs::Vector{::Integer}`: Integers specifying the indices of the atoms to two which
  the distances in `vecs` correspond.
- `vecs::Vector{SVector{3, ::AbstractFloat}}`: Vectors between the the source atom and
  the target atom.

# Returns
- `index::Integer`: an index `k` for which `vecs[k]` will yield the vector between the
  source atom and the closest periodic image of atom `j`.

# Notes
If multiple minimal vectors are found, then the first one will be returned. 

# Todo
- This will error out when the cutoff distance is lower than the bond distance. While such
  an occurrence is unlikely in smaller cells it will no doubt occur in larger ones.

"""
function _locate_minimum_image(j::Integer, idxs::AbstractVector{<:Integer}, vecs::AbstractVector{<:AbstractVector{<:AbstractFloat}})
    # Locate all entries in the neighbour list that correspond to atom `j`
    js = findall(==(j), idxs)
    if length(js) == 0
        # See the "Todo" section in the docstring.
        error("Neighbour not in range")
    end

    # Identify which image of atom `j` is closest
    return js[findmin(norm, vecs[js])[2]]
end

"""
    _locate_target_image(j, idxs, images, image)

Search through the neighbour list for atoms with the atomic index `j` that reside in
the specified `image` and return its index. If no such match is found, as can happen
when the cutoff distance is too short, then an index of zero is returned.

# Arguments
- `j`: index of the desired neighbour.
- `idxs`: atomic indices of atoms in the neighbour list.
- `images`: image in which the neighbour list atoms reside.
- `image`: image in which the target neighbour should reside.

# Returns
- `idx::Int`: index of the first entry in the neighbour list representing an atom with the
  atom index `j` residing in the image `image`. Zero if no matches are found.

# Notes
The `images` argument is set vector of vectors here as this is represents the type returned
by the neighbour list constructor. Blocking other types prevents any misunderstandings.

# Todo:
- Test for type instabilities

"""
function _locate_target_image(j::I, idxs::AbstractVector{I}, images::AbstractVector{<:AbstractVector{I}}, image::AbstractVector{I})::I where I<:Integer
    js = findall(==(j), idxs)
    idx = findfirst(i -> all(i .== image), images[js])
    return isnothing(idx) ? zero(I) : js[idx]
end


"""
    _guard_position(vec, rr0, i, j[; cutoff=0.05])

Clean environmental bond state positions to prevent erroneous behaviour.

If, when constructing a `BondState` entity, an environmental atom is found to lie too close
to the midpoint between the two bonding atoms then ACE will crash. Thus, such cases must be
carefully checked for and, when encountered, a small offset applied.

# Arguments
- `vec`: the vector to be checked and, if necessary, cleaned. This should be relative to
  the midpoint of the bond.
- `rr0`: bond vector.
- `i`: index of the first bonding atom.
- `j`: index of the second bonding atom.
- `cutoff`: the minimum permitted distance from the origin. Vectors closer to the origin
  than this value will be offset. 

# Returns
- `vec`: the original vector `vec` or a safely offset version thereof.

# Notes
While best efforts have been made to make the offset as reproducible as possible it is not
invariant to permutation for instances where the atom lies **exactly** at the midpoint.

"""
function _guard_position(vec::T, rr0::T, i::I, j::I; cutoff=0.05) where {I<:Integer, T}
    # If the an environmental atom lies too close to the origin it must be offset to avoid
    # errors. While this introduces noise, it is better than not being able to fit. A
    # better solution should be found where and if possible. 
    vec_norm = norm(vec)
    
    # If the vector is outside of the cutoff region then no action is required 
    if vec_norm >= cutoff
        return vec
    # If the atom is inside of the cutoff region, but not at exactly at the mid-point then
    # scale the vector so that it lies outside of the cutoff region.
    elseif 0 < vec_norm
        return normalize(vec) * cutoff
    # If the atom lies exactly at the bond origin, then offset it along the bond vector.
    elseif vec_norm == 0 
        # Shift vector is in direction of the atom with the lowest atomic index, or if both
        # are the same then the first atom. This helps to make the offset a little more
        # consistent and reproducible.
        o = i <= j ? -one(I) : one(I)
        return normalize(rr0) * (cutoff * o)
    end
end


# ╭────────┬───────────╮
# │ States │ Overrides │
# ╰────────┴───────────╯
# Local override to account for the `BondState` field `be::Symbol` being replaced with the
# field `bond::Bool`.  
function _inner_evaluate(env::BondEnvelope, state::BondState)
    if state.bond
        return _evaluate_bond(env, state)
    else
        return _evaluate_env(env, state)
    end
end

end
