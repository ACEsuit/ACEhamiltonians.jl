module Bases

using ACE
using ACE: SymmetricBasis, SphericalMatrix, Utils.RnYlm_1pbasis, SimpleSparseBasis,
           CylindricalBondEnvelope, Categorical1pBasis, cutoff_radialbasis, cutoff_env
using ACEbase
import ACEbase: read_dict, write_dict
using ACEbase.ObjectPools: VectorPool
using ACEhamiltonians
using ACEhamiltonians.Parameters: OnSiteParaSet, OffSiteParaSet


export Basis, IsoBasis, AnisoBasis, radial, angular, categorical, envelope, on_site_ace_basis, off_site_ace_basis, filter_offsite_be, is_fitted
"""
TODO:
    - Figure out what is going on with filter_offsite_be and its arguments.
    - A warning should perhaps be given if no filter function is given when one is
      expected; such as off-site functions. If no-filter function is desired than
      a dummy filter should be required.
    - Improve typing for the Model structure.
    - Replace e_cutₒᵤₜ, e_cutᵢₙ, etc. with more "typeable" names.
"""
###################
# Basis Structure #
###################
# Todo:
#   - Document
#   - Give type information
#   - Serialization routines
#   - `AnisoBasis` instances should raise an error if it is used inappropriately.


# ╔═══════╗
# ║ Basis ║
# ╚═══════╝

# abstract type Basis{T} end
abstract type Basis end


"""


ACE basis for modelling symmetry invariant interactions.


# Fields
- `basis::SymmetricBasis`:
- `id::Tuple`:
- `coefficients::Vector`:
- `mean::Matrix`:

"""
struct IsoBasis{T₁<:SymmetricBasis, T₂, T₃, T₄} <: Basis
    basis::T₁
    id::T₂
    coefficients::T₃
    mean::T₄

    function IsoBasis(basis, id)
        t = ACE.valtype(basis)
        F = real(t.parameters[5])
        IsoBasis(basis, id, zeros(F, length(basis)), zeros(F, size(zero(t))))
    end

    function IsoBasis(basis::T₁, id::T₂, coefficients::T₃, mean::T₄) where {T₁, T₂, T₃, T₄}
        new{T₁, T₂, T₃, T₄}(basis, id, coefficients, mean)
    end

end


"""

ACE basis for modelling symmetry variant interactions.


- `basis::SymmetricBasis`:
- `basis_i::SymmetricBasis`:
- `id::Tuple`:
- `coefficients::Vector`:
- `coefficients_i::Vector`:
- `mean::Matrix`:
- `mean_i::Matrix`:

"""
struct AnisoBasis{T₁<:SymmetricBasis, T₂<:SymmetricBasis, T₃, T₄, T₅, T₆, T₇} <: Basis
    basis::T₁
    basis_i::T₂
    id::T₃
    coefficients::T₄
    coefficients_i::T₅
    mean::T₆
    mean_i::T₇

    function AnisoBasis(basis, basis_i, id)
        t₁, t₂ = ACE.valtype(basis), ACE.valtype(basis_i)
        F = real(t₁.parameters[5])
        AnisoBasis(
            basis, basis_i,  id, zeros(F, length(basis)), zeros(F, length(basis_i)),
            zeros(F, size(zero(t₁))), zeros(F, size(zero(t₂))))
    end

    function AnisoBasis(basis::T₁, basis_i::T₂, id::T₃, coefficients::T₄, coefficients_i::T₅, mean::T₆, mean_i::T₇) where {T₁, T₂, T₃, T₄, T₅, T₆, T₇}
        new{T₁, T₂, T₃, T₄, T₅, T₆, T₇}(basis, basis_i,  id, coefficients, coefficients_i, mean, mean_i)
    end
end


Basis(basis, id) = IsoBasis(basis, id)
Basis(basis, basis_i, id) = AnisoBasis(basis, basis_i, id)

# ╭───────┬───────────────────────╮
# │ Basis │ General Functionality │
# ╰───────┴───────────────────────╯ 
"""Boolean indicating whether a `Basis` instance is fitted; i.e. has non-zero coefficients"""
is_fitted(basis::IsoBasis) = !all(basis.coefficients .≈ 0.0)
is_fitted(basis::AnisoBasis) = !(
    all(basis.coefficients .≈ 0.0) && all(basis.coefficients_i .≈ 0.0))

"""Check if two `Basis` instances are equivalent"""
function Base.:(==)(x::T₁, y::T₂) where {T₁<:Basis, T₂<:Basis}
    if T₁ != T₂ || typeof(x.basis) != typeof(y.basis)
        return false
    elseif T₁<:AnisoBasis && (typeof(x.basis_i) != typeof(y.basis_i))
        return false
    else
        return all(getfield(x, i) == getfield(y, i) for i in fieldnames(T₁))
    end
end


"""Expected shape of the sub-block associated with the `Basis`; 3×3 for a pp basis etc."""
Base.size(basis::Basis) = (ACE.valtype(basis.basis).parameters[3:4]...,)

# """Expected type of resulting sub-blocks."""
# Base.valtype(::Basis{T}) where T = T

"""Expected type of resulting sub-blocks."""
function Base.valtype(::Basis)
    throw("Basis structure type has been changed this function must be updated.")
end


"""Azimuthal quantum numbers associated with the `Basis`."""
azimuthals(basis::Basis) = (ACE.valtype(basis.basis).parameters[1:2]...,)

"""Returns a boolean indicating if the basis instance represents an on-site interaction."""
Parameters.ison(x::Basis) = length(x.id) ≡ 3


"""
    _filter_bases(basis, type)

Helper function to retrieve specific basis function information out of a `Basis` instance.
This is an internal function which is not expected to be used outside of this module. 

Arguments:
- `basis::Basis`: basis instance from which function is to be extracted.
- `type::DataType`: type of the basis functions to extract; e.g. `CylindricalBondEnvelope`.
"""
function _filter_bases(basis::Basis, T)
    functions = filter(i->i isa T, basis.basis.pibasis.basis1p.bases)
    if length(functions) == 0
        error("Could not locate basis function matching the supplied type")
    elseif length(functions) ≥ 2
        @warn "Multiple matching basis functions found, only the first will be returned"
    end
    return functions[1]
end

"""Extract and return the radial component of a `Basis` instance."""
radial(basis::Basis) = _filter_bases(basis, ACE.Rn1pBasis)

"""Extract and return the angular component of a `Basis` instance."""
angular(basis::Basis) = _filter_bases(basis, ACE.Ylm1pBasis)

"""Extract and return the categorical component of a `Basis` instance."""
categorical(basis::Basis) = _filter_bases(basis, ACE.Categorical1pBasis)

"""Extract and return the bond envelope component of a `Basis` instance."""
envelope(basis::Basis) = _filter_bases(basis, ACE.BondEnvelope)


# ╭───────┬──────────────────╮
# │ Basis │ IO Functionality │
# ╰───────┴──────────────────╯
"""
    write_dict(basis[,hash_basis])

Convert an `IsoBasis` structure instance into a representative dictionary.

# Arguments
- `basis::IsoBasis`: the `IsoBasis` instance to parsed.
- `hash_basis::Bool`: ff `true` then hash values will be stored in place of
  the `SymmetricBasis` objects.
"""
function write_dict(basis::T, hash_basis=false) where T<:IsoBasis
    return Dict(
        "__id__"=>"IsoBasis",
        "basis"=>hash_basis ? string(hash(basis.basis)) : write_dict(basis.basis),
        "id"=>basis.id,
        "coefficients"=>write_dict(basis.coefficients),
        "mean"=>write_dict(basis.mean))

end


"""
    write_dict(basis[,hash_basis])

Convert an `AnisoBasis` structure instance into a representative dictionary.

# Arguments
- `basis::AnisoBasis`: the `AnisoBasis` instance to parsed.
- `hash_basis::Bool`: ff `true` then hash values will be stored in place of
  the `SymmetricBasis` objects.
"""
function write_dict(basis::T, hash_basis::Bool=false) where T<:AnisoBasis
    return Dict(
        "__id__"=>"AnisoBasis",
        "basis"=>hash_basis ? string(hash(basis.basis)) : write_dict(basis.basis),
        "basis_i"=>hash_basis ? string(hash(basis.basis_i)) : write_dict(basis.basis_i),
        "id"=>basis.id,
        "coefficients"=>write_dict(basis.coefficients),
        "coefficients_i"=>write_dict(basis.coefficients_i),
        "mean"=>write_dict(basis.mean),
        "mean_i"=>write_dict(basis.mean_i))
end

"""Instantiate an `IsoBasis` instance from a representative dictionary."""
function ACEbase.read_dict(::Val{:IsoBasis}, dict::Dict)
    return IsoBasis(
        read_dict(dict["basis"]),
        Tuple(dict["id"]),
        read_dict(dict["coefficients"]),
        read_dict(dict["mean"]))
end

"""Instantiate an `AnisoBasis` instance from a representative dictionary."""
function ACEbase.read_dict(::Val{:AnisoBasis}, dict::Dict)
    return AnisoBasis(
        read_dict(dict["basis"]),
        read_dict(dict["basis_i"]),
        Tuple(dict["id"]),
        read_dict(dict["coefficients"]),
        read_dict(dict["coefficients_i"]),
        read_dict(dict["mean"]),
        read_dict(dict["mean_i"]))
end


function Base.show(io::IO, basis::T) where T<:Basis
    print(io, "$(nameof(T))(id: $(basis.id), fitted: $(is_fitted(basis))")
end



# ╔════════════════════════╗
# ║ ACE Basis Constructors ║
# ╚════════════════════════╝

@doc raw"""

    on_site_ace_basis(ℓ₁, ℓ₂, ν, deg, e_cutₒᵤₜ[, e_cutᵢₙ])

Initialise a simple on-site `SymmetricBasis` instance with sensible default parameters.

The on-site `SymmetricBasis` entities are produced by applying a `SimpleSparseBasis`
selector to a `Rn1pBasis` instance. The latter of which is initialised via the `Rn_basis`
method, using all the defaults associated therein except `e_cutₒᵤₜ` and `e_cutᵢₙ` which are
provided by this function. This facilitates quick construction of simple on-site `Bases`
instances; if more fine-grain control over over the initialisation process is required
then bases must be instantiated manually. 

# Arguments
- `(ℓ₁,ℓ₂)::Integer`: azimuthal numbers of the basis function.
- `ν::Integer`: maximum correlation order.
- `deg::Integer`: maximum polynomial degree.
- `e_cutₒᵤₜ::AbstractFloat`: only atoms within the specified cutoff radius will contribute
   to the local environment.
- `e_cutᵢₙ::AbstractFloat`: inner cutoff radius, defaults to 2.5.

# Returns
- `basis::SymmetricBasis`: ACE basis entity for modelling the specified interaction. 

"""
function on_site_ace_basis(ℓ₁::I, ℓ₂::I, ν::I, deg::I, e_cutₒᵤₜ::F, e_cutᵢₙ::F=2.5
    ) where {I<:Integer, F<:AbstractFloat}
    # Build i) a matrix indicating the desired sub-block shape, ii) the one
    # particle Rₙ·Yₗᵐ basis describing the environment, & iii) the basis selector.
    # Then instantiate the SymmetricBasis required by the Basis structure.
    return SymmetricBasis(
        SphericalMatrix(ℓ₁, ℓ₂; T=ComplexF64),
        RnYlm_1pbasis(maxdeg=deg, r0=e_cutᵢₙ, rcut=e_cutₒᵤₜ),
        SimpleSparseBasis(ν, deg))
end

@doc raw"""

    off_site_ace_basis(ℓ₁, ℓ₂, ν, deg, b_cut[,e_cutₒᵤₜ, e_cutᵢₙ, λₙ, λₗ])


Initialise a simple off-site `SymmetricBasis` instance with sensible default parameters.

Operates similarly to [`on_site_ace_basis`](@ref) but applies a `CylindricalBondEnvelope` to
the `Rn1pBasis` basis instance. The length and radius of the cylinder are defined as
maths: ``b_{cut}+2e_{cut\_out}`` and maths: ``e_{cut\_out}`` respectively; all other
parameters resolve to their defaults as defined by their constructors. Again, instances
must be manually instantiated if more fine-grained control is desired.

# Arguments
- `(ℓ₁,ℓ₂)::Integer`: azimuthal numbers of the basis function.
- `ν::Integer`: maximum correlation order.
- `deg::Integer`: maximum polynomial degree.
- `b_cut::AbstractFloat`: cutoff distance for bonded interactions.
- `e_cutₒᵤₜ::AbstractFloat`: radius and axial-padding of the cylindrical bond envelope that
   is used to determine which atoms impact to the bond's environment.
- `e_cutᵢₙ::AbstractFloat`: inner cutoff radius, defaults to 2.5.
- `λₙ::AbstractFloat`: ???
- `λₗ::AbstractFloat`: ???

# Returns
- `basis::SymmetricBasis`: ACE basis entity for modelling the specified interaction. 

"""
function off_site_ace_basis(ℓ₁::I, ℓ₂::I, ν::I, deg::I, b_cut::F, e_cutₒᵤₜ::F=5., e_cutᵢₙ::F=0.05,
    λₙ::F=.5, λₗ::F=.5) where {I<:Integer, F<:AbstractFloat}

    # Bond envelope which controls which atoms are seen by the bond.
    env = CylindricalBondEnvelope(b_cut, e_cutₒᵤₜ, e_cutₒᵤₜ, floppy=false, λ=0.0)

    # Categorical1pBasis is applied to the basis to allow atoms which are part of the
    # bond to be treated differently to those that are just part of the environment.
    discriminator = Categorical1pBasis([true, false]; varsym=:bond, idxsym=:bond)

    # The basis upon which the above entities act.
    RnYlm = RnYlm_1pbasis(maxdeg=deg, r0=e_cutᵢₙ, rcut=cutoff_env(env), trans=PolyTransform(1, 1/e_cutᵢₙ^2))

    
    return SymmetricBasis(
        SphericalMatrix(ℓ₁, ℓ₂; T=ComplexF64),
        RnYlm * env * discriminator,
        SimpleSparseBasis(ν + 1, deg),
        filterfun=states -> _filter_offsite_be(states, deg, λₙ, λₗ))
end

"""
    _filter_offsite_be(states, max_degree[, λ_n=0.5, λ_l=0.5])

Some mysterious filtering function.

This filter function should be passed, via the keyword `filterfun`, to `SymmetricBasis`
when instantiating.


# Arguments
- `states:?????`: Unknown, this is supplied by ase when used?
- `max_degree::Integer`: maximum polynomial degree.
- `λ_n::AbstractFloat`: ???
- `λ_l::AbstractFloat`: ???

# Developers Notes
This function and its doc-string will be rewritten once its function and arguments have
been identified satisfactorily.

# Examples
This is primarily intended to act as a filter function for off site bases like so:   
```
julia> off_site_sym_basis = SymmetricBasis(
        φ, basis, selector,
        filterfun = states -> filter_offsite_be(states, max_degree)
```

# Todo
    - This should be inspected and documented.
    - Refactoring may improve performance.

"""
function _filter_offsite_be(states, max_degree, λ_n=.5, λ_l=.5)
    if length(states) == 0; return false; end 
    deg_n, deg_l = ceil(Int, max_degree * λ_n), ceil(Int, max_degree * λ_l)
    for state in states
        if !state.bond && (state.n>deg_n || state.l>deg_l)
            return false
        end
    end
    return sum(state.bond for state in states) == 1
end

end