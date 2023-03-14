module Bases

using ACEhamiltonians, ACE, ACEbase, SparseArrays, LinearAlgebra

using ACEhamiltonians.Parameters: OnSiteParaSet, OffSiteParaSet
using ACE: SymmetricBasis, SphericalMatrix, Utils.RnYlm_1pbasis, SimpleSparseBasis,
           CylindricalBondEnvelope, Categorical1pBasis, cutoff_radialbasis, cutoff_env,
           get_spec, coco_dot

using ACEbase.ObjectPools: VectorPool
using ACEhamiltonians: BOND_ORIGIN_AT_MIDPOINT, SYMMETRY_FIX_ENABLED


import ACEbase: read_dict, write_dict
import LinearAlgebra.adjoint, LinearAlgebra.transpose, Base./
import Base
import ACE: SphericalMatrix


export AHBasis, radial, angular, categorical, envelope, on_site_ace_basis, off_site_ace_basis, filter_offsite_be, is_fitted, Basis, AnisoBasis
"""
TODO:
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



# ╔═══════╗
# ║ Basis ║
# ╚═══════╝

abstract type AHBasis end



"""


ACE basis for modelling symmetry invariant interactions.


# Fields
- `basis::SymmetricBasis`:
- `id::Tuple`:
- `coefficients::Vector`:
- `mean::Matrix`:

"""
struct Basis{T₁<:SymmetricBasis, T₂, T₃, T₄} <: AHBasis
    basis::T₁
    id::T₂
    coefficients::T₃
    mean::T₄

    function Basis(basis, id)
        t = ACE.valtype(basis)
        F = real(t.parameters[5])
        Basis(basis, id, zeros(F, length(basis)), zeros(F, size(zero(t))))
    end

    function Basis(basis::T₁, id::T₂, coefficients::T₃, mean::T₄) where {T₁, T₂, T₃, T₄}
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
struct AnisoBasis{T₁<:SymmetricBasis, T₂<:SymmetricBasis, T₃, T₄, T₅, T₆, T₇} <: AHBasis
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

AHBasis(basis, id) = Basis(basis, id)
AHBasis(basis, basis_i, id) = AnisoBasis(basis, basis_i, id)


# ╭───────┬───────────────────────╮
# │ Basis │ General Functionality │
# ╰───────┴───────────────────────╯ 
"""Boolean indicating whether a `Basis` instance is fitted; i.e. has non-zero coefficients"""
is_fitted(basis::AHBasis) = !all(basis.coefficients .≈ 0.0) || !all(basis.mean .≈ 0.0)


"""Check if two `Basis` instances are equivalent"""
function Base.:(==)(x::T₁, y::T₂) where {T₁<:AHBasis, T₂<:AHBasis}
    
    # Check that the ID's, coefficients and means match up first 
    check = x.id == y.id && size(x.mean) == size(y.mean) && x.mean == y.mean
    
    # If they don't then return false. Otherwise perform a check of the basis object
    # itself. A try/catch block must be used when comparing the bases as this can
    # result in a DimensionMismatch.
    if !check
        return check
    else
        try
            return x.basis == y.basis
        catch y
            if isa(y, DimensionMismatch)
                return false
            else
                rethrow(y)
            end

        end
    end
end


"""Expected shape of the sub-block associated with the `Basis`; 3×3 for a pp basis etc."""
Base.size(basis::AHBasis) = (ACE.valtype(basis.basis).parameters[3:4]...,)

# """Expected type of resulting sub-blocks."""
# Base.valtype(::Basis{T}) where T = T

"""Expected type of resulting sub-blocks."""
function Base.valtype(::AHBasis)
    throw("AHBasis structure type has been changed this function must be updated.")
end


"""Azimuthal quantum numbers associated with the `Basis`."""
azimuthals(basis::AHBasis) = (ACE.valtype(basis.basis).parameters[1:2]...,)

"""Returns a boolean indicating if the basis instance represents an on-site interaction."""
Parameters.ison(x::AHBasis) = length(x.id) ≡ 3


"""
    _filter_bases(basis, type)

Helper function to retrieve specific basis function information out of a `AHBasis` instance.
This is an internal function which is not expected to be used outside of this module. 

Arguments:
- `basis::AHBasis`: basis instance from which function is to be extracted.
- `type::DataType`: type of the basis functions to extract; e.g. `CylindricalBondEnvelope`.
"""
function _filter_bases(basis::AHBasis, T)
    functions = filter(i->i isa T, basis.basis.pibasis.basis1p.bases)
    if length(functions) == 0
        error("Could not locate basis function matching the supplied type")
    elseif length(functions) ≥ 2
        @warn "Multiple matching basis functions found, only the first will be returned"
    end
    return functions[1]
end

"""Extract and return the radial component of a `AHBasis` instance."""
radial(basis::AHBasis) = _filter_bases(basis, ACE.Rn1pBasis)

"""Extract and return the angular component of a `AHBasis` instance."""
angular(basis::AHBasis) = _filter_bases(basis, ACE.Ylm1pBasis)

"""Extract and return the categorical component of a `AHBasis` instance."""
categorical(basis::AHBasis) = _filter_bases(basis, ACE.Categorical1pBasis)

"""Extract and return the bond envelope component of a `AHBasis` instance."""
envelope(basis::AHBasis) = _filter_bases(basis, ACE.BondEnvelope)


# ╭───────┬──────────────────╮
# │ Basis │ IO Functionality │
# ╰───────┴──────────────────╯
"""
    write_dict(basis[,hash_basis])

Convert a `Basis` structure instance into a representative dictionary.

# Arguments
- `basis::Basis`: the `Basis` instance to parsed.
- `hash_basis::Bool`: ff `true` then hash values will be stored in place of
  the `SymmetricBasis` objects.
"""
function write_dict(basis::T, hash_basis=false) where T<:Basis
    return Dict(
        "__id__"=>"Basis",
        "basis"=>hash_basis ? string(hash(basis.basis)) : write_dict(basis.basis),
        "id"=>basis.id,
        "coefficients"=>write_dict(basis.coefficients),
        "mean"=>write_dict(basis.mean))

end


"""Instantiate a `Basis` instance from a representative dictionary."""
function ACEbase.read_dict(::Val{:Basis}, dict::Dict)
    return Basis(
        read_dict(dict["basis"]),
        Tuple(dict["id"]),
        read_dict(dict["coefficients"]),
        read_dict(dict["mean"]))
end


function Base.show(io::IO, basis::T) where T<:AHBasis
    print(io, "$(nameof(T))(id: $(basis.id), fitted: $(is_fitted(basis)))")
end


# ╔════════════════════════╗
# ║ ACE Basis Constructors ║
# ╚════════════════════════╝

@doc raw"""

    on_site_ace_basis(ℓ₁, ℓ₂, ν, deg, e_cutₒᵤₜ[, r0])

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
- `r0::AbstractFloat`: scaling parameter (typically set to the nearest neighbour distances).

# Returns
- `basis::SymmetricBasis`: ACE basis entity for modelling the specified interaction. 

"""
function on_site_ace_basis(ℓ₁::I, ℓ₂::I, ν::I, deg::I, e_cutₒᵤₜ::F, r0::F=2.5
    ) where {I<:Integer, F<:AbstractFloat}
    # Build i) a matrix indicating the desired sub-block shape, ii) the one
    # particle Rₙ·Yₗᵐ basis describing the environment, & iii) the basis selector.
    # Then instantiate the SymmetricBasis required by the Basis structure.
    return SymmetricBasis(
        SphericalMatrix(ℓ₁, ℓ₂; T=ComplexF64),
        RnYlm_1pbasis(maxdeg=deg, r0=r0, rcut=e_cutₒᵤₜ),
        SimpleSparseBasis(ν, deg))
end



function _off_site_ace_basis_no_sym(ℓ₁::I, ℓ₂::I, ν::I, deg::I, b_cut::F, e_cutₒᵤₜ::F=5.;
    λₙ::F=.5, λₗ::F=.5) where {I<:Integer, F<:AbstractFloat}


    # Bond envelope which controls which atoms are seen by the bond.
    @static if BOND_ORIGIN_AT_MIDPOINT
        env = CylindricalBondEnvelope(b_cut, e_cutₒᵤₜ, e_cutₒᵤₜ, floppy=false, λ=0.0)
    else
        env = CylindricalBondEnvelope(b_cut, e_cutₒᵤₜ, e_cutₒᵤₜ, floppy=false, λ=0.5)
    end
    
    # Categorical1pBasis is applied to the basis to allow atoms which are part of the
    # bond to be treated differently to those that are just part of the environment.
    discriminator = Categorical1pBasis([true, false]; varsym=:bond, idxsym=:bond)

    # The basis upon which the above entities act. Note that the internal cutoff "rin" must
    # be set to 0.0 to allow for atoms a the bond's midpoint to be observed. 
    RnYlm = RnYlm_1pbasis(
        maxdeg=deg, rcut=cutoff_env(env),
        trans=IdTransform(), rin=0.0)
        
    
    # Finally, construct and return the SymmetricBasis entity
    basis = SymmetricBasis(
        SphericalMatrix(ℓ₁, ℓ₂; T=ComplexF64),
        RnYlm * env * discriminator,
        SimpleSparseBasis(ν + 1, deg),
        filterfun=states -> _filter_offsite_be(states, deg, λₙ, λₗ))
    

    return basis
end


function _off_site_ace_basis_sym(ℓ₁::I, ℓ₂::I, ν::I, deg::I, b_cut::F, e_cutₒᵤₜ::F=5.;
    λₙ::F=.5, λₗ::F=.5) where {I<:Integer, F<:AbstractFloat}
 
    basis = _off_site_ace_basis_no_sym(ℓ₁, ℓ₂, ν, deg, b_cut, e_cutₒᵤₜ; λₙ=λₙ, λₗ=λₗ)

    if ℓ₁ == ℓ₂
        Uᵢ = let A = get_spec(basis.pibasis)
            U = adjoint.(basis.A2Bmap) * (_perm(A) * _dpar(A))
            (basis.A2Bmap + U) .* 0.5
        end

        # Purge system of linear dependence
        svdC = svd(_build_G(Uᵢ))
        rk = rank(Diagonal(svdC.S), rtol = 1e-7)          
        Uⱼ = sparse(Diagonal(sqrt.(svdC.S[1:rk])) * svdC.U[:, 1:rk]')
        U_new = Uⱼ * Uᵢ
 
        # construct symmetric offsite basis
        basis = SymmetricBasis(basis.pibasis, U_new, basis.symgrp, basis.real)

    elseif ℓ₁ > ℓ₂
        U_new = let A = get_spec(basis.pibasis)
            adjoint.(_off_site_ace_basis_no_sym(ℓ₂, ℓ₁, ν, deg, b_cut, e_cutₒᵤₜ; λₙ=λₙ, λₗ=λₗ).A2Bmap) * _perm(A) * _dpar(A)
        end

        basis = SymmetricBasis(basis.pibasis, U_new, basis.symgrp, basis.real)    
    end


    return basis
end

@doc raw"""

    off_site_ace_basis(ℓ₁, ℓ₂, ν, deg, b_cut[,e_cutₒᵤₜ, λₙ, λₗ])


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
- `λₙ::AbstractFloat`: 
- `λₗ::AbstractFloat`:

# Returns
- `basis::SymmetricBasis`: ACE basis entity for modelling the specified interaction. 

"""
function off_site_ace_basis(ℓ₁::I, ℓ₂::I, ν::I, deg::I, b_cut::F, e_cutₒᵤₜ::F=5.;
    λₙ::F=.5, λₗ::F=.5, symfix=true) where {I<:Integer, F<:AbstractFloat}
    # WARNING symfix might cause issues when applied to interactions between different species.
    # It is still not clear how appropriate this non homo-shell interactions.
    @static if SYMMETRY_FIX_ENABLED
        if symfix
            basis = _off_site_ace_basis_sym(ℓ₁, ℓ₂, ν, deg, b_cut, e_cutₒᵤₜ; λₙ=λₙ, λₗ=λₗ)
        else
            basis = _off_site_ace_basis_no_sym(ℓ₁, ℓ₂, ν, deg, b_cut, e_cutₒᵤₜ; λₙ=λₙ, λₗ=λₗ)
        end
    else
        basis = _off_site_ace_basis_no_sym(ℓ₁, ℓ₂, ν, deg, b_cut, e_cutₒᵤₜ; λₙ=λₙ, λₗ=λₗ)
    end

    return basis

end




"""
    _filter_offsite_be(states, max_degree[, λ_n=0.5, λ_l=0.5])

Cap the the maximum polynomial components.

This filter function should be passed, via the keyword `filterfun`, to `SymmetricBasis`
when instantiating.


# Arguments
- `states:?????`: Unknown, this is supplied by ase when used?
- `max_degree::Integer`: maximum polynomial degree.
- `λ_n::AbstractFloat`: 
- `λ_l::AbstractFloat`: 

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


adjoint(A::SphericalMatrix{ℓ₁, ℓ₂, LEN1, LEN2, T, LL}) where {ℓ₁, ℓ₂, LEN1, LEN2, T, LL} = SphericalMatrix(A.val', Val{ℓ₂}(), Val{ℓ₁}())
transpose(A::SphericalMatrix{ℓ₁, ℓ₂, LEN1, LEN2, T, LL}) where {ℓ₁, ℓ₂, LEN1, LEN2, T, LL} = SphericalMatrix(transpose(A.val), Val{ℓ₂}(), Val{ℓ₁}())
/(A::SphericalMatrix{ℓ₁, ℓ₂, LEN1, LEN2, T, LL}, b::Number) where {ℓ₁, ℓ₂, LEN1, LEN2, T, LL} = SphericalMatrix(A.val / b, Val{ℓ₁}(), Val{ℓ₂}())
Base.:(*)(A::SphericalMatrix{ℓ₁, ℓ₂, LEN1, LEN2, T, LL}, b::Number) where {ℓ₁, ℓ₂, LEN1, LEN2, T, LL} = SphericalMatrix(A.val * b, Val{ℓ₁}(), Val{ℓ₂}())


function _dpar(A)
    parity = spzeros(Int, length(A), length(A))
    
    for i=1:length(A)
        for j=1:length(A[i])
            if A[i][j].bond
                parity[i,i] = (-1)^A[i][j].l
                break
            end
        end
    end
 
    return parity
 end

"""
This function that takes an arbitrary named tuple and returns an identical copy with the
value of its "m" field inverted, if present. 
"""
@generated function _invert_m(i)

    filed_names = Meta.parse(
        join(
            [field_name ≠ :m ? "i.$field_name" : "-i.$field_name"
            for field_name in fieldnames(i)],
            ", ")
    )

    return quote
        $i(($(filed_names)))
    end
end

function _perm(A::Vector{Vector{T}}) where T
    # This function could benefit from some optimisation. However it is stable for now.

    # Dictionary mapping groups of A to their column index.
    D = Dict(sort(j)=>i for (i,j) in enumerate(A))

    # Ensure that there is no double mapping going on; as there is not clear way
    # to handel such an occurrence.
    @assert length(D) ≡ length(A) "Double mapping ambiguity present in \"A\""
 
    # Sparse zero matrix to hold the results
    P = spzeros(length(A), length(A))
 
    # Track which systems have been evaluated ahead of their designated loop via
    # symmetric equivalence. This done purely for the sake of speed. 
    done_by_symmetry = zeros(Int, length(A))
    
    for (i, A_group) in enumerate(A)
        # Skip over groups that have already been assigned during evaluation of their
        # symmetric equivalent.
        i in done_by_symmetry && continue

        # Construct the "m" inverted named tuple 
        U = sort([_invert_m(t) for t in A_group])

        # Identify which column the conjugate group `U` is associated.
        idx = D[U]

        # Update done_by_symmetry checklist to prevent evaluating the symmetrically
        # equivalent group. (if it exists)
        done_by_symmetry[i] = idx
        
        # Compute and assign the cumulative "m" parity term for the current group and
        # its symmetric equivalent (if present.)  
        P[i, idx] = P[idx, i] = (-1)^sum(o->o.m, A_group)

    end

    return P
end

function _build_G(U)
    # This function is highly inefficient and would benefit from a rewrite. It should be
    # possible to make use of sparsity information present in U ahead of time to Identify
    # when, where, and how many non-sparse points there are. 
    n_rows = size(U, 1)
    # A sparse matrix would be more appropriate here given the high degree of sparsity
    # present in `G`. However, this matrix is to be passed into the LinearAlgebra.svd
    # method which is currently not able to operate upon sparse arrays. Hence a dense
    # array is used here. 
    # G = spzeros(valtype(U[1].val), n_rows, n_rows)
    G = zeros(valtype(U[1].val), n_rows, n_rows)
    for row=1:n_rows, col=row:n_rows
        result = sum(coco_dot.(U[row, :], U[col, :]))
        if !iszero(result)
            G[row, col] = G[col, row] = result
        end
    end
    return G
end



"""
At the time of writing there is an oversite present in the SparseArrays module which
prevents it from correctly identifying if an operation is sparsity preserving. That
is to say, in many cases a sparse matrix will be converted into its dense form which
can have profound impacts on performance. This function exists correct this behaviour,
and should be removed once the fixes percolate through to the stable branch.  
"""
function Base.:(==)(x::SphericalMatrix, y::Integer)
    return !(any(real(x.val) .!= y) || any(imag(x.val) .!= y))
end


end