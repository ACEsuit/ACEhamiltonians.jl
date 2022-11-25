# All general io functionality should be placed in this file. With the exception of the
# `read_dict` and `write_dict` methods.  



"""
This module contains a series of functions that are intended to aid in the loading of data
from HDF5 structured databases. The code within this module is primarily intended to be
used only during the fitting of new models. Therefore, i) only loading methods are
currently supported, ii) the load target for each function is always the top level group
for each system.

A brief outline of the expected HDF5 database structure is provided below. Note that
arrays **must** be stored in column major format!

Database
├─System-1
│ ├─Structure
│ │ ├─atomic_numbers:
│ │ │   > A vector of integers specifying the atomic number of each atom present in the
│ │ │   > target system. Read by the function `load_atoms`.
│ │ │
│ │ ├─positions:
│ │ │   > A 3×N matrix, were N is the number of atoms present in the system, specifying
│ │ │   > cartesian coordinates for each atom. Read by the function `load_atoms`.
│ │ │
│ │ ├─lattice:
│ │ │   > A 3×3 matrix specifying the lattice systems lattice vector in column-wise
│ │ │   > format; i.e. columns loop over vectors. Read by the function `load_atoms`
│ │ │   > if and when present.
│ │ └─pbc:
│ │     > A boolean, or a vector of booleans, indicating if, or along which dimensions,
│ │     > periodic conditions are enforced. This is only read when the lattice is given. 
│ │     > This defaults to true for non-molecular/cluster cases. Read by `load_atoms`.
│ │
│ ├─Info
│ │ ├─Basis:
│ │ │   > Specifies the .
│ │ │   > This group contains one dataset for each species that specifies the 
│ │ │   > azimuthal quantum numbers of each shell present on that species. Read
│ │ │   > by the `load_basis_set_definition` method.
│ │ │
│ │ │
│ │ ├─Translations:
│ │ │   > A 3×N matrix specifying the cell translation vectors associated with the real 
│ │ │   > space Hamiltonian & overlap matrices. Only present when Hamiltonian & overlap
│ │ │   > matrices are given in their M×M×N real from. Read by `load_cell_translations`.
│ │ │ 
│ │ └─k-points:
│ │     > A 4×N matrix where N is the number of k-points. The first three rows specify
│ │     > k-points themselves with the final row specifying their associated weights.
│ │     > Read by `load_k_points_and_weights`. Only present for multi-k-point calculations. 
│ │
│ └─Data
│   ├─H:
│   │   > The Hamiltonian. Either an M×M matrix or an M×M×N real-space tensor; where M is 
│   │   > is the number of orbitals and N then number of primitive cell equivalents. Read
│   │   > in by the `load_hamiltonian` function.
│   │
│   ├─S:
│   │   > The Overlap matrix; identical in format to the Hamiltonian matrix. This is read
│   │   > in by the `load_overlap` function.
│   │
│   ├─total_energy:
│   │   > A single float value specifying the total system energy.
│   │
│   ├─fermi_level:
│   │   > A single float value specifying the total fermi level.
│   │
│   ├─forces:
│   │   > A 3×N matrix specifying the force vectors on each atom.
│   │
│   ├─H_gamma:
│   │   > This can be used to store the gamma point only Hamiltonian matrix when 'H' is
│   │   > used to store the real space matrix. This is mostly for debugging & testing.
│   │
│   └─S_gamma:
|       > Overlap equivalent of `H_gamma`
│
├─System-2
│ ├─Structure
│ │ └─ ...
│ │
│ ├─Info
│ │ └─ ...
│ │
│ └─Data
│   └─ ...
...
└─System-n
  └─ ...


Datasets and groups should provide information about what units the data they contain are
given in. This can be done through the of the HDF5 metadata `attributes`.  When calling
the various load methods within `DatabaseIO` the `src` Group argument must always point
to the target systems top level Group. In the example structure tree given above these
would be 'System-1', 'System-2', and 'System-n'.
"""
module DatabaseIO
using ACEhamiltonians
using HDF5: Group, h5open
using JuLIP: Atoms
using HDF5  # ← Can be removed once support for old HDF5 formats is dropped
using LinearAlgebra: pinv

# Developers Notes:
#   - The functions within this module are mostly just convenience wrappers for the HDF5
#     `read` method and as such very little logic is contained within. Thus no unit tests
#     are provided for this module at this time. However, this will change when and if
#     unit and write functionality are added.  
#     
# Todo:
#   - A version number flag should be added to each group when written to prevent the
#     breaking compatibility with existing databases every time an update is made.
#   - The unit information provided in the HDF5 databases should be made use of once
#     a grand consensus as to what internal units should be used.

export load_atoms, load_hamiltonian, load_overlap, gamma_only, load_k_points_and_weights, load_cell_translations, load_basis_set_definition, load_density_of_states, load_fermi_level


# Booleans stored by python are interpreted as Int8 by Julia rather than as booleans. Thus
# a cleaner is required.
_clean_bool(bool::I) where I<:Integer = Bool(bool)
_clean_bool(bool::Vector{<:Integer}) = convert(Vector{Bool}, bool)
_clean_bool(bool) = bool


function _recentre!(x, l, l_inv)
    x[:] = l_inv' * x .- 1E-8
    x[:] = l' * (x - round.(x) .+ 1E-8)
    nothing 
end

"""
    load_atoms(src)

Instantiate a `JuLIP.Atoms` object from an HDF5 `Group`.


# Arguments
- `src::Group`: top level HDF5 `Group` of the target system whose `Atoms` object is to be
  returned.
- `recentre:Bool`: By default, atoms are assumed to span the fractional coordinate domain
  [0.0, 1.0). Setting `recentre` to `true` will remap atomic positions to the fractional
  coordinate domain of [-0.5, 0.5). This is primarily used when interacting with real-space
  matrices produced by the FHI-aims code base.

# Returns
- `atoms::Atoms`: atoms object representing the structure of the target system.

"""
function load_atoms(src::Group; recentre=false)
    # Developers Notes:
    #   - Currently non-molecular/cluster systems are assumed to be fully periodic
    #     along each axis if no `pbc` condition is explicitly specified.
    # Todo:
    #   - Use unit information provided by the "positions" & "lattice" datasets.

    # All system specific structural data should be contained within the "Structure"
    # sub-group. Extract the group to a variable for ease of access.
    src = src["Structure"]    

    # Species and positions are always present to read them in
    species, positions = read(src["atomic_numbers"]), read(src["positions"])

    if haskey(src, "lattice")  # If periodic
        l = collect(read(src["lattice"])')
        if recentre
            l_inv = pinv(l)
            for x in eachcol(positions)
                _recentre!(x, l, l_inv)
            end
        end

        pbc = haskey(src, "pbc") ? _clean_bool(read(src["pbc"])) : true
        return Atoms(; Z=species, X=positions, cell=l, pbc=pbc)
    else  # If molecular/cluster
        return Atoms(; Z=species, X=positions)
    end
end


"""
    load_basis_set_definition(src)

Load the basis definition of the target system.

This returns a `BasisDef` dictionary which specifies the azimuthal quantum number of each
shell for each species.

# Arguments
- `src::Group`: top level HDF5 `Group` of the target system whose basis set definition is
  to be read.

# Returns
- `basis_def::BasisDef`: a dictionary keyed by species and valued by a vector specifying
  the azimuthal quantum number for each shell on said species.

"""
function load_basis_set_definition(src::Group)
    # Basis set definition is stored in "/Info/Basis" relative to the system's top level
    # group.
    src = src["Info/Basis"]
    # Basis set definition is stored as a series of vector datasets with names which
    # correspond the associated atomic number. 
    return BasisDef{Int}(parse(Int, k) => read(v)[2, :] for (k, v) in zip(keys(src), src))
end

"""
    load_k_points_and_weights(src)

Parse k-points and their weights.

# Arguments
- `src::Group`: top level HDF5 `Group` of the target system whose k-points & k-weights
  are to be returned.
  
# Returns
- `k_points::Matrix`: a 3×n matrix where n is the number of k-points.
- `k_weights::Vector`: a vector with a weight for each k-point.

# Warnings
This will error out for gamma-point only calculations.
 
"""
function load_k_points_and_weights(src::Group)
    # Read in the k-points and weights from the "/Info/k-points" matrix.
    knw = read(src["Info/k-points"])
    return knw[1:3, :], knw[4, :]
end


"""
    load_cell_translations(src)

Load the cell translation vectors associated with the real Hamiltonian & overlap matrices.
Relevant when Hamiltonian & overlap matrices are stored in their real N×N×M form, where N
is the number of orbitals per primitive cell and M is the number of cell equivalents.

# Arguments
- `src::Group`: top level HDF5 `Group` of the target system whose cell translation vectors
  are to be returned.

# Returns
- `T::Matrix`: a 3×M matrix where M is the number of primitive cell equivalents. 

# Notes
There is one translation vector for each translated cell; i.e. if the Hamiltonian matrix
is N×N×M then there will be M cell translation vectors. Here, the first translation
vector is always that of the origin cell, i.e. [0, 0, 0].

# Warnings
This will error out for gamma point only calculations or datasets in which the real
matrices are not stored.
"""
load_cell_translations(src::Group) = read(src["Info/Translations"])

"""
    load_hamiltonian(src)

Load the Hamiltonian matrix stored for the target system. This may be either an N×N single
k-point (commonly the gamma point) matrix, or an N×N×M real space matrix; where N is the
number or orbitals and M the number of unit cell equivalents.

# Arguments
- `src::Group`: top level HDF5 `Group` of the target system whose Hamiltonian matrix is
  to be returned.

# Returns
- `H::Array`: Hamiltonian matrix. This may be either an N×N matrix, as per the single
  k-point case, or an N×N×M array for the real space case. 
"""
load_hamiltonian(src::Group) = read(src["Data/H"])
# Todo: add unit conversion to `load_hamiltonian`

"""
    load_overlap(src)

Load the overlap matrix stored for the target system. This may be either an N×N single
k-point (commonly the gamma point) matrix, or an N×N×M real space matrix; where N is the
number or orbitals and M the number of unit cell equivalents.

# Arguments
- `src::Group`: top level HDF5 `Group` of the target system whose overlap matrix is to be
  returned.

# Returns
- `H::Array`: overlap matrix. This may be either an N×N matrix, as per the single
  k-point case, or an N×N×M array for the real space case. 
"""
load_overlap(src::Group) = read(src["Data/S"])
# Todo: add unit conversion to `load_overlap`


"""
    gamma_only(src)

Returns `true` if the stored Hamiltonian & overlap matrices are for a single k-point only.
Useful for determining whether or not one should attempt to read cell translations or
k-points, etc.
"""
gamma_only(src::Group) = !haskey(src, "Info/Translations")

# Get the gamma point only matrix; these are for debugging and are will be removed later.
load_hamiltonian_gamma(src::Group) = read(src["Data/H_gamma"])
# Todo: add unit conversion to `load_hamiltonian_gamma`
load_overlap_gamma(src::Group) = read(src["Data/S_gamma"])
# Todo: add unit conversion to `load_overlap_gamma`

"""
    load_density_of_states(src)

Load the density of states associated with the target system.  

# Arguments
- `src::Group`: top level HDF5 `Group` of the target system for which the density of
  states is to be returned.

# Returns
- `values::Vector`: density of states.
- `energies::Vector`: energies at which densities of states were evaluated relative to
   the fermi-level.
- `broadening::AbstractFloat`: broadening factor used by the smearing function.

"""
function load_density_of_states(src::Group)
    # Todo:
    #   - This currently returns units of eV for energy and 1/(eV.V_unit_cell) for DoS.
    return (
        read(src, "Data/DoS/values"),
        read(src, "Data/DoS/energies"),
        read(src, "Data/DoS/broadening"))
end


"""
    load_fermi_level(src)

Load the calculated Fermi level (chemical potential).

# Arguments
- `src::Group`: top level HDF5 `Group` of the target system for which the fermi level is
  to be returned.

# Returns
- `fermi_level::AbstractFloat`: the fermi level.
"""
function load_fermi_level(src)
  # Todo:
  #   - This really should make use of unit attribute that is provided. 
  return read(src, "Data/fermi_level")
end


# These functions exist to support backwards compatibility with previous database structures.
# They are not intended to be called by general users as they will eventually be excised.  
function _load_old_hamiltonian(path::String)
    return h5open(path) do database
        read(database, "aitb/H")[:, :]
    end
end

function _load_old_overlap(path::String)
  return h5open(path) do database
      read(database, "aitb/S")[:, :]
  end
end

function _load_old_atoms(path::String; groupname=nothing)
    h5open(path, "r") do fd
        groupname === nothing && (groupname = HDF5.name(first(fd)))
        positions = HDF5.read(fd, string(groupname,"/positions"))
        unitcell = HDF5.read(fd, string(groupname,"/unitcell"))
        species = HDF5.read(fd, string(groupname,"/species"))
        atoms = Atoms(; X = positions, Z = species,
                        cell = unitcell,
                        pbc = [true, true, true])
        return atoms
    end
end

end



