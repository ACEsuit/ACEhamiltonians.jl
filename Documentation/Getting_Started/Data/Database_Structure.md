

The `ACEhamiltonians.IO` module contains a series of functions that are intended to aid in the loading of data from HDF5 structured databases.
While this module offers read functionally is does not offer any write subroutines as of the time of writing.
Furthermore, all loading methods are hard-code to look for the requested data in a specific location relative to top level group of each system.  
Thus a all HDF5 databases must strictly adhere to specified file structure specification to be readable.

A brief outline of the expected HDF5 database structure is provided below. Note that
arrays **must** be stored in column major format and names are case sensitive!
```tree
Database
├─System_1
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
│ │ │
│ │ └─pbc:
│ │     > A boolean, or a vector of booleans, indicating if, or along which dimensions,
│ │     > periodic conditions are enforced. This is only read when the lattice is given. 
│ │     > This defaults to true for non-molecular/cluster cases. Read by `load_atoms`.
│ │
│ ├─Info
│ │ ├─Basis:
│ │ │   > This group contains one dataset for each species that specifies the 
│ │ │   > azimuthal quantum numbers of each shell present on that species. Read
│ │ │   > by the `load_basis_set_definition` method.
│ │ │
│ │ │
│ │ ├─Translations:
│ │ │   > A 3×N matrix specifying the cell translation vectors associated with the real 
│ │ │   > space Hamiltonian & overlap matrices. Only present when Hamiltonian & overlap
│ │ │   > matrices are given in their M×M×N real from. Read by `load_cell_translations`.
│ │ │   > Should be integers specifying the cell indices, rather than cartesian vectors.
│ │ │   > The origin cell, [0, 0, 0], must always be first!
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
├─System_2
│ ├─Structure
│ │ └─ ...
│ │
│ ├─Info
│ │ └─ ...
│ │
│ └─Data
│   └─ ...
...
└─System_n
  └─ ...
```
When calling the various load methods within `DatabaseIO` the `src` Group argument must always point to the target systems top level Group.
In the example structure tree given above these would be 'System_1', 'System_2', and 'System_n'.
Datasets and groups should provide information about what units the data they contain are given in.
This can be done through the of the HDF5 metadata `attributes`.
However, units are not yet fully supported. 

An example is provided below demonstrating how one may load data from a HDF5 database.
```julia
# Path to the database from which data is to be loaded
database_path = "/home/ajmhpc/Documents/Work/Projects/ACEtb/Data/Si/Construction/example_data.h5"
# Path, relative to the top level, to the group from which data is to be extracted
target_system = "0224"

# Open the HDF5 database file 
H, S, atoms, images = h5open(database_path) do database
    # Select the target group
    target = database[target_system]
    # Load the Hamiltonian and overlap matrices followed by the atoms object
    # and the cell translation vectors
    (load_hamiltonian(target), load_overlap(target), 
    load_atoms(target; recentre=true), load_cell_translations(target))
end
```