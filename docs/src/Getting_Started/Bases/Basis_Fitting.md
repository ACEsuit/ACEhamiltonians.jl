# Basis Fitting
Individual `Basis` instances may be fitted by supplying the `Basis` and fitting `data` to the `fit!` function. This single basis `fit!` function takes the two aforementioned positional arguments along with three optional keyword arguments: `solver` which specifies the solver to use during fitting which defaults to `LSQR`, `λ` is the degree of regularisation to be applied which defaults to `1E-7`, and `enable_mean` with permits a mean offset to be applied while this defaults to `false` it is advised to enable it when fitting on-site bases.
```julia

# Path to the database within which the fitting data is stored
database_path = "example_data.h5"

# Name/path of/to the system to which fitting data should be drawn 
target_system = "0224"

# Load the real-space Hamiltonian matrix, atoms object, cell translation vectors, and basis set definition
H, atoms, images, basis_definition = h5open(database_path) do database
    target = database[target_system]
    load_hamiltonian(target), load_atoms(target; recentre=true), load_cell_translations(target), load_basis_set_definition(target)
end

# Get the species of the system - it can be done in several ways and here just comes one
# This step can be skipped if all atoms in this system are of the same type
species = unique(atoms.Z)

# Construct a pair of on and off-site bases
on_site_basis = Basis(on_site_ace_basis(0, 1, 2, 4, 6.0; species = (try species catch nothing end)), (14, 3, 4))
off_site_basis = Basis(off_site_ace_basis(1, 1, 1, 4, 8.0, 4.0; species = (try species catch nothing end)), (14, 14, 6, 6))

# Gather all relevant data for fitting 
on_site_data = get_dataset(H, atoms, on_site_basis, basis_definition, images)
off_site_data = get_dataset(H, atoms, off_site_basis, basis_definition, images)

# Perform the fitting operation
fit!(on_site_basis, on_site_data)
fit!(off_site_basis, off_site_data)

```

The fitting data must be provided as a `DataSet` instance. These structures contain all of the data required to carry out the fitting process; specifically the target sub-blocks and state objects required to perform a fitting operation. The `get_dataset` convenience function auto-collects all relevant data, which can be filtered as and when needed by the user. This avoids users having to manually collect and construct `DataSet` instances themselves.
