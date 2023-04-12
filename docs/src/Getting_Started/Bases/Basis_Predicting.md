# SubModel Predicting
Predictions can be made for individual bases by providing a `SubModel` instance along with a state-vector (a vector of `AbstractState` instance) to the `predict` function.
```julia


# System upon which predictions are to be made.
database_path = "example_data.h5"
target_system = "0224"

# Load the atoms object
atoms = h5open(database_path) do database
    load_atoms(database[target_system]; recentre=true)
end

# Get the on-site state representing the environment about atom 1
on_site_state = get_state(1, atoms)

# Get the off-site state representing the environment about the bond
# between atom 1 in the origin cell and atom 2 in image [0, 0, 1].
off_site_state = get_state(1, 2, atoms, envelope(off_site_pp_model), [0, 0, 1])

# Make the predictions
on_site_sp_block = predict(on_site_sp_basis, on_site_state)
off_site_pp_block = predict(off_site_pp_basis, off_site_state)
```
Predictions can also be made for multiple blocks simultaneously by providing a vector containing multiple state-vectors.
