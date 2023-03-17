

```julia
# K-point for which the complex matrix is to be constructed for
k_point = [ 0,  0,  0]

# Load the atoms object of the system to make predictions for
atoms = h5open(database_path) do database
    load_atoms(database[target_systems[1]]; recentre=true)
end

# Specify the cell translation vectors; needed when wanting to compute real-space matrices
images = cell_translations(atoms, model)

# Predict the real-space matrix
predicted_real = predict(model, atoms, images)

# Construct the complex matrix
prdicted_k = real_to_complex(predicted_real, images, k_point)
```