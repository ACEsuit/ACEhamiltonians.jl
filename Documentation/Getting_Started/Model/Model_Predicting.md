
Within the `ACEhamiltonians` framework, predictions are made via the `predict` methods.
To construct the real space matrix for a given system one may call the `predict` method and provide it with i) a `model` with which to make predictions, ii) an `JuLIP.Atoms` entity representing to system for which predictions are to be made, and iii) the cell translation vectors.
The real-space matrix may then be used to construct the complex matrix at a specific k-point via the `real_to_complex` method.
```julia
# K-point for which the complex matrix is to be constructed for
k_point = [ 0,  0,  0]

# Load the JuLIP.Atoms object of the system to make predictions for
atoms = h5open(database_path) do database
	# The argument recentre` is only required when requiring comparability
	# with the FHI-aims real-space matrix format.
    load_atoms(database[target_systems[1]]; recentre=true)
end

# Specify the cell translation vectors; needed when wanting to compute real-space matrices
images = cell_translations(atoms, model)

# Predict the real-space matrix
predicted_real = predict(model, atoms, images)

# Construct the complex matrix
prdicted_k = real_to_complex(predicted_real, images, k_point)
```
The cell translation vectors control which cells are included when constructing the real space matrix.
The `cell_translations` method can be used to make a reliable estimate based on the distance cutoffs present within the model.