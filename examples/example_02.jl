using ACEhamiltonians, HDF5, Serialization

# ╔════════════╗
# ║ Example  2 ║
# ╚════════════╝
# Band Structure Calculation

# Function to write band structure data to a file
function write_band_out(path, k_points, energies)
    data = hcat(k_points', energies')
    open(path, "w") do file
        for i in eachrow(data)
            for j in i
                write(file, "$j ")
            end
            write(file, "\n")

        end
    end
    nothing
end

# ╭────────────┬───────╮
# │ Section  1 │ Setup │
# ╰────────────┴───────╯
@info "Setup"

# Paths of the Hamiltonian and overlap models
H_model_path = "Si_H_model.bin"
S_model_path = "Si_S_model.bin"

# Path to the database containing the target system
database_path = "example_data.h5"
target_system = "0224"

# Define the band path segments
band_path = [
  # ([       start      ], [        end       ], n )
    ([0.25 , 0.50 , 0.75], [0.00 , 0.50 , 0.50], 50), # W → X
    ([0.00 , 0.50 , 0.50], [0.00 , 0.00 , 0.00], 50), # X → G
    ([0.00 , 0.00 , 0.00], [0.50 , 0.50 , 0.50], 50), # G → L
    ([0.50 , 0.50 , 0.50], [0.375, 0.375, 0.75], 50), # L → K
    ([0.375, 0.375, 0.75], [0.00 , 0.00 , 0.00], 50), # K → G
    ([0.00 , 0.00 , 0.00], [0.50 , 0.50 , 0.50], 50), # G → L
]


# ╭────────────┬─────────╮
# │ Section  2 │ Loading │
# ╰────────────┴─────────╯
@info "Loading Model and Data"

# Load ACEhamiltonians models
H_model = deserialize(H_model_path)
S_model = deserialize(S_model_path)

# Load the Atoms object of the target system
atoms = h5open(database_path) do database
    load_atoms(database[target_system])
end

# ╭────────────┬────────────╮
# │ Section  3 │ Predicting │
# ╰────────────┴────────────╯
@info "Calculating Band Path"


# Specify the cell translation vectors (needed for real-space matrices)
images = cell_translations(atoms, H_model)

# Predict the real-space Hamiltonian and overlap matrices
H_real = predict(H_model, atoms, images)
S_real = predict(S_model, atoms, images)

# Iterate through the band path segments
for (i, segment) in (enumerate(band_path))

    # Construct the k-points for the current band path segment
    k_points = hcat(LinRange(segment...)...);

    # Calculate the eigenvalues for each k-point in this band-path segment
    ϵ = band_structure(H_real, S_real, images, k_points);

    # Save the band path to a file
    write_band_out("band_$i.dat", k_points, ϵ)
end

@info "Finished"

# Results may be visualised using the "tools/plot_band.py" script.