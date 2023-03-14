using ACEhamiltonians, HDF5

# ╔════════════╗
# ║ Example  1 ║
# ╚════════════╝
# Note that this example is intended as a placeholder

# ╭────────────┬──────────────╮
# │ Section  1 │ Construction │
# ╰────────────┴──────────────╯
@info "Constructing Model"

# Provide a label for the model (this should be H or S)
model_type = "H"

# Define the basis definition
basis_definition = Dict(14=>[0, 0, 0, 1, 1, 1, 2])

# On site parameter deceleration
on_site_parameters = OnSiteParaSet(
    # Maximum correlation order
    GlobalParams(2),
    # Maximum polynomial degree
    GlobalParams(6),
    # Environmental cutoff radius
    GlobalParams(10.),
    # Scaling factor "r₀"
    GlobalParams(2.5)
)

# Off-site parameter deceleration
off_site_parameters = OffSiteParaSet(
    # Maximum correlation order
    GlobalParams(1),
    # Maximum polynomial degree
    GlobalParams(6),
    # Bond cutoff radius
    GlobalParams(10.),
    # Environmental cutoff radius
    GlobalParams(5.),
)

# initialisation
model = Model(basis_definition, on_site_parameters, off_site_parameters, model_type)

# ╭────────────┬─────────────────────╮
# │ Section  2 │ Fitting (Automated) │
# ╰────────────┴─────────────────────╯
@info "Fitting Model"

# Path to the database within which the fitting data is stored
database_path = "/home/ajmhpc/Documents/Work/Projects/ACEtb/Data/Si/Construction/batch_0.h5"
# Names of the systems to which the model should be fitted
target_systems = ["0224"]

# Open up the HDF5 database within which the target data is stored
h5open(database_path) do database
    # Load the target system(s) for fitting
    systems = [database[target_system] for target_system in target_systems]

    # Perform the fitting operation
    fit!(model, systems; recentre=true)
end


# ╭────────────┬────────────────────────╮
# │ Section  3 │ Predicting (Automated) │
# ╰────────────┴────────────────────────╯
@info "Predicting"

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

@info "Finished"