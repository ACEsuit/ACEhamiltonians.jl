# using Distributed, SlurmClusterManager
# addprocs(SlurmManager())

# @everywhere using ACEhamiltonians, HDF5, Serialization

using ACEhamiltonians, HDF5, Serialization

# ╔════════════╗
# ║ Example  1 ║
# ╚════════════╝
# Note that this example is intended as a placeholder

# ╭────────────┬──────────────╮
# │ Section  1 │ Construction │
# ╰────────────┴──────────────╯
@info "Constructing Model"

# Provide a label for the model (this should be H or S)
model_type = "dm"

# Define the basis definition

basis_definition = Dict(1=>[0, 0, 0, 0, 1, 1, 2], 8=>[0, 0, 0, 1, 1, 2])

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
database_path = "./Data/Datasets/smpl10_md_w6.h5"

# Names of the systems to which the model should be fitted
target_systems = h5open(database_path) do database keys(database) end

# Open up the HDF5 database within which the target data is stored
h5open(database_path) do database
    # Load the target system(s) for fitting
    systems = [database[target_system] for target_system in target_systems]

    # Perform the fitting operation
    fit!(model, systems; recentre=true)
end

# database = h5open(database_path)
# systems = [database[target_system] for target_system in target_systems]
# show(fit!(model, systems; recentre=true))
# close(database)

filename = split(basename(database_path), ".")[begin]
# serialize("./Models/"*filename*".bin", model)
serialize("./Models/"*filename*"_single.bin", model)