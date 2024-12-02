using BenchmarkTools, Serialization, Random
using Distributed, SlurmClusterManager
addprocs(28)
@everywhere begin
    using ACEhamiltonians, HDF5, Serialization
    using Statistics
    using ACEhamiltonians.DatabaseIO: load_hamiltonian_gamma, load_overlap_gamma, load_density_matrix_gamma
    using JuLIP: Atoms
    using PeriodicTable
    using Statistics
    include("./utils.jl")
end

# -----------------------
# |***general setting***|
# ----------------------- 

database_path = "./Data/Datasets/dyn-wd-300K_3.h5"
output_path = "./Result/dm_H2O_2_300K_rcut_14_n_512"
nsamples = 512 #5200
mkpath(output_path)

@info "Constructing Model"

# Provide a label for the model (this should be H, S or dm)
model_type = "dm"

# Define the basis definition
basis_definition = Dict(1=>[0, 0, 0, 0, 1, 1, 2], 8=>[0, 0, 0, 1, 1, 2])

# On site parameter deceleration
on_site_parameters = OnSiteParaSet(
    # Maximum correlation order
    GlobalParams(2),
    # Maximum polynomial degree
    GlobalParams(14), #(6),
    # Environmental cutoff radius
    GlobalParams(14.), #(10.),
    # Scaling factor "r₀"
    GlobalParams(0.9) #(2.5)
)

# Off-site parameter deceleration
off_site_parameters = OffSiteParaSet(
    # Maximum correlation order
    GlobalParams(1),
    # Maximum polynomial degree
    GlobalParams(14), #(6),
    # Bond cutoff radius
    GlobalParams(14.), #(10.),
    # Environmental cutoff radius
    GlobalParams(7.), #(5.),
)

# initialisation
model = Model(basis_definition, on_site_parameters, off_site_parameters, model_type)

# -----------------------
# |***fitting model***|
# ----------------------- 
@info "Fitting Model"

# Names of the systems to which the model should be fitted
target_systems = h5open(database_path) do database keys(database) end
rng = MersenneTwister(1234)
@assert nsamples <= length(target_systems) "nsample should be smaller or equal to nsample" 
target_systems = shuffle(rng, target_systems)[begin:nsamples]
target_systems = [target_systems[i:5:end] for i in 1:5]
train_systems = vcat(target_systems[1:end-1]...)
test_systems = target_systems[end]

# Open up the HDF5 database within which the target data is stored
h5open(database_path) do database
    # Load the target system(s) for fitting
    systems = [database[system] for system in train_systems]

    # Perform the fitting operation
    fit!(model, systems; recentre=true)
end

filename = split(basename(database_path), ".")[begin]
serialize(joinpath(output_path, filename*".bin"), model)

# model = deserialize(joinpath(output_path, filename*".bin"))

################################# test set #########################################
#prediction
atoms = h5open(database_path) do database
    [load_atoms(database[system]) for system in test_systems]
    end
images = cell_translations.(atoms, Ref(model))
predicted = predict.(Ref(model), atoms, images)

#groud truth data
get_matrix = Dict(  # Select an appropriate function to load the target matrix
"H"=>load_hamiltonian, "S"=>load_overlap, "dm"=>load_density_matrix,
"Hg"=>load_hamiltonian_gamma, "Sg"=>load_overlap_gamma, "dmg"=>load_density_matrix_gamma)[model.label]
gt = h5open(database_path) do database
    [get_matrix(database[system]) for system in test_systems]
end

matrix_dict = Dict("predicted"=>predicted, "gt"=>gt, "atoms"=>atoms)
open(joinpath(output_path, "matrix_dict_test.jls"), "w") do file
    serialize(file, matrix_dict)
end

error = predicted-gt
data_dict = get_error_dict(error, atoms, model)
open(joinpath(output_path, "data_dict_test.jls"), "w") do file
    serialize(file, data_dict)
end


################################# training set #########################################
#prediction
atoms = h5open(database_path) do database
    [load_atoms(database[system]) for system in train_systems]
    end
images = cell_translations.(atoms, Ref(model))
predicted = predict.(Ref(model), atoms, images)

#groud truth data
get_matrix = Dict(  # Select an appropriate function to load the target matrix
"H"=>load_hamiltonian, "S"=>load_overlap, "dm"=>load_density_matrix,
"Hg"=>load_hamiltonian_gamma, "Sg"=>load_overlap_gamma, "dmg"=>load_density_matrix_gamma)[model.label]
gt = h5open(database_path) do database
    [get_matrix(database[system]) for system in train_systems]
end

matrix_dict = Dict("predicted"=>predicted, "gt"=>gt, "atoms"=>atoms)
open(joinpath(output_path, "matrix_dict_train.jls"), "w") do file
    serialize(file, matrix_dict)
end

error = predicted-gt
data_dict = get_error_dict(error, atoms, model)
open(joinpath(output_path, "data_dict_train.jls"), "w") do file
    serialize(file, data_dict)
end