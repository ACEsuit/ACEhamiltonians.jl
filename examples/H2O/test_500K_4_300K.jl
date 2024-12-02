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
output_path = "./Result/H_H2O_2_500K_4_300K"
model_path = "./Result/H_H2O_2_500K_rcut_10/dyn-wd-500K_3.bin"
nsamples = 512 #5000
mkpath(output_path)

@info "Constructing Model"

model = deserialize(model_path)

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

#prediction
atoms = h5open(database_path) do database
    [load_atoms(database[system]) for system in test_systems] # train_systems]
    end
images = cell_translations.(atoms, Ref(model))
predicted = predict.(Ref(model), atoms, images)

#groud truth data
get_matrix = Dict(  # Select an appropriate function to load the target matrix
"H"=>load_hamiltonian, "S"=>load_overlap, "dm"=>load_density_matrix,
"Hg"=>load_hamiltonian_gamma, "Sg"=>load_overlap_gamma, "dmg"=>load_density_matrix_gamma)[model.label]
gt = h5open(database_path) do database
    [get_matrix(database[system]) for system in test_systems] # train_systems]
end

matrix_dict = Dict("predicted"=>predicted, "gt"=>gt, "atoms"=>atoms)
open(joinpath(output_path, "matrix_dict.jls"), "w") do file
    serialize(file, matrix_dict)
end

error = predicted-gt
data_dict = get_error_dict(error, atoms, model)
open(joinpath(output_path, "data_dict.jls"), "w") do file
    serialize(file, data_dict)
end