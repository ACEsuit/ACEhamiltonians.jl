using BenchmarkTools, Serialization, Random
using Distributed, SlurmClusterManager
# addprocs(SlurmManager())
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

database_path = "./Data/Datasets/H2O_H_aims.h5"
data_name = split(basename(database_path), ".")[1]
output_path = joinpath("./Result/hyperparameter_searching_S_H2O_1", data_name*"_crossval") 
nsamples = 512
mkpath(output_path)

# ---------------------------
# |function to get the model|
# ---------------------------

nfolds = 5

model_type = "S"

basis_definition = Dict(1=>[0, 0, 0, 0, 1, 1, 2], 8=>[0, 0, 0, 1, 1, 2])

function get_model(basis_definition::BasisDef, d_max::Int64=14, r_cutoff::Float64=6.,
                    label::String="H", meta_data::Union{Dict, Nothing}=nothing)

    on_site_parameters = OnSiteParaSet(
        # Maximum correlation order
        GlobalParams(1), # (2),
        # Maximum polynomial degree
        GlobalParams(d_max),
        # Environmental cutoff radius
        GlobalParams(r_cutoff),
        # Scaling factor "r₀"
        GlobalParams(0.9) 
    )
    
    # Off-site parameter deceleration
    off_site_parameters = OffSiteParaSet(
        # Maximum correlation order
        GlobalParams(1),
        # Maximum polynomial degree
        GlobalParams(d_max),
        # Bond cutoff radius
        GlobalParams(r_cutoff),
        # Environmental cutoff radius
        GlobalParams(r_cutoff/2.0),
    )

        model = Model(basis_definition, on_site_parameters, off_site_parameters, label, meta_data)

        return model

        @info "finished buiding a model with d_max=$(d_max), r_cutoff=$(r_cutoff)"

end

# ---------------------------------
# |***cross validation function***|
# ---------------------------------
# single model evaluation
function evaluate_single(model::Model, database_path::String, train_systems::Vector{String}, test_systems::Vector{String})

    model = deepcopy(model)

    #model fitting
    h5open(database_path) do database
        # Load the target system(s) for fitting
        systems = [database[system] for system in train_systems]

        # Perform the fitting operation
        fit!(model, systems; recentre=true)
    end

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

    error = predicted-gt

    return error, atoms

end


function cross_validation(model::Model, database_path::String, nfolds::Int=5)


    target_systems = h5open(database_path) do database keys(database) end
    rng = MersenneTwister(1234)
    target_systems = shuffle(rng, target_systems)[begin:nsamples]
    target_systems = [target_systems[i:nfolds:end] for i in 1:nfolds]

    errors = []
    atomsv = []
    for fold in 1:nfolds   # nfolds:nfolds   # 1:nfolds
        train_systems = vcat(target_systems[1:fold-1]..., target_systems[fold+1:end]...)
        test_systems = target_systems[fold]
        error, atoms = evaluate_single(model, database_path, train_systems, test_systems)
        push!(errors, error)
        push!(atomsv, atoms)
    end
    errors = vcat(errors...)
    atomsv = vcat(atomsv...)

    data_dict = get_error_dict(errors, atomsv, model)

    return data_dict

end


# ---------------------
# |initialize the test| 
# ---------------------
data_dict = Dict{Tuple, Dict}()
output_path_figs = joinpath(output_path, "figures")

for d_max in 6:14
    r_cutoff = 6.
    dict_name = (d_max, r_cutoff)
    model = get_model(basis_definition, d_max, r_cutoff, model_type)
    data_dict_sub = cross_validation(model, database_path, nfolds)
    data_dict[dict_name] = data_dict_sub
    open(joinpath(output_path, "data_dict_co_1_d_max_r_6.jls"), "w") do file
        serialize(file, data_dict)
    end
    plot_hyperparams(data_dict, "d_max", output_path_figs)
    @info "finished testing a model with d_max=$(d_max), r_cutoff=$(r_cutoff)"
end