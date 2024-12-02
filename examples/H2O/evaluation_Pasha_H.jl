using Distributed, SlurmClusterManager, SparseArrays
addprocs(28)
#addprocs(SlurmManager())
@everywhere begin
    using ACEhamiltonians, HDF5, Serialization
    using Statistics
    using ACEhamiltonians.DatabaseIO: load_hamiltonian_gamma, load_overlap_gamma, load_density_matrix_gamma
end

model_path = "./Result/H_12_light/smpl10_md_w12.bin" 

w6_path = "./Data/Datasets/smpl10_md_w6.h5"
w12_path = "./Data/Datasets/smpl10_md_w12.h5"
w21_path = "./Data/Datasets/smpl10_md_w21.h5"
w101_path = "./Data/Datasets/smpl10_md_w101.h5"


function get_matrices_dict(model_path::String, data_path::String)

    model = deserialize(model_path)

    get_matrix = Dict(  # Select an appropriate function to load the target matrix
    "H"=>load_hamiltonian, "S"=>load_overlap, "dm"=>load_density_matrix,
    "Hg"=>load_hamiltonian_gamma, "Sg"=>load_overlap_gamma, "dmg"=>load_density_matrix_gamma)[model.label]

    target_systems = h5open(data_path) do database keys(database) end

    atoms = h5open(data_path) do database
        [load_atoms(database[system]) for system in target_systems]
    end

    images = [cell_translations(atoms, model) for atoms in atoms]

    predicted = [sparse(dropdims(pred, dims=3)) for pred in predict.(Ref(model), atoms, images)]
    
    groud_truth = h5open(data_path) do database
        [sparse(dropdims(get_matrix(database[system]), dims=3)) for system in target_systems]
    end

    data_dict = Dict{String, Dict}()

    for (system, pred, gt) in zip(target_systems, predicted, groud_truth)

        data_dict[system] = Dict("gt"=>gt, "pred"=>pred)

    end

    return data_dict

end


function evaluate_on_data(model_path::String, data_path::String)

    matrices_dict = get_matrices_dict(model_path, data_path)
    dict_path = joinpath(dirname(model_path), split(split(basename(data_path), ".")[1], "_")[end]*"_dict.jls")
    open(dict_path, "w") do file
        serialize(file, matrices_dict)
    end

end


for data_path in [w6_path, w12_path, w21_path]
    evaluate_on_data(model_path, data_path)
end