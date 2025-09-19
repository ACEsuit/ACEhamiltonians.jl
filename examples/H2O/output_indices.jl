using HDF5, Random
using Serialization

output_path = "./Result/output_indices"
indices_dict = Dict()

for database_path in ["./Data/Datasets/H2O_H_aims.h5", "./Data/Datasets/dyn-wd-300K_3.h5", "./Data/Datasets/dyn-wd-500K_3.h5"]

    nsamples = 512 #5200
    # Names of the systems to which the model should be fitted
    target_systems = h5open(database_path) do database keys(database) end
    rng = MersenneTwister(1234)
    @assert nsamples <= length(target_systems) "nsample should be smaller or equal to nsample" 
    target_systems = shuffle(rng, target_systems)[begin:nsamples]
    target_systems = [target_systems[i:5:end] for i in 1:5]
    train_systems = vcat(target_systems[1:end-1]...)
    test_systems = target_systems[end]

    data_name = split(basename(database_path), ".")[1]
    indices_dict[data_name] = Dict("train"=>parse.(Ref(Int), train_systems).+1, "test"=>parse.(Ref(Int), test_systems).+1)

end

open(joinpath(output_path, "indices_dict.jls"), "w") do file
    serialize(file, indices_dict)
end