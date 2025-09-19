using Serialization
using JuLIP
using LinearAlgebra
using Statistics

data_file = "/home/c/chenqian3/ACEhamiltonians/H2O_PASHA/H2O_Pasha/Result/H_H2O_2_500K_rcut_10/data_dict_test.jls"
data_dict = open(data_file, "r") do file
    deserialize(file)
end

data_dict["MAE"]["all"]*27211.4
data_dict["MAE"]["on"]["all"]*27211.4
data_dict["MAE"]["off"]["all"]*27211.4



matrix_file = "/home/c/chenqian3/ACEhamiltonians/H2O_PASHA/H2O_Pasha/Result/dm_H2O_2_500K_rcut_10/matrix_dict_test.jls"
data_dict = open(matrix_file, "r") do file
    deserialize(file)
end

data_dict["predicted"].-data_dict["gt"]

mean(norm.(data_dict["predicted"].-data_dict["gt"], 2))
0.13157953160181063



matrix_file = "/home/c/chenqian3/ACEhamiltonians/H2O_PASHA/H2O_Pasha/Result/dm_H2O_2_300K_rcut_10/matrix_dict_test.jls"


data_file  = "/home/c/chenqian3/ACEhamiltonians/H2O_PASHA/H2O_Pasha/Result/hyperparameter_searching_dm_r_cutoff/dyn-wd-500K_3/data_dict_d_max_14_r_cutoff.jls"
data_dict = open(data_file, "r") do file
    deserialize(file)
end





matrix_file = "/home/c/chenqian3/ACEhamiltonians/H2O_PASHA/H2O_Pasha/Result/dm_H2O_2_500K_rcut_14_n_512/matrix_dict_test.jls"
matrix_dict = open(matrix_file, "r") do file
    deserialize(file)
end
mean(norm.(matrix_dict["predicted"].-matrix_dict["gt"], 2))

