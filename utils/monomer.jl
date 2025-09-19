using Serialization
using JuLIP
using LinearAlgebra: norm, pinv
using StatsBase
using Statistics
using Plots

# model_path = "/home/c/chenqian3/ACEhamiltonians/H2O_PASHA/H2O_Pasha/Result/H_H2O_2_500K_rcut_10/dyn-wd-500K_3.bin"

# bond_cutoff = 1.
num_bond = 2

Hartree2meV=27211.4

basis_definition = Dict(1=>[0, 0, 0, 0, 1, 1, 2], 8=>[0, 0, 0, 1, 1, 2])
n_1 =  sum([2*i+1 for i in basis_definition[1]])
n_8 =  sum([2*i+1 for i in basis_definition[8]])
basis_num = Dict(1=>n_1, 8=>n_8)
mol_basis_num = n_1*2 + n_8

data_file = "/home/c/chenqian3/ACEhamiltonians/H2O_PASHA/H2O_Pasha/Result/H_H2O_1_rcut_6/matrix_dict_test.jls"
data_dict = open(data_file, "r") do file
    deserialize(file)
end

predicts = data_dict["predicted"].*Hartree2meV
gts = data_dict["gt"].*Hartree2meV
atoms_list = data_dict["atoms"]

# model = deserialize(model_path)
# images_list = cell_translations.(atoms_list, Ref(model))
# pairs = [ for image in images for atom in atoms ]

# errors_list = [abs.(predict-gt) for (predict, gt) in zip(predicts, gts)]
# error_intra =  cat([cat(errors[1:mol_basis_num, 1:mol_basis_num], errors[1+mol_basis_num:end, 1+mol_basis_num:end], dims=3) for errors in errors_list]..., dims=3)
# error_inter =  cat([cat(errors[1+mol_basis_num:end, 1:mol_basis_num], errors[1:mol_basis_num, 1+mol_basis_num:end], dims=3) for errors in errors_list]..., dims=3)
# mae_intra = mean(error_intra)
# mae_inter = mean(error_inter)

# gts_intra = cat([cat(gt[1:mol_basis_num, 1:mol_basis_num], gt[1+mol_basis_num:end, 1+mol_basis_num:end], dims=3) for gt in gts]..., dims=3)
# gts_inter = cat([cat(gt[1+mol_basis_num:end, 1:mol_basis_num], gt[1:mol_basis_num, 1+mol_basis_num:end], dims=3) for gt in gts]..., dims=3)
# gt_intra = std(gts_intra)
# gt_inter = std(gts_inter)

errors = abs.(cat((gts.-predicts)..., dims=3)) # cat([abs.(predict-gt) for (predict, gt) in zip(predicts, gts)]..., dims=3)
gts = cat(gts..., dims=3)
mae = mean(errors)
gt = std(gts)
mae_norm = mae/gt

mae = mean(errors)
println("mae, mae_norm: $mae, $mae_norm")

# mae_norm_intra = mae_intra/gt_intra
# mae_norm_inter = mae_inter/gt_inter

mae_plot = dropdims(mean(errors, dims=3), dims=3)
mae_norm_plot = dropdims(mean(errors, dims=3)./std(gts, dims=3), dims=3)

p=heatmap(mae_plot, size=(800, 750), color=:jet, title="Monomer MAE (meV)", titlefont=font(20, "times", :bold), xticks=([7, 21.5, 36.5],
 ["O", "H", "H"]), yticks=([7, 21.5, 36.5], ["O", "H", "H"]), tickfont=font(18, "Courier", :bold),)
vline!(p, [14.5], color=:grey, linestyle=:dash, linewidth=2, label=false) 
vline!(p, [29.5], color=:grey, linestyle=:dash, linewidth=2, label=false) 
hline!(p, [14.5], color=:grey, linestyle=:dash, linewidth=2, label=false)  
hline!(p, [29.5], color=:grey, linestyle=:dash, linewidth=2, label=false) 
savefig(joinpath(dirname(data_file), "mae.png"))
display(p)

p=heatmap(mae_norm_plot, size=(800, 750), color=:jet, title="Normalized monomer MAE", titlefont=font(20, "times", :bold), xticks=([7, 21.5, 36.5],
 ["O", "H", "H"]), yticks=([7, 21.5, 36.5], ["O", "H", "H"]), tickfont=font(18, "Courier", :bold))
vline!(p, [14.5], color=:grey, linestyle=:dash, linewidth=2, label=false) 
vline!(p, [29.5], color=:grey, linestyle=:dash, linewidth=2, label=false) 
hline!(p, [14.5], color=:grey, linestyle=:dash, linewidth=2, label=false)  
hline!(p, [29.5], color=:grey, linestyle=:dash, linewidth=2, label=false) 
p1=plot(p, right_margin=13Plots.mm)
savefig(p1, joinpath(dirname(data_file), "mae_norm.png"))
display(p)



# mae_intra_plot = dropdims(mean(error_intra, dims=3), dims=3)
# mae_inter_plot = dropdims(mean(error_inter, dims=3), dims=3)

# p=heatmap(mae_intra_plot, size=(800, 750), color=:jet, title="Intra molecule MAE (meV)", titlefont=font(20, "times", :bold), xticks=([7, 21.5, 36.5],
#  ["O", "H", "H"]), yticks=([7, 21.5, 36.5], ["O", "H", "H"]), tickfont=font(18, "Courier", :bold),)
# vline!(p, [14.5], color=:grey, linestyle=:dash, linewidth=2, label=false) 
# vline!(p, [29.5], color=:grey, linestyle=:dash, linewidth=2, label=false) 
# hline!(p, [14.5], color=:grey, linestyle=:dot, linewidth=2, label=false)  
# hline!(p, [29.5], color=:grey, linestyle=:dot, linewidth=2, label=false) 
# savefig(joinpath(dirname(data_file), "mae_intra_plot.png"))
# display(p)

# p=heatmap(mae_inter_plot, size=(800, 750), color=:jet, title="Inter molecule MAE (meV)", titlefont=font(20, "times", :bold), xticks=([7, 21.5, 36.5],
#  ["O", "H", "H"]), yticks=([7, 21.5, 36.5], ["O", "H", "H"]), tickfont=font(18, "Courier", :bold),)
# vline!(p, [14.5], color=:grey, linestyle=:dash, linewidth=2, label=false) 
# vline!(p, [29.5], color=:grey, linestyle=:dash, linewidth=2, label=false) 
# hline!(p, [14.5], color=:grey, linestyle=:dot, linewidth=2, label=false)  
# hline!(p, [29.5], color=:grey, linestyle=:dot, linewidth=2, label=false) 
# savefig(joinpath(dirname(data_file), "mae_inter_plot.png"))
# display(p)



# mae_norm_intra = dropdims(mean(error_intra, dims=3)./std(gts_intra, dims=3), dims=3)
# mae_norm_inter = dropdims(mean(error_inter, dims=3)./std(gts_inter, dims=3), dims=3)

# p=heatmap(mae_norm_intra, size=(800, 750), color=:jet, title="Normalized intra molecule MAE", titlefont=font(20, "times", :bold), xticks=([7, 21.5, 36.5],
#  ["O", "H", "H"]), yticks=([7, 21.5, 36.5], ["O", "H", "H"]), tickfont=font(18, "Courier", :bold))
# vline!(p, [14.5], color=:grey, linestyle=:dash, linewidth=2, label=false) 
# vline!(p, [29.5], color=:grey, linestyle=:dash, linewidth=2, label=false) 
# hline!(p, [14.5], color=:grey, linestyle=:dot, linewidth=2, label=false)  
# hline!(p, [29.5], color=:grey, linestyle=:dot, linewidth=2, label=false) 
# p1=plot(p, right_margin=13Plots.mm)
# savefig(p1, joinpath(dirname(data_file), "mae_norm_intra_plot.png"))
# display(p)

# p=heatmap(mae_norm_inter, size=(800, 750), color=:jet, title="Normalized intra molecule MAE", titlefont=font(20, "times", :bold), xticks=([7, 21.5, 36.5],
#  ["O", "H", "H"]), yticks=([7, 21.5, 36.5], ["O", "H", "H"]), tickfont=font(18, "Courier", :bold), clims=(0, 1))
# vline!(p, [14.5], color=:grey, linestyle=:dash, linewidth=2, label=false) 
# vline!(p, [29.5], color=:grey, linestyle=:dash, linewidth=2, label=false) 
# hline!(p, [14.5], color=:grey, linestyle=:dot, linewidth=2, label=false)  
# hline!(p, [29.5], color=:grey, linestyle=:dot, linewidth=2, label=false) 
# p1=plot(p, right_margin=10Plots.mm)
# savefig(p1, joinpath(dirname(data_file), "mae_norm_inter_plot.png"))
# display(p)




# for (atoms, images, errors) in zip(atoms_list, images_list, errors_list)
#     shift_vectors = collect(eachrow(images' * atoms.cell))
#     pos = hcat(collect.(atoms.X)...)
#     dis = reshape(pos, size(pos, 1), size(pos, 2), 1) .- reshape(pos, size(pos, 1), 1, size(pos, 2))
#     dis = reshape(dis, (1, size(dis)...))
#     dis = dis .- reshape(hcat(collect.(shift_vectors)...)', length(shift_vectors), 3, 1, 1)
#     dis = dropdims(mapslices(norm, dis; dims=2); dims=2)
#     dis = permutedims(dis, (3,2,1))
#     for i_idx, (X₁, Z₁) in enumerate(atoms.X, atoms.Z)
#         if Z₁ == 8
#             H_OO = extract_block(errors, i_idx, i_idx, findall([images[:,i]==[0, 0, 0] for i in range(1,size(images,2))])[1], basis_num, atoms)
#             idx_line = partialsortperm(dis[i_idx, :, :][:], 1:3; rev=false)[2:end]
#             j_idxes, image_idxes =  rem.(idx_line, Ref(length(atoms.Z))), Int.(floor.(idx_line./Ref(length(atoms.Z))))
#             O_indices = i_idx
#             H_indices = j_idxes
#             i_idxes = [i_idx for i in 1: num_bond]
#             for j_idx in j_idxes
#                 push!(partialsortperm(dis[j_idx, i_idx, :], 1, rev=false), image_idxes)
#                 push!(j_idx, i_idxes)
#                 push!(i_idx, j_idxes)
#             push!(partialsortperm(dis[j_idx, i_idx, :], 1, rev=false), image_idxes)

        


#             for (X₂, Z₂) in zip(atoms.X, atoms.Z)
#                 if Z₂==1 and 
                    
#                     norm.( X₁ - X₂ + shift_vectors)



        

#     mask = norm.(atoms.X - atoms.X + shift_vectors[block_idxs[3, :]]) .<= distance


    




# function extract_block(matrix::Array{Float64, 3}, i_idx::Int, j_idx::Int, image_idx::Int, basis_num::Dict, atoms::Atoms)
#     idx_begin = vcat([1],cumsum([basis_num[i] for i in atoms.Z])[1:end-1].+1)
#     idx_end = cumsum([basis_num[i] for i in atoms.Z])
#     return matrix[idx_begin[i_idx]: idx_end[i_idx], idx_begin[j_idx]: idx_end[j_idx], image_idx]
# end




