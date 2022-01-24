using LinearAlgebra
using Printf
using ACEhamiltonians:Data
using ACEhamiltonians.DataProcess: get_atoms

function wm2pardata(ModelW::TBModelWhole)

    rcutset = ModelW.params.rcutset
    maxdegset = ModelW.params.maxdegset
    ordset =  ModelW.params.ordset
    regset = ModelW.params.regset
    regtype = ModelW.params.regtype

    filenumber = ModelW.data.file_number
    index = ModelW.data.index
    type = ModelW.data.type

    return Params(rcutset,maxdegset,ordset,regset,regtype), Data(filenumber, index, type)

end

function wm2fn(ModelW_H::TBModelWhole, ModelW_S::TBModelWhole)
    key_dict = Dict(     "file_name" => ModelW_H.data.file_name,
                             "index" => ModelW_H.data.index,
                         "rcutset_H" => ModelW_H.params.rcutset,
                       "maxdegset_H" => ModelW_H.params.maxdegset,
                          "ordset_H" => ModelW_H.params.ordset,
                          "regpar_H" => ModelW_H.params.regset,
                         "rcutset_S" => ModelW_S.params.rcutset,
                       "maxdegset_S" => ModelW_S.params.maxdegset,
                          "ordset_S" => ModelW_S.params.ordset,
                          "regpar_S" => ModelW_S.params.regset,
                           "regtype" => ModelW_H.params.regtype,
                            "solver" => ModelW_H.params.solver )
    return "$(hash(key_dict))_H.json", "$(hash(key_dict))_S.json", "$(hash(key_dict))_predictedHS.h5"
end

# added just enable us to write single model (with which model_S and model_H is not tied anymore)
function write_modeljson(ModelW::TBModelWhole)
   dict_mw = write_dict(ModelW)
   save_model_h = JSON.json(dict_mw)
   outfile1, outfile3 = wm2fn(ModelW_H,ModelW_H)
   open(outfile1, "w") do f
      write(f, save_model_h)
   end
end

function write_modeljson(ModelW_H::TBModelWhole,ModelW_S::TBModelWhole,err = nothing)
   dict_mwh = write_dict(ModelW_H)
   dict_mws = write_dict(ModelW_S)
   if err != nothing
      dict_errh = Dict( "RMSE_train" => err[1][1],
                         "RMSE_test" => err[1][2] )
      dict_errs = Dict( "RMSE_train" => err[2][1],
                         "RMSE_test" => err[2][2] )
      merge!(dict_mwh,dict_errh)
      merge!(dict_mws,dict_errs)
   end
   save_model_h = JSON.json(dict_mwh)
   outfile1, outfile2, outfile3 = wm2fn(ModelW_H, ModelW_S)
   open(outfile1, "w") do f
      write(f, save_model_h)
   end
   save_model_s = JSON.json(dict_mws)
   open(outfile2, "w") do f
      write(f, save_model_s)
   end
end

function write_atomjson(infile::String, outfile::String)
   _, atoms = get_atoms(infile)
   dict_atom = JuLIP.write_dict(atoms)
   save_structure = JSON.json(dict_atom)
   open(outfile, "w") do f
      write(f, save_structure)
   end
end

function write_atomjson(i::Int64) 
   fmt = @sprintf("%03d", i)
   write_atomjson("data/FCC-MD-500K/SK-supercell-$idx.h5",
                  "structure_$fmt.json")
end

function fname2atom(fname::String)
   data = []
   HDF5.h5open(fname, "r") do fd
   groupnames = []
   for obj in fd
      push!(groupnames, HDF5.name(obj))
   end
   group_name = groupnames[1]
   dataset_names = []
   for obj in fd[group_name]
      push!(dataset_names, HDF5.name(obj))
   end
   positions = HDF5.read(fd, string(group_name,"/positions"))
   unitcell = HDF5.read(fd, string(group_name,"/unitcell"))
   species = HDF5.read(fd, string(group_name,"/species"))

   push!(data, JuLIP.Atoms(; X = positions, Z = species,
                            cell = unitcell,
                            pbc = [true, true, true]))
   end
   return data[1]
end

"""
fname2index(fname,N,flag) maps file name to an index set with (≤)N tuples
flag = 1, random sampling (omit all blocks having bonds longer than 10.0)
flag = 2, evenly sampling (w.r.t. bond length), default method
flag = 3, maxval sampling, choose the shortest N bonds
"""

function fname2index(fname::String, N::Int64, flag = 2)
   at = fname2atom(fname)
   nlist = JuLIP.neighbourlist(at,20.0)
   if flag == 1
      set = []
      while length(set) < N
         (i,j) = map(rand, (1:728, 1:729))
         if j<i
            (i,j) = (j,i)
         end
         neigh_i = JuLIP.Potentials.neigsz(nlist,at,i)
         idx = ACEhamiltonians.DataProcess.get_bond(at,i,j)
         if norm(neigh_i[2][idx]) < 10.0 && ((i,j)∉set)
            set = [set; (i,j)]
         end
      end
   elseif flag == 2
      normlist = zeros(265356)
      for i = 1:728, j = (i+1):729
         neigh_i = JuLIP.Potentials.neigsz(nlist,at,i)
         idx = ACEhamiltonians.DataProcess.get_bond(at,i,j)
         normlist[ij2t(i,j)] = norm(neigh_i[2][idx])
      end
      order = sortperm(normlist)
      idx = findfirst(x -> x>10.0, normlist[order])
      diff = floor(Int,idx/N)
      set = t2ij.(order[1:diff:idx])
      if length(set) > N
         set = set[1:N]
      end
   elseif flag == 3
      normlist = zeros(265356)
      for i = 1:728, j = (i+1):729
         neigh_i = JuLIP.Potentials.neigsz(nlist,at,i)
         idx = ACEhamiltonians.DataProcess.get_bond(at,i,j)
         normlist[ij2t(i,j)] = norm(neigh_i[2][idx])
      end
      order = sortperm(normlist)
      set = t2ij.(order[1:N])
   end
   
   if length(set) == N
      return Vector{Tuple{Int64, Int64}}(set)
   elseif length(set) < N
      @warn("Only $(length(set)) indices are selected! ")
      return Vector{Tuple{Int64, Int64}}(set)
   elseif length(set) > N
      @warn(" A BUG EXISTS ! ")
      return Vector{Tuple{Int64, Int64}}(set)
   end
end

S(i) = Int( (728 + (728 - i + 2)) * (i-1) / 2 )
ij2t(i,j) = Int( S(i) + j - i)
function t2ij(t)
   i = 1
   while S(i) < t
      i += 1
   end
   i = i - 1
   j = i + t - S(i)
   return (i,j)
end
   
function rescaling_mat(dat::Data)
   data_train = data_read(dat.file_name, dat.index)
   if ison(dat)
      return rescaling_on(data_train)
   elseif isoff(dat)
      return rescaling_off(data_train)
   end
end

function rescaling_on(data_train)
   rescaling_mat = zeros(14,14)
   rescaling_mat[1:3,1:3] = [ maximum([ norm(data_train[1][val][k][i,j]) for k = 1:length(data_train[1][val]) ]) for i =1:3, j=1:3 ]
   for i = 1:2, j = 1:3
      rescaling_mat[4+3(i-1):6+3(i-1),j] = maximum([ norm(data_train[2][1][k][i,j]) for k = 1:length(data_train[2][1]) ]) * ones(3)
   end
   for j = 1:3
      rescaling_mat[10:14,j] = maximum([ norm(data_train[3][1][k][1,j]) for k = 1:length(data_train[3][1]) ]) * ones(5)
   end
   for i = 1:2, j = 1:2
      rescaling_mat[4+3(i-1):6+3(i-1),4+3(j-1):6+3(j-1)] = maximum([ norm(data_train[4][1][k][i,j]) for k = 1:length(data_train[4][1]) ]) * ones(3,3)
   end
   for j = 1:2
      rescaling_mat[10:14,4+3(j-1):6+3(j-1)] = maximum([ norm(data_train[5][1][k][1,j]) for k = 1:length(data_train[5][1]) ]) * ones(5,3)
   end
   rescaling_mat[10:14,10:14] = maximum([ norm(data_train[6][1][k][1,1]) for k = 1:length(data_train[6][1]) ]) * ones(5,5)
   rescaling_mat = rescaling_mat+rescaling_mat'
   for i=1:14
      rescaling_mat[i,i] = rescaling_mat[i,i]/2
   end
   return rescaling_mat
end

function rescaling_off(data_train)
   rescaling_mat_H = zeros(14,14)
   rescaling_mat_S = zeros(14,14)
   rescaling_mat_H[1:3,1:3] = [ maximum([ norm(data_train_off[1][1][k][i,j]) for k = 1:length(data_train_off[1][1]) ]) for i =1:3, j=1:3 ]
   rescaling_mat_S[1:3,1:3] = [ maximum([ norm(data_train_off[1][2][k][i,j]) for k = 1:length(data_train_off[1][2]) ]) for i =1:3, j=1:3 ]
   for i = 1:2, j = 1:3
       rescaling_mat_H[4+3(i-1):6+3(i-1),j] = maximum([ norm(data_train_off[2][1][k][i,j]) for k = 1:length(data_train_off[2][1]) ]) * ones(3)
       rescaling_mat_S[4+3(i-1):6+3(i-1),j] = maximum([ norm(data_train_off[2][2][k][i,j]) for k = 1:length(data_train_off[2][2]) ]) * ones(3)
   end
   for j = 1:3
       rescaling_mat_H[10:14,j] = maximum([ norm(data_train_off[3][1][k][1,j]) for k = 1:length(data_train_off[3][1]) ]) * ones(5)
       rescaling_mat_S[10:14,j] = maximum([ norm(data_train_off[3][2][k][1,j]) for k = 1:length(data_train_off[3][2]) ]) * ones(5)
   end
   for i = 1:3, j = 1:2
       rescaling_mat_H[i,4+3(j-1):6+3(j-1)] = maximum([ norm(data_train_off[4][1][k][i,j]) for k = 1:length(data_train_off[4][1]) ]) * ones(1,3)
       rescaling_mat_S[i,4+3(j-1):6+3(j-1)] = maximum([ norm(data_train_off[4][2][k][i,j]) for k = 1:length(data_train_off[4][2]) ]) * ones(1,3)
   end
   for i = 1:2, j = 1:2
       rescaling_mat_H[4+3(i-1):6+3(i-1),4+3(j-1):6+3(j-1)] = maximum([ norm(data_train_off[5][1][k][i,j]) for k = 1:length(data_train[5][1]) ]) * ones(3,3)
       rescaling_mat_S[4+3(i-1):6+3(i-1),4+3(j-1):6+3(j-1)] = maximum([ norm(data_train_off[5][2][k][i,j]) for k = 1:length(data_train[5][2]) ]) * ones(3,3)
   end
   for j = 1:2
       rescaling_mat_H[10:14,4+3(j-1):6+3(j-1)] = maximum([ norm(data_train_off[6][1][k][1,j]) for k = 1:length(data_train[6][1]) ]) * ones(5,3)
       rescaling_mat_S[10:14,4+3(j-1):6+3(j-1)] = maximum([ norm(data_train_off[6][2][k][1,j]) for k = 1:length(data_train[6][2]) ]) * ones(5,3)
   end
   for i = 1:3
       rescaling_mat_H[i,10:14] = maximum([ norm(data_train_off[7][1][k][i,1]) for k = 1:length(data_train_off[7][1]) ]) * ones(1,5)
       rescaling_mat_S[i,10:14] = maximum([ norm(data_train_off[7][2][k][i,1]) for k = 1:length(data_train_off[7][2]) ]) * ones(1,5)
   end
   for i = 1:2
       rescaling_mat_H[4+3(i-1):6+3(i-1),10:14] = maximum([ norm(data_train_off[8][1][k][i,1]) for k = 1:length(data_train_off[8][1]) ]) * ones(3,5)
       rescaling_mat_S[4+3(i-1):6+3(i-1),10:14] = maximum([ norm(data_train_off[8][2][k][i,1]) for k = 1:length(data_train_off[8][2]) ]) * ones(3,5)
   end
   rescaling_mat_H[10:14,10:14] = maximum([ norm(data_train_off[9][1][k][1,1]) for k = 1:length(data_train_off[9][1]) ]) * ones(5,5)
   rescaling_mat_S[10:14,10:14] = maximum([ norm(data_train_off[9][2][k][1,1]) for k = 1:length(data_train_off[9][2]) ]) * ones(5,5)
   return rescaling_mat_H, rescaling_mat_S
end

