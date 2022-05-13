using ACEhamiltonians
using JSON, HDF5, JuLIP

using ACEhamiltonians: Data, Params, write_modeljson,
              write_atomjson, wm2fn, fname2index
using ACEhamiltonians.Predict: predict_main, predict_onsite_HS, predict_offsite_HS, wmodels2err
using ACEhamiltonians.Dictionary: recover_at,recover_model

## On-site H model (S is also trainned but we might not need it)

# If we want hybrid training, here is one suitable place to involve BCC data in!
file_name = ["data/FCC-MD-500K/SK-supercell-001.h5", "data/FCC-MD-500K/SK-supercell-002.h5", "data/FCC-MD-500K/SK-supercell-003.h5"]
index = [Vector(1:3:729), Vector(2:3:729), Vector(3:3:729)]
dat_train = Data(file_name,index)
# to see how many training sites do we have
get_sites(dat_train)
# test data given
file_name_test = ["data/FCC-MD-500K/SK-supercell-008.h5"]
index_test = [Vector(1:6:729)]
dat_test = Data(file_name_test,index_test)

# Parameter grids; see ?Params for more information.
rcutset = [9.0, 9.0, 9.0, 9.0, 9.0, 9.0]
maxdegset = [6, 6, 6, 6, 6, 6]
ordset = [2, 2, 2, 2, 2, 3]
λ = [1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7]
regtype = 2
par_H = Params(rcutset,maxdegset,ordset,λ,regtype,"LSQR")

rcutset = [6.0, 6.0, 6.0, 6.0, 6.0, 6.0]
maxdegset = [4, 4, 4, 4, 4, 4]
ordset = [2, 2, 2, 2, 2, 2]
λ = [1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7]
regtype = 2
par_S = Params(rcutset,maxdegset,ordset,λ,regtype,"LSQR")

# Now we use the length of params to indicate on-site or off-site 
# (6 - on-site; 9 - off-site)
ison(par_H) == true
isoff(par_H) == false
# We understood that on-site S model doesn't need to be fitted, but I still found it is not that "diagonal"... 
# I'll just keep on-site S fitting here for now (with fairly low degree).
# The followings are somewhat equivalent to `predict_main`, but give us additional info. about training data!
OnModel_H, OnModel_S, data_train = params2wmodels(dat_train,par_H,par_S)
errset_on = wmodels2err(OnModel_H,OnModel_S,dat_test,data_train) 
# model -> .json file
write_modeljson(OnModel_H,OnModel_S,errset_on)
# get the corresponding file ID!
Hmodelfile, Smodelfile, outfile = wm2fn(OnModel_H,OnModel_S)

## Offsite H&S model trainning

# Same thing, given traning set
file_name = ["data/FCC-MD-500K/SK-supercell-001.h5", "data/FCC-MD-500K/SK-supercell-002.h5"]
# 600 - #sites in each file; 2 - evenly sampling
index = fname2index.(file_name,600,2)
dat_train = Data(file_name,index)
index_test = [map(rand, (1:728, 1:729)) for i = 1:600]
for (i,k) in enumerate(index_test)
    if k[2] == k[1]
        index_test[i] = (k[1],k[2]+1)
    elseif k[2] < k[1]
        index_test[i] = (k[2],k[1])
    end
end
dat_test = Data(file_name,index_test)
# Another way to give parameter grid, see `?Params` for more details
rcutset = [[4.0 4.0 6.0;4.0 5.5 8.0;8.0 8.0 10.5], [4.5 6.0 8.0;6.0 8.0 10.5], [6.0 8.0 10.0], [4.5 6.0; 8.0 6.0; 8.0 10.5], [6.0 8.0;8.0 10.5], [8.0 10.5], Matrix([6.0 8.0 10.0]'), Matrix([8.0 10.5]'), [10.0 for i=1:1,j=1:1]]
maxdegset = [6, 6, 6, 6, 6, 6, 6, 6, 6]
ordset = [2, 2, 2, 2, 2, 2, 2, 2, 2]
λ = [1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7]
regtype = 2
par_H = Params(rcutset,maxdegset,ordset,λ,regtype,"LSQR")
ison(par_H)
isoff(par_H)

rcutset = [[4.0 4.0 6.0;4.0 5.5 8.0;8.0 8.0 10.5], [4.5 6.0 8.0;6.0 8.0 10.5], [6.0 8.0 10.0], [4.5 6.0; 8.0 6.0; 8.0 10.5], [6.0 8.0;8.0 10.5], [8.0 10.5], Matrix([6.0 8.0 10.0]'), Matrix([8.0 10.5]'), [10.0 for i=1:1,j=1:1]]
maxdegset = [8, 8, 8, 8, 8, 8, 8, 8, 8]
ordset = [1, 1, 1, 1, 1, 1, 1, 1, 1] # owing to env independency, no matter how we choose this parameters, the basis may not change as it depends only on maxdeg
λ = [1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7]
regtype = 2
par_S = Params(rcutset,maxdegset,ordset,λ,regtype,"LSQR")
# !!! Environment independency has already been incorporated in main code (See ACEhamiltonians-Fiiting-`λ_n, λ_l` part)
OffModel_H, OffModel_S, data_train = params2wmodels(dat_train,par_H,par_S)
errset_off = wmodels2err(OffModel_H,OffModel_S,dat_test,data_train) 
# model -> .json file
write_modeljson(OffModel_H,OffModel_S,errset_off)
# get the corresponding file ID!
Hmodelfile, Smodelfile, outfile = wm2fn(OffModel_H,OffModel_S)

