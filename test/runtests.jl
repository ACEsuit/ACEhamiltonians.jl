using ACEhamiltonians
using JSON, HDF5, JuLIP

using ACEhamiltonians: Data, Params, write_modeljson,
              write_atomjson, wm2fn
using ACEhamiltonians.Predict: predict_main, predict_onsite_HS
using ACEhamiltonians.Dictionary: recover_at,recover_model

# Step 1. Given parameters
file_name = ["data/FCC-MD-500K/SK-supercell-001.h5", "data/FCC-MD-500K/SK-supercell-002.h5", "data/FCC-MD-500K/SK-supercell-003.h5"]
index = [Vector(1:20:729), Vector(2:20:729), Vector(3:20:729)]
dat_train = Data(file_name,index)

file_name_test = ["data/FCC-MD-500K/SK-supercell-008.h5"]
index_test = [Vector(1:6:729)]
dat_test = Data(file_name_test,index_test)

rcutset = [9.0, 9.0, 9.0, 9.0, 9.0, 9.0]
maxdegset = [4, 4, 4, 4, 4, 4]
ordset = [2, 2, 2, 2, 2, 2]
λ = [1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7]
regtype = 2

par = Params(rcutset,maxdegset,ordset,λ,regtype,"LSQR")

# Step 2. Models training, generating corresponding errors
OnMW_H, OnMW_S, errset = predict_main(dat_train,dat_test,par,par)
isoff(OnMW_H)
ison(OnMW_S)

# Step 3. Save the models/errors/parameters/etc. in a json file
write_modeljson(OnMW_H,OnMW_S,errset)

# Step 4. Generate a structure.json file for testing
ii = 10
write_atomjson(ii)

# End of learning
##Step 5. Start predicting - read json file
Hmodelfile, Smodelfile, outfile = wm2fn(OnMW_H,OnMW_S)
infile = "structure_$ii.json"

at = recover_at( load_json(infile) )
Hmodel_whole = recover_model(load_json(Hmodelfile))
Smodel_whole = recover_model(load_json(Smodelfile))

# Step 6. predict H&S from models&positions
H = predict_onsite_HS(at,Hmodel_whole)
S = predict_onsite_HS(at,Smodel_whole)

h5open( outfile, "w" ) do fid
    write(fid, "H", H)
    write(fid, "S", S)
end

## We can recover the training&testing error from json file and the corresponding parameter grid from whole_onsite_models
#  e.g., the following
using ACEhamiltonians.Dictionary:recover_error
error_h = recover_error(load_json(Hmodelfile))
error_s = recover_error(load_json(Smodelfile))
# params_set = wmodels2params(Hmodel_whole,Smodel_whole)
