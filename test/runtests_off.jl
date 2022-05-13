using ACEhamiltonians
using JSON, HDF5, JuLIP

using ACEhamiltonians: Data, Params, write_modeljson,
              write_atomjson, wm2fn, fname2index
using ACEhamiltonians.Predict: predict_main, predict_offsite_HS
using ACEhamiltonians.Dictionary: recover_at,recover_model

## Offsite implementation

# Step 1. Given parameters
file_name = ["data/FCC-MD-500K/SK-supercell-001.h5", "data/FCC-MD-500K/SK-supercell-002.h5"]
index = fname2index.(file_name,600,2)
dat = Data(file_name,index)
ison(dat)
isoff(dat)
index_test = [map(rand, (1:728, 1:729)) for i = 1:600]
for (i,k) in enumerate(index_test)
    if k[2] == k[1]
        index_test[i] = (k[1],k[2]+1)
    elseif k[2] < k[1]
        index_test[i] = (k[2],k[1])
    end
end
dat_test = Data(file_name,index_test)

rcutset = repeat([10.0],9)
maxdegset = repeat([4],9)
ordset = repeat([1],9)
λ = repeat([1e-7],9)
regtype = 2
par_H = Params(rcutset,maxdegset,ordset,λ,regtype,"LSQR")

rcutset = repeat([10.0],9)
maxdegset = repeat([4],9)
ordset = repeat([1],9)
λ = repeat([1e-7],9)
regtype = 2
par_S = Params(rcutset,maxdegset,ordset,λ,regtype,"LSQR")

# Step 2. Models training, generating corresponding errors
OffMW_H, OffMW_S, errset = predict_main(dat,dat_test,par_H,par_S)
isoff(OffMW_H)
ison(OffMW_H)

# Step 3. Save the models/errors/parameters/etc. in a json file
write_modeljson(OffMW_H,OffMW_S,errset)

# Step 4. Generate a structure.json file for testing
ii = 10
write_atomjson(ii)

## End of learning

# Step 5. Start predicting - read json file
Hmodelfile, Smodelfile, outfile = wm2fn(OffMW_H,OffMW_S)
infile = "structure_$ii.json"

at = recover_at( load_json(infile) )

Hmodel_whole = recover_model(load_json(Hmodelfile))
Smodel_whole = recover_model(load_json(Smodelfile))
