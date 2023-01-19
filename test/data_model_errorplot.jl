using ACEhamiltonians
using JSON, HDF5, JuLIP, Statistics, Plots
using ACEhamiltonians: Data, Params, write_modeljson,
              write_atomjson, wm2fn, fname2index
using ACEhamiltonians.Predict: predict_main, predict_offsite_HS
using ACEhamiltonians.Dictionary: recover_at, recover_model, recover_error

# Step 0. Specify training and test set
file_name = ["ACEhamiltonians-data/training_data/BCC/SK-supercell-001.h5", "ACEhamiltonians-data/training_data/FCC/SK-supercell-002.h5"] # Specify the data file that we read
index = fname2index.(file_name,1000) # Read 1000 blocks from each file specified above as the training set - 2000 data in total
dat = Data(file_name,index)
index_test = vec([ k for k in Iterators.product(Vector(1:20:729).+1,Vector(1:20:729).+2)]) # Specify a fixed test set
dat_test = Data(file_name,index_test)

# We do not need S models so just keep its parameters fixed - NOTE: This will lead to different model id but that is of course not a big deal
rcutset = repeat([10.0],9)
maxdegset = repeat([4],9)
ordset = repeat([0],9)
λ = repeat([1e-7],9)
regtype = 2
par_S = Params(rcutset,maxdegset,ordset,λ,regtype,"LSQR")

# Loop over the degree set - here we fix the order to be 1
ordset = repeat([1],9)
degset = 6:14
rmse_train = zeros(9,length(degset)) # the 9 rows refer to ss, sp, sd, ps, pp, pd, ds, dp, dd, respectively
rmse_test  = zeros(9,length(degset))
for (i,deg) in enumerate(degset)
    # Step 1. Given parameters
    maxdegset = repeat([deg],9)
    par_H = Params(rcutset,maxdegset,ordset,λ,regtype,"LSQR")
    # Step 2. Models training, generating corresponding errors
    println("Performing fit...")
    OffMW_H, OffMW_S, errset = predict_main(dat,dat_test,par_H,par_S)
    # Step 3. Save models/errors/parameters/etc. in a json file
    println("Saving degree $deg models...")
    write_modeljson(OffMW_H,OffMW_S,errset)
    println("Done.")
    # Step 4. Extract errors: read json file
    Hmodelfile, Smodelfile, outfile = wm2fn(OffMW_H,OffMW_S)
    @show Hmodelfile
    # Hmodel_whole = recover_model(load_json(Hmodelfile)) # command to read models from a json file
    error = recover_error(load_json(Hmodelfile))
    rmse_train[:,i] = mean.(error[1])
    rmse_test[:,i] = mean.(error[2])
end

# Plot
labels = ["ss","sp","sd","ps","pp","pd","ds","dp","dd"]
colors = [1, 2, 3, 2, 4, 5, 3, 5, 6]
shapes = [:none, :diamond, :diamond, :rect, :none, :diamond, :rect, :rect, :none]
plt = plot(degset,rmse_train[1,:],yscale=:log,color=1,shape=:none,label=labels[1],xlabel="Maximum degree",ylabel="RMSE / ev",title="Order 1. Offsite H RMSE",size=(600,800))
plot!(degset,rmse_test[1,:],yscale=:log,ls=:dash,color=1,label=false)
for i in 2:9
    plot!(degset,rmse_train[i,:],yscale=:log,color=colors[i],shape=shapes[i],label=labels[i])
    plot!(degset,rmse_test[i,:],yscale=:log,ls=:dash,color=colors[i],label=false)
end
ylims!(10^(-3.7), 1e-1)
plt
