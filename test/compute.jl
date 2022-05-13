using ACEhamiltonians
using JSON, HDF5, JuLIP, LinearAlgebra

using ACEhamiltonians.Predict: predict_onsite_HS, predict_offsite_HS, predict_offsite_HS_sym, predict_offsite_HS_redundant, predict_offsite_HS_redundant_sym
using ACEhamiltonians.Dictionary: recover_at, recover_model, recover_error

if length(ARGS) != 4
    error("Usage: compute.jl ONSITE_MODEL_ID OFFSITE_MODEL_ID INFILE.json OUTFILE.h5")
end

onsite_model_ID = ARGS[1]
offsite_model_ID = ARGS[2]
infile = ARGS[3]
outfile = ARGS[4]

@show onsite_model_ID
@show offsite_model_ID
@show infile
@show outfile

const hartree2ev = 27.211386024367243

at = recover_at( load_json(infile) )

println("Loading onsite model...")
onsiteHmodelfile = onsite_model_ID * "_H.json"
onsiteHmodel_data = load_json(onsiteHmodelfile)
onsiteHmodel_whole = read_dict(onsiteHmodel_data)
onsiteHerror = recover_error(onsiteHmodel_data)
println("onsite H error ", onsiteHerror)

onsiteSmodelfile = onsite_model_ID * "_S.json"
onsiteSmodel_data = load_json(onsiteSmodelfile)
onsiteSmodel_whole = recover_model(onsiteSmodel_data)
onsiteSerror = recover_error(onsiteSmodel_data)
println("onsite S error ", onsiteSerror)

println("Predicting onsite Hamiltonian...")
blocks = [365]
H_on = predict_onsite_HS(at, onsiteHmodel_whole, blocks)
S_on = predict_onsite_HS(at, onsiteSmodel_whole, blocks)

println("Loading offsite model...")
offsiteHmodelfile = offsite_model_ID * "_H.json"
offsiteHmodel_data = load_json(offsiteHmodelfile)
offsiteHmodel_whole = read_dict(offsiteHmodel_data)
offsiteHerror = recover_error(offsiteHmodel_data)
println("offsite H error ", offsiteHerror)

offsiteSmodelfile = offsite_model_ID * "_S.json"
offsiteSmodel_data = load_json(offsiteSmodelfile)
offsiteSmodel_whole = read_dict(load_json(offsiteSmodelfile))
offsiteSerror = recover_error(offsiteSmodel_data)
println("offsite S error ", offsiteSerror)

println("Predicting offsite H and S...")
blocks =  [ (365, i) for i = 1:729 ]
H_off = predict_offsite_HS_sym(at, offsiteHmodel_whole, blocks)
S_off = predict_offsite_HS_sym(at, offsiteSmodel_whole, blocks)

# fill in diagonal block
H_off[:,:,365] = H_on[:,:,1]
S_off[:,:,365] = S_on[:,:,1]
#hartree2ev*I(14)

# unit conversion
H_off ./= hartree2ev
S_off ./= hartree2ev

h5open( outfile, "w" ) do fid
    write(fid, "H", H_off)
    write(fid, "S", S_off)
end
