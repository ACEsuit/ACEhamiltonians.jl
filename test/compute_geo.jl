using ACEhamiltonians
using JSON, HDF5, JuLIP, LinearAlgebra

using ACEhamiltonians.Predict: predict_onsite_HS, predict_offsite_HS
using ACEhamiltonians.Dictionary: recover_at, recover_model, recover_error

onsite_model_ID = "../on_model_hybrid_1120_bcc1fcc1_correctmean/2653067664384673997"
offsite_model_ID = "../tuned_model/2606434113214136884"

const hartree2ev = 27.211386024367243

println("Loading onsite model...")
onsiteHmodelfile = onsite_model_ID * "_H.json"
onsiteHmodel_data = load_json(onsiteHmodelfile)
onsiteHmodel_whole = recover_model(onsiteHmodel_data)
onsiteHerror = recover_error(onsiteHmodel_data)
println("onsite H error ", onsiteHerror)

onsiteSmodelfile = onsite_model_ID * "_S.json"
onsiteSmodel_data = load_json(onsiteSmodelfile)
onsiteSmodel_whole = recover_model(onsiteSmodel_data)
onsiteSerror = recover_error(onsiteSmodel_data)
println("onsite S error ", onsiteSerror)

println("Loading offsite model...")
offsiteHmodelfile = offsite_model_ID * "_H.json"
offsiteHmodel_data = load_json(offsiteHmodelfile)
offsiteHmodel_whole = recover_model(offsiteHmodel_data)
offsiteHerror = recover_error(offsiteHmodel_data)
println("offsite H error ", offsiteHerror)

offsiteSmodelfile = offsite_model_ID * "_S.json"
offsiteSmodel_data = load_json(offsiteSmodelfile)
offsiteSmodel_whole = recover_model(load_json(offsiteSmodelfile))
offsiteSerror = recover_error(offsiteSmodel_data)
println("offsite S error ", offsiteSerror)

for i = 0:15
    if i <= 9
        infile = "geometry/geometry-00$(i).json"
        outfile = "out_00$(i)_2_9.h5"
    else
        infile = "geometry/geometry-0$(i).json"
        outfile = "out_0$(i)_2_9.h5"
    end
    println("Begin to deal with $(i)-th file")
    @show infile
    @show outfile

    at = recover_at( load_json(infile) )

    println("Predicting onsite Hamiltonian...")
    blocks = [365]
    H_on = predict_onsite_HS(at, onsiteHmodel_whole, blocks)
    S_on = predict_onsite_HS(at, onsiteSmodel_whole, blocks)

    println("Predicting offsite H and S...")
    blocks =  [ (365, i) for i = 1:729 ]
    H_off = predict_offsite_HS(at, offsiteHmodel_whole, blocks)
    S_off = predict_offsite_HS(at, offsiteSmodel_whole, blocks)

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
    println("Saving $(i)-th result successfully! ")
end