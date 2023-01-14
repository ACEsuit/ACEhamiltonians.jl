using ACEhamiltonians, ACE
using JSON, HDF5, JuLIP, LinearAlgebra
using ACEhamiltonians: Data, Params, write_modeljson,
              write_atomjson, wm2fn, fname2index
using ACEhamiltonians.Predict: predict_main, predict_offsite_HS
using ACEhamiltonians.Dictionary: recover_at,recover_model,recover_error

const hartree2ev = 27.2114

## Data read
fname = "ACEhamiltonians-data/training_data/FCC/SK-supercell-001.h5"
h5 = h5open(fname)
dict = HDF5.read(h5["aitb"])
H = dict["H"]
at = get_atoms(fname)[2]

## Specify a test set
testset = []
k = 0
n_test = 200 # in our paper we have 2000 test data but it makes no huge difference so I choose 200 here for a quick illustration
while k<n_test
    i = rand(1:728)
    j = rand(i+1:729)
    testset = setdiff(push!(testset,(i,j)))
    k = length(testset)
end

## Then index the models
Dict_id = Dict("6" =>"7014526518680934587",
               "7" =>"8594416159488562244",
               "8" =>"10204186688118368371",
               "9" =>"13078304848585360574",
               "10"=>"14750835312950641338",
               "11"=>"9883802224093245794",
               "12"=>"3907899412408606585",
               "13"=>"201683837542179657",
               "14"=>"277744202775070779")
degset = 6:14
RMSE_SS = zeros(length(degset))
RMSE_SP = zeros(length(degset))
RMSE_PS = zeros(length(degset))
RMSE_PP = zeros(length(degset))

for deg in degset
    # Read whole model
    Hmodelfile = "ACEhamiltonians-data/offsite_models_ord1/$(Dict_id["$deg"])_H.json"
    Hmodel_whole = read_dict(load_json(Hmodelfile))
    # NOTE: Below is a wrong read - I should have removed this function!
    # Hmodel_whole1 = recover_model(load_json(Hmodelfile))

    # SS blocks
    rmse_total = zeros(length(Hmodel_whole.ModelSS))
    pos = [(1,1),(1,2),(1,3),(2,1),(2,2),(2,3),(3,1),(3,2),(3,3)]
    for nn = 1:length(Hmodel_whole.ModelSS)
        model = Hmodel_whole.ModelSS[nn]
        # basis and coeffs - we are checking a offsite model so the mean field is just 0
        basis = model.basis
        C = model.coeffs

        for (i,j) in testset
            st = ACEhamiltonians.DataProcess.get_state(at,[(i,j)])[1]
            cfg = ACEConfig(st)

            A = ACE.evaluate(basis,cfg)
            AA = ACEhamiltonians.Fitting.evaluateval_real(A)

            Hpred = sum([AA[t] * C[t] for t = 1:length(C)])
            Href = H[14(i-1)+pos[nn][1]:14(i-1)+pos[nn][1],14(j-1)+pos[nn][2]:14(j-1)+pos[nn][2]]# .*hartree2ev
            E = Hpred - Href
            rmse_total[nn] += norm(E)^2
        end
        rmse_total[nn] = sqrt(rmse_total[nn]/k)
    end
    # To get the correct RMSE
    # we should have used the following line in fact 
    # but they differed just slightly and do not have difference in terms of trend
    # @show sum(rmse_total.^2/4) |>sqrt |> log10
    RMSE_SS[deg-5] = mean(rmse_total)

    # PS blocks
    rmse_total = zeros(length(Hmodel_whole.ModelPS))
    pos = [(1,4),(1,7),(2,4),(2,7),(3,4),(3,7)]
    for nn = 1:length(Hmodel_whole.ModelPS)
        model = Hmodel_whole.ModelPS[nn]
        # basis and coeffs - we are checking a offsite model so the mean field is just 0
        basis = model.basis
        C = model.coeffs

        for (i,j) in testset
            st = ACEhamiltonians.DataProcess.get_state(at,[(i,j)])[1]
            cfg = ACEConfig(st)

            A = ACE.evaluate(basis,cfg)
            AA = ACEhamiltonians.Fitting.evaluateval_real(A)

            Hpred = sum([AA[t] * C[t] for t = 1:length(C)])
            Href = H[14(i-1)+pos[nn][1]:14(i-1)+pos[nn][1],14(j-1)+pos[nn][2]:14(j-1)+pos[nn][2]+2]# .*hartree2ev
            E = Hpred - Href
            rmse_total[nn] += norm(E)^2
        end
        rmse_total[nn] = sqrt(rmse_total[nn]/3/k)
    end
    @show deg, log10.(mean(rmse_total))
    RMSE_PS[deg-5] = mean(rmse_total)

    # SP blocks
    rmse_total = zeros(length(Hmodel_whole.ModelSP))
    pos = [(1,4),(2,4),(3,4),(1,7),(2,7),(3,7)]
    for nn = 1:length(Hmodel_whole.ModelSP)
        model = Hmodel_whole.ModelSP[nn]
        # basis and coeffs - we are checking a offsite model so the mean field is just 0
        basis = model.basis
        C = model.coeffs

        for (i,j) in testset
            st = ACEhamiltonians.DataProcess.get_state(at,[(i,j)])[1]
            cfg = ACEConfig(st)

            A = ACE.evaluate(basis,cfg)
            AA = ACEhamiltonians.Fitting.evaluateval_real(A)

            Hpred = sum([AA[t] * C[t] for t = 1:length(C)])
            Href = H[14(i-1)+pos[nn][2]:14(i-1)+pos[nn][2]+2,14(j-1)+pos[nn][1]:14(j-1)+pos[nn][1]]# .*hartree2ev
            E = Hpred - Href
            rmse_total[nn] += norm(E)^2
        end
        rmse_total[nn] = sqrt(rmse_total[nn]/3/k)
    end
    @show deg, log10.(mean(rmse_total))
    RMSE_SP[deg-5] = mean(rmse_total)

    # PP blocks
    rmse_total = zeros(length(Hmodel_whole.ModelPP))
    pos = [(4,4),(4,7),(7,4),(7,7)]
    for nn = 1:length(Hmodel_whole.ModelPP)
        model = Hmodel_whole.ModelPP[nn];
        # basis and coeffs - we are checking a offsite model so the mean field is just 0
        basis = model.basis;
        C = model.coeffs;

        for (i,j) in testset
            st = ACEhamiltonians.DataProcess.get_state(at,[(i,j)])[1]
            cfg = ACEConfig(st)

            A = ACE.evaluate(basis,cfg)
            AA = ACEhamiltonians.Fitting.evaluateval_real(A)

            Hpred = sum([AA[t] * C[t] for t = 1:length(C)])
            Href = H[14(i-1)+pos[nn][1]:14(i-1)+pos[nn][1]+2,14(j-1)+pos[nn][2]:14(j-1)+pos[nn][2]+2]# .*hartree2ev
            E = Hpred - Href
            rmse_total[nn] += norm(E)^2
        end
        rmse_total[nn] = sqrt(rmse_total[nn]/9/k)
    end
    @show deg, log10.(mean(rmse_total))
    # @show sum(rmse_total.^2/4) |>sqrt |> log10
    RMSE_PP[deg-5] = mean(rmse_total)
end

## Plot
plt = plot(degset,log10.(RMSE_SS),label="ss",ls=:dash)
plot!(degset,log10.(RMSE_SP),label="sp",color=2,shape=:diamond,ls=:dash)
plot!(degset,log10.(RMSE_PS),label="ps",color=2,shape=:rect,ls=:dash)
plot!(degset,log10.(RMSE_PP),label="pp",color=4,ls=:dash,xlabel="Maximum degree",ylabel="log10(RMSE) / ev",title="Order 1. Offsite H RMSE",size = (600,900))
ylims!(-3.7, -1)
plt


## PS1: I am assuming that nothing beyond the PP blocks is of interest at the current stage
## PS2: Similar codes can be written for onsite check and/or order 2 offsite check
