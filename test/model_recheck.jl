using ACEhamiltonians, ACE
using JSON, HDF5, JuLIP, LinearAlgebra

using ACEhamiltonians: Data, Params, write_modeljson,
              write_atomjson, wm2fn, fname2index
using ACEhamiltonians.Predict: predict_main, predict_offsite_HS
using ACEhamiltonians.Dictionary: recover_at,recover_model,recover_error

## Model read

const hartree2ev = 27.2114

Hmodelfile = "ACEhamiltonians-data/offsite_models_ord1/10204186688118368371_H.json"
# Hmodelfile = "ACEhamiltonians-data/offsite_models_ord1/14750835312950641338_H.json"
# Hmodelfile = "ACEhamiltonians-data/optimized_model/offsite/4570230078043807257_H.json"

Hmodel_whole = read_dict(load_json(Hmodelfile))
# NOTE: A wrong read here - I should have removed this function!
# Hmodel_whole1 = recover_model(load_json(Hmodelfile))

model_dd = Hmodel_whole.ModelDD[1]
# basis and coeffs - we are checking a offsite model so the mean field is just 0
basis = model_dd.basis
C = model_dd.coeffs

# read reference data, including Hamiltonian, overlap and positions
fname = "ACEhamiltonians-data/reference_data/FCC/FCC-supercell-000.h5"
# fname = "ACEhamiltonians-data/training_data/FCC/SK-supercell-001.h5"
h5 = h5open(fname)
dict = HDF5.read(h5["aitb"])
H = dict["H"]
at = get_atoms(fname)[2]
i,j = 54,55
st = ACEhamiltonians.DataProcess.get_state(at,[(i,j)])[1]
cfg = ACEConfig(st)

A = ACE.evaluate(basis,cfg)
AA = ACEhamiltonians.Fitting.evaluateval_real(A)

Hpred = sum([AA[i] * C[i] for i = 1:length(C)])
Href = H[14(i-1)+10:14(i-1)+14,14(j-1)+10:14(j-1)+14].*hartree2ev
E = Hpred - Href
rmse = norm(E)/5

rmse_total = 0
k = 0
for i = 1:20:729
    for j = 2:20:729
        st = ACEhamiltonians.DataProcess.get_state(at,[(i,j)])[1]
        cfg = ACEConfig(st)

        A = ACE.evaluate(basis,cfg)
        AA = ACEhamiltonians.Fitting.evaluateval_real(A)

        Hpred = sum([AA[t] * C[t] for t = 1:length(C)])
        Href = H[14(i-1)+10:14(i-1)+14,14(j-1)+10:14(j-1)+14].*hartree2ev
        E = Hpred - Href
        rmse_total += norm(E)^2
        k += 1
        @show k
    end
end
rmse_total = sqrt(rmse_total/25/k)
@show log10(rmse_total)

## It returns ~-1.73 for (1,8) and ~-2.43 for (1,14), which is consistent to what we had in the paper

## Onsite check

Hmodelfile = "ACEhamiltonians-data/onsite_models_ord2/1646489440533135164_H.json"
Hmodel_whole = read_dict(load_json(Hmodelfile));

model_dd = Hmodel_whole.ModelDD[1]
# basis, coeffs and mean
basis = model_dd.basis
C = model_dd.coeffs
Mean = model_dd.mean

# read reference data, including Hamiltonian, overlap and positions
fname = "ACEhamiltonians-data/reference_data/FCC/FCC-supercell-000.h5"
# fname = "ACEhamiltonians-data/training_data/FCC/SK-supercell-001.h5"
h5 = h5open(fname)
dict = HDF5.read(h5["aitb"])
H = dict["H"]
at = get_atoms(fname)[2]
i = 1
st = ACE.PositionState.(Vector(JuLIP.Potentials.neigsz(JuLIP.neighbourlist(at,20.0),at,i)[2]))
cfg = ACEConfig(st)

A = ACE.evaluate(basis,cfg)
AA = ACEhamiltonians.Fitting.evaluateval_real(A)

Hpred = (sum([AA[i] * C[i] for i = 1:length(C)])+Mean)
Href = H[14(i-1)+10:14(i-1)+14,14(i-1)+10:14(i-1)+14].*hartree2ev
E = Hpred - Href
rmse = norm(E)/5

rmse_total = 0
k = 0
for i = 1:5:729
    st = ACEhamiltonians.DataProcess.get_state(at,[(i,j)])[1]
    cfg = ACEConfig(st)

    A = ACE.evaluate(basis,cfg)
    AA = ACEhamiltonians.Fitting.evaluateval_real(A)

    Hpred = (sum([AA[i] * C[i] for i = 1:length(C)])+Mean)
    Href = H[14(i-1)+10:14(i-1)+14,14(i-1)+10:14(i-1)+14].*hartree2ev
    E = Hpred - Href
    rmse_total += norm(E)^2
    k += 1
    @show k
end

rmse_total = sqrt(rmse_total/25/k)
@show log10(rmse_total)