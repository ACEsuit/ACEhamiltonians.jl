module Predict

using JuLIP, LinearAlgebra, Statistics
using ACEhamiltonians.Structure
using ACEhamiltonians.DataProcess
using ACE: evaluate, PositionState, ACEConfig
using ACEhamiltonians.DataProcess: get_state
using ACEhamiltonians.Fitting: evaluateval_real, params2wmodels
## Predict Hamiltonian blocks and compute the error of blocks
@doc raw"""
# Input: TBModel, Exact data
# Output: Predicted Hamiltonian H_pre,
          RMSE = |H_pre - H_exact|/sqrt(#Observations),
          RMSRE = |H_pre - H_exact|/|H_exact|/#Observations,
          ...
"""

function model2valerr(model::Vector{TBModel}, data, symbol=1)
   L1,L2 = size(data[1][1][1])
   L1 = Int((L1 - 1)/2)
   L2 = Int((L2 - 1)/2)
   imax = 3 - L1
   jmax = 3 - L2
   Hexact = data[symbol]
   H_predict = [[zeros(Float64,2L1+1,2L2+1) for i=1:imax,j=1:jmax] for k = 1:length(data[1])]
   for idat = 1:length(Hexact)
      cfg = ACEConfig(data[3][idat])
      for i = 1:imax, j = 1:jmax
         submodel = model[ij2k(i,j,jmax)]
         Aval = evaluate(submodel.basis, cfg)
         A = evaluateval_real(Aval)
         H_predict[idat][i,j] = (A'*submodel.coeffs)' + submodel.mean
      end
   end
   DH = H_predict - Hexact
   rmse = [0.0 for i = 1:imax, j = 1:jmax]
   rmsre = [0.0 for i = 1:imax, j = 1:jmax]
   for i = 1:imax, j = 1:jmax
      rmse[i,j] = sqrt(sum([sum(DH[k][i,j].^2) for k = 1:length(DH)])/length(DH)/length(DH[1][1]))
      rmsre[i,j] = sqrt(sum([ sum((DH[k][i,j].^2)./((Hexact[k][i,j].^2).+1e-20)) for k = 1:length(DH) ] )/length(DH)/length(DH[1][1]))
   end

   return H_predict, rmse, rmsre#, model.mean
end

ij2k(i,j,jmax) = jmax*(i-1)+j


# TODO: Add a function that recover Data structure to data_whole and data_test!!!
# Simply like this:
# data_whole(dat::Data) = data_read(dat.file_number,dat.index,i)
function wmodels2err(MWH::TBModelWhole, MWS::TBModelWhole, dat_test::Data, data_whole = nothing)
   @assert(ison(MWH.data) == ison(dat_test))
   @assert(isoff(MWH.data) == isoff(dat_test))
   if ison(dat_test)
      modelset = ["ModelSS","ModelSP","ModelSD","ModelPP","ModelPD","ModelDD"]
   elseif isoff(dat_test)
      modelset = ["ModelSS","ModelSP","ModelSD","ModelPS","ModelPP","ModelPD","ModelDS","ModelDP","ModelDD"]
   else
      @error("Wrong type!")
   end
   if data_whole == nothing
      data_whole = data_read(MWH.data.file_name,MWH.data.index)
   end
   data_test  = data_read(dat_test.file_name,dat_test.index)
   RMSE_H_train_set = Vector{Matrix{Float64}}([])
   RMSE_S_train_set = Vector{Matrix{Float64}}([])
   RMSE_H_test_set = Vector{Matrix{Float64}}([])
   RMSE_S_test_set = Vector{Matrix{Float64}}([])
   for (i, field_name) in enumerate(modelset)
      Htrain, RMSE_H_train, RMSRE_H_train = model2valerr(getproperty(MWH,Symbol(field_name)),data_whole[i],1)
      Strain, RMSE_S_train, RMSRE_S_train = model2valerr(getproperty(MWS,Symbol(field_name)),data_whole[i],2)
      Htest, RMSE_H_test, RMSRE_H_test = model2valerr(getproperty(MWH,Symbol(field_name)),data_test[i],1)
      Stest, RMSE_S_test, RMSRE_S_test = model2valerr(getproperty(MWS,Symbol(field_name)),data_test[i],2)
      push!(RMSE_H_train_set, RMSE_H_train)
      push!(RMSE_S_train_set, RMSE_S_train)
      push!(RMSE_H_test_set, RMSE_H_test)
      push!(RMSE_S_test_set, RMSE_S_test)
   end

   return [RMSE_H_train_set,RMSE_H_test_set],[RMSE_S_train_set,RMSE_S_test_set]#,[H_E_M_train[1], H_E_M_test[1]],[S_E_M_train[1], S_E_M_test[1]]
end

@doc raw"""
`predict_main(dat_train::Data, dat_test::Data, par_H::Params, par_S::Params)`

# Input: Trainning data, testing data, parameters for H&S fitting
# Output: WholeModels for H&S and Averaged RMSE
"""
function predict_main(dat_train::Data, dat_test::Data, par_H::Params, par_S::Params)

   MWH, MWS, data_whole = params2wmodels(dat_train,par_H,par_S)
   return MWH, MWS, wmodels2err(MWH,MWS,dat_test,data_whole)

end


## Predict whole onsite Hamiltonian from model

function predict_onsite_HS(at::Atoms, model_whole::OnModelWhole, blocks)

   if isoff(model_whole)
      @warn("the given model might not refer to onsite model")
   end

   nlist = JuLIP.neighbourlist(at,20.0)

   modelSS = model_whole.ModelSS
   modelSP = model_whole.ModelSP
   modelSD = model_whole.ModelSD
   modelPP = model_whole.ModelPP
   modelPD = model_whole.ModelPD
   modelDD = model_whole.ModelDD

   H = zeros(ComplexF64,14,14,length(blocks))

   for (bi, i) in enumerate(blocks)

      Rs = PositionState.(Vector(JuLIP.Potentials.neigsz(nlist,at,i)[2]))
      cfg = ACEConfig(Rs)

      #  compute the hamiltonian
      #  -> H, S  as Norb x Norb x Nat tensor

      # TODO: Again, there must exist a more elegant way to do the same thing!!
      #       Thought: labeled the models (or maybe a dictionary for 0,1,2&spd?) and
      #       construct a function to do such thing.

      for j=1:3, k=1:3
         Aval = evaluate(modelSS[ij2k(j,k,3)].basis, cfg)
         A = evaluateval_real(Aval)
         H[j,k,bi] = (modelSS[ij2k(j,k,3)].coeffs' * A)[1] + modelSS[ij2k(j,k,3)].mean[1]
      end

      for j=1:2, k=1:3
         Aval = evaluate(modelSP[ij2k(j,k,3)].basis, cfg)
         A = evaluateval_real(Aval)
         H[3j+1:3(j+1),k,bi] = modelSP[ij2k(j,k,3)].coeffs' * A
      end

      for j=1:1, k=1:3 
         Aval = evaluate(modelSD[ij2k(j,k,3)].basis, cfg)
         A = evaluateval_real(Aval)
         H[j+9:j+13,k,bi] = modelSD[ij2k(j,k,3)].coeffs' * A
      end

      for j=1:2, k=1:2
         Aval = evaluate(modelPP[ij2k(j,k,2)].basis, cfg)
         A = evaluateval_real(Aval)
         H[3j+1:3(j+1),3k+1:3(k+1),bi] = modelPP[ij2k(j,k,2)].coeffs' * A + modelPP[ij2k(j,k,2)].mean
      end

      for j=1:1, k=1:2
         Aval = evaluate(modelPD[ij2k(j,k,2)].basis, cfg)
         A = evaluateval_real(Aval)
         H[j+9:j+13,3k+1:3(k+1),bi] = modelPD[ij2k(j,k,2)].coeffs' * A
      end

      for j=1:1, k=1:1
         Aval = evaluate(modelDD[ij2k(j,k,1)].basis, cfg)
         A = evaluateval_real(Aval)
         H[j+9:j+13,k+9:k+13,bi] = modelDD[ij2k(j,k,2)].coeffs' * A  + modelDD[ij2k(j,k,2)].mean
      end

      H[1:3,4:9,bi] = H[4:9,1:3,bi]'
      H[1:3,10:14,bi] = H[10:14,1:3,bi]'
      H[4:9,10:14,bi] = H[10:14,4:9,bi]'

   end
   return real(H)
end

predict_onsite_HS(at::Atoms, model_whole::TBModelWhole) = predict_onsite_HS(at, model_whole, 1:length(at))

## Predict whole offsite Hamiltonian from model

function predict_offsite_HS(at::Atoms, model_whole::OffModelWhole, blocks)

   if !isoff(model_whole)
      @warn("the given model might not refer to offsite model")
   end

   modelSS = model_whole.ModelSS
   modelSP = model_whole.ModelSP
   modelSD = model_whole.ModelSD
   modelPS = model_whole.ModelPS
   modelPP = model_whole.ModelPP
   modelPD = model_whole.ModelPD
   modelDS = model_whole.ModelDS
   modelDP = model_whole.ModelDP
   modelDD = model_whole.ModelDD

   # H = zeros(ComplexF64,14,14,maximum(maximum(blocks)))   
   H = zeros(ComplexF64,14,14,length(blocks)) 
   for (bi, idx) in enumerate(blocks)
      i, j = Tuple(idx)
      i == j && continue
      println("Predicting block $i $j")
      Rs = get_state(at,[(i,j)])
      cfg = ACEConfig(Rs[1])

      #  compute the hamiltonian
      #  -> H, S  as Norb x Norb x Nat × Nat tensor

      # TODO: Again, there must exist a more elegant way to do the same thing!!
      #       Thought: labeled the models (or maybe a dictionary for 0,1,2&spd?) and
      #       construct a function to do such thing.

      for k = 1:3, l=1:3
         Aval = evaluate(modelSS[ij2k(k,l,3)].basis, cfg)
         A = evaluateval_real(Aval)      
         H[k,l,bi] = (modelSS[ij2k(k,l,3)].coeffs' * A)[1] # + modelSS.mean[1]
      end

      for k = 1:2, l=1:3
         Aval = evaluate(modelSP[ij2k(k,l,3)].basis, cfg)
         A = evaluateval_real(Aval)
         H[3k+1:3(k+1),l,bi] = modelSP[ij2k(k,l,3)].coeffs' * A
      end
      
      for k = 1:3, l=1:2
         Aval = evaluate(modelPS[ij2k(k,l,2)].basis, cfg)
         A = evaluateval_real(Aval)
         H[k,3l+1:3(l+1),bi] = modelPS[ij2k(k,l,2)].coeffs' * A
      end
      
      for k = 1:1, l=1:3
         Aval = evaluate(modelSD[ij2k(k,l,3)].basis, cfg)
         A = evaluateval_real(Aval)
         H[k+9:k+13,l,bi] = modelSD[ij2k(k,l,3)].coeffs' * A
      end
      
      for k = 1:3, l=1:1
         Aval = evaluate(modelDS[ij2k(k,l,1)].basis, cfg)
         A = evaluateval_real(Aval)
         H[k,l+9:l+13,bi] = modelDS[ij2k(k,l,1)].coeffs' * A
      end

      for k = 1:2, l=1:2
         Aval = evaluate(modelPP[ij2k(k,l,2)].basis, cfg)
         A = evaluateval_real(Aval)
         H[3k+1:3(k+1),3l+1:3(l+1),bi] = modelPP[ij2k(k,l,2)].coeffs' * A # + modelPP.mean[k,l]
      end
      
      for k = 1:1, l=1:2
         Aval = evaluate(modelPD[ij2k(k,l,2)].basis, cfg)
         A = evaluateval_real(Aval)
         H[k+9:k+13,3l+1:3(l+1),bi] = modelPD[ij2k(k,l,2)].coeffs' * A
      end
      
      for k = 1:2, l=1:1
         Aval = evaluate(modelDP[ij2k(k,l,1)].basis, cfg)
         A = evaluateval_real(Aval)
         H[3k+1:3(k+1),l+9:l+13,bi] = modelDP[ij2k(k,l,1)].coeffs' * A
      end

      for k = 1:1, l=1:1
         Aval = evaluate(modelDD[ij2k(k,l,1)].basis, cfg)
         A = evaluateval_real(Aval)
         H[k+9:k+13,l+9:l+13,bi] = modelDD[ij2k(k,l,1)].coeffs' * A # + modelDD.mean[k,l]
      end

      # FIXME symmetrisation of i,j -> j,i removed: should be re-added in method below
   end
   return real(H)
end

function predict_offsite_HS(at::Atoms, model_whole::TBModelWhole) 
   natoms = length(at)
   blocks = CartesianIndices((1:natoms, 1:natoms)) # FIXME should be triangular
   H_blocks = predict_offsite_HS(at, model_whole, blocks)
   H_full = zeros(ComplexF64, 14, 14, natoms, natoms)
   for (bi, idx) in enumerate(blocks)
      i, j = Tuple(idx)
      i > j && continue
      H_full[:, :, i, j] = H_blocks[:, :, bi]
      H_full[:, :, j, i] = H_blocks[:, :, bi]'
   end
   return H_full
end

## dual symmetrisation
using ACE: State
function dual_state(st::State)
   if st.be==:bond
      return State(rr=-st.rr,rr0=-st.rr0,be=st.be)
   elseif st.be==:env
      return State(rr=st.rr-st.rr0,rr0=-st.rr0,be=st.be)
   end
end

function dual_state(vst::Vector)
   return dual_state.(vst)
end

function predict_offsite_HS_sym(at::Atoms, model_whole::OffModelWhole, blocks)

   if !isoff(model_whole)
      @warn("the given model might not refer to offsite model")
   end

   modelSS = model_whole.ModelSS
   modelSP = model_whole.ModelSP
   modelSD = model_whole.ModelSD
   modelPS = model_whole.ModelPS
   modelPP = model_whole.ModelPP
   modelPD = model_whole.ModelPD
   modelDS = model_whole.ModelDS
   modelDP = model_whole.ModelDP
   modelDD = model_whole.ModelDD

   # H = zeros(ComplexF64,14,14,maximum(maximum(blocks)))   
   H = zeros(ComplexF64,14,14,length(blocks))
   for (bi, idx) in enumerate(blocks)
      i, j = Tuple(idx)
      i == j && continue
      println("Predicting block $i $j")
      Rs = get_state(at,[(i,j)])
      cfg = ACEConfig(Rs[1])
      cfg_dual = ACEConfig(dual_state(Rs[1]))

      #  compute the hamiltonian
      #  -> H, S  as Norb x Norb x Nat × Nat tensor

      # TODO: Again, there must exist a more elegant way to do the same thing!!
      #       Thought: labeled the models (or maybe a dictionary for 0,1,2&spd?) and
      #       construct a function to do such thing.

      for k = 1:3, l=1:3
         Aval = evaluate(modelSS[ij2k(k,l,3)].basis, cfg)
         A = evaluateval_real(Aval)
         H[k,l,bi] += (modelSS[ij2k(k,l,3)].coeffs' * A)[1] # + modelSS.mean[1]
         Aval = evaluate(modelSS[ij2k(l,k,3)].basis, cfg_dual)
         A = evaluateval_real(Aval)
         H[k,l,bi] += ((modelSS[ij2k(l,k,3)].coeffs' * A)[1])'
      end

      for k = 1:2, l=1:3
         Aval = evaluate(modelSP[ij2k(k,l,3)].basis, cfg)
         A = evaluateval_real(Aval)
         H[3k+1:3(k+1),l,bi] += modelSP[ij2k(k,l,3)].coeffs' * A
         Aval = evaluate(modelPS[ij2k(l,k,2)].basis, cfg_dual)
         A = evaluateval_real(Aval)
         H[3k+1:3(k+1),l,bi] += (modelPS[ij2k(l,k,2)].coeffs' * A)'
      end
      
      for k = 1:3, l=1:2
         Aval = evaluate(modelPS[ij2k(k,l,2)].basis, cfg)
         A = evaluateval_real(Aval)
         H[k:k,3l+1:3(l+1),bi] += modelPS[ij2k(k,l,2)].coeffs' * A
         Aval = evaluate(modelSP[ij2k(l,k,3)].basis, cfg_dual)
         A = evaluateval_real(Aval)
         H[k:k,3l+1:3(l+1),bi] += (modelSP[ij2k(l,k,3)].coeffs' * A)'
      end
      
      for k = 1:1, l=1:3
         Aval = evaluate(modelSD[ij2k(k,l,3)].basis, cfg)
         A = evaluateval_real(Aval)
         H[k+9:k+13,l,bi] = modelSD[ij2k(k,l,3)].coeffs' * A
         Aval = evaluate(modelDS[ij2k(k,l,1)].basis, cfg_dual)
         A = evaluateval_real(Aval)
         H[k+9:k+13,l,bi] += (modelDS[ij2k(k,l,1)].coeffs' * A)'
      end
      
      for k = 1:3, l=1:1
         Aval = evaluate(modelDS[ij2k(k,l,1)].basis, cfg)
         A = evaluateval_real(Aval)
         H[k:k,l+9:l+13,bi] = modelDS[ij2k(k,l,1)].coeffs' * A
         Aval = evaluate(modelSD[ij2k(k,l,1)].basis, cfg_dual)
         A = evaluateval_real(Aval)
         H[k:k,l+9:l+13,bi] += (modelSD[ij2k(k,l,1)].coeffs' * A)'
      end

      for k = 1:2, l=1:2
         Aval = evaluate(modelPP[ij2k(k,l,2)].basis, cfg)
         A = evaluateval_real(Aval)
         H[3k+1:3(k+1),3l+1:3(l+1),bi] += modelPP[ij2k(k,l,2)].coeffs' * A # + modelPP.mean[k,l]
         Aval = evaluate(modelPP[ij2k(l,k,2)].basis, cfg_dual)
         A = evaluateval_real(Aval)
         H[3k+1:3(k+1),3l+1:3(l+1),bi] += (modelPP[ij2k(l,k,2)].coeffs' * A)' # + modelPP.mean[k,l]
      end
      
      for k = 1:1, l=1:2
         Aval = evaluate(modelPD[ij2k(k,l,2)].basis, cfg)
         A = evaluateval_real(Aval)
         H[k+9:k+13,3l+1:3(l+1),bi] = modelPD[ij2k(k,l,2)].coeffs' * A
         Aval = evaluate(modelDP[ij2k(k,l,2)].basis, cfg_dual)
         A = evaluateval_real(Aval)
         H[k+9:k+13,3l+1:3(l+1),bi] += (modelDP[ij2k(k,l,2)].coeffs' * A)'
      end
      
      for k = 1:2, l=1:1
         Aval = evaluate(modelDP[ij2k(k,l,1)].basis, cfg)
         A = evaluateval_real(Aval)
         H[3k+1:3(k+1),l+9:l+13,bi] = modelDP[ij2k(k,l,1)].coeffs' * A
         Aval = evaluate(modelPD[ij2k(k,l,1)].basis, cfg_dual)
         A = evaluateval_real(Aval)
         H[3k+1:3(k+1),l+9:l+13,bi] += (modelPD[ij2k(k,l,1)].coeffs' * A)'
      end

      for k = 1:1, l=1:1
         Aval = evaluate(modelDD[ij2k(k,l,1)].basis, cfg)
         A = evaluateval_real(Aval)
         H[k+9:k+13,l+9:l+13,bi] += modelDD[ij2k(k,l,1)].coeffs' * A # + modelDD.mean[k,l]
         Aval = evaluate(modelDD[ij2k(k,l,1)].basis, cfg_dual)
         A = evaluateval_real(Aval)
         H[k+9:k+13,l+9:l+13,bi] += (modelDD[ij2k(k,l,1)].coeffs' * A)' # + modelDD.mean[k,l]
      end

      # FIXME symmetrisation of i,j -> j,i removed: should be re-added in method below
   end
   return real(H)./2
end

function predict_offsite_HS_sym(at::Atoms, model_whole::TBModelWhole) 
   natoms = length(at)
   blocks = CartesianIndices((1:natoms, 1:natoms)) # FIXME should be triangular
   H_blocks = predict_offsite_HS_sym(at, model_whole, blocks)
   H_full = zeros(ComplexF64, 14, 14, natoms, natoms)
   for (bi, idx) in enumerate(blocks)
      i, j = Tuple(idx)
      i > j && continue
      H_full[:, :, i, j] = H_blocks[:, :, bi]
      H_full[:, :, j, i] = H_blocks[:, :, bi]'
   end
   return H_full
end

## getting rid of redundant models
function predict_offsite_HS_redundant(at::Atoms, model_whole::OffModelWhole, blocks)

   if !isoff(model_whole)
      @warn("the given model might not refer to offsite model")
   end

   modelSS = model_whole.ModelSS
   modelSP = model_whole.ModelSP
   modelSD = model_whole.ModelSD
   # modelPS = model_whole.ModelPS
   modelPP = model_whole.ModelPP
   modelPD = model_whole.ModelPD
   # modelDS = model_whole.ModelDS
   # modelDP = model_whole.ModelDP
   modelDD = model_whole.ModelDD

   # H = zeros(ComplexF64,14,14,maximum(maximum(blocks)))   
   H = zeros(ComplexF64,14,14,length(blocks))
   for (bi, idx) in enumerate(blocks)
      i, j = Tuple(idx)
      i == j && continue
      println("Predicting block $i $j")
      Rs = get_state(at,[(i,j)])
      cfg = ACEConfig(Rs[1])
      cfg_dual = ACEConfig(dual_state(Rs[1]))

      #  compute the hamiltonian
      #  -> H, S  as Norb x Norb x Nat × Nat tensor

      # TODO: Again, there must exist a more elegant way to do the same thing!!
      #       Thought: labeled the models (or maybe a dictionary for 0,1,2&spd?) and
      #       construct a function to do such thing.

      for k = 1:3, l=1:3
         Aval = evaluate(modelSS[ij2k(k,l,3)].basis, cfg)
         A = evaluateval_real(Aval)
         # Aval_dual = evaluate(modelSS[ij2k(k,l,3)].basis, cfg_dual)
         # A = ( A + evaluateval_real(Aval_dual) )/2
         H[k,l,bi] = (modelSS[ij2k(k,l,3)].coeffs' * A)[1] # + modelSS.mean[1]
      end

      for k = 1:2, l=1:3
         Aval = evaluate(modelSP[ij2k(k,l,3)].basis, cfg)
         A = evaluateval_real(Aval)
         H[3k+1:3(k+1),l,bi] = modelSP[ij2k(k,l,3)].coeffs' * A
      end
      
      for k = 1:3, l=1:2
         # Aval = evaluate(modelPS[ij2k(k,l,2)].basis, cfg)
         # A = evaluateval_real(Aval)
         # H[k,3l+1:3(l+1),bi] = modelPS[ij2k(k,l,2)].coeffs' * A
         Aval = evaluate(modelSP[ij2k(l,k,3)].basis, cfg_dual)
         A = evaluateval_real(Aval)
         H[k,3l+1:3(l+1),bi] = (modelSP[ij2k(l,k,3)].coeffs' * A)'
      end
      
      for k = 1:1, l=1:3
         Aval = evaluate(modelSD[ij2k(k,l,3)].basis, cfg)
         A = evaluateval_real(Aval)
         H[k+9:k+13,l,bi] = modelSD[ij2k(k,l,3)].coeffs' * A
      end
      
      for k = 1:3, l=1:1
         # Aval = evaluate(modelDS[ij2k(k,l,1)].basis, cfg)
         # A = evaluateval_real(Aval)
         # H[k,l+9:l+13,bi] = modelDS[ij2k(k,l,1)].coeffs' * A
         Aval = evaluate(modelSD[ij2k(k,l,1)].basis, cfg_dual)
         A = evaluateval_real(Aval)
         H[k,l+9:l+13,bi] = (modelSD[ij2k(k,l,1)].coeffs' * A)'
      end

      for k = 1:2, l=1:2
         Aval = evaluate(modelPP[ij2k(k,l,2)].basis, cfg)
         A = evaluateval_real(Aval)
         # Aval_dual = evaluate(modelPP[ij2k(k,l,2)].basis, cfg_dual)
         # A = ( A + evaluateval_real(Aval_dual) )/2
         H[3k+1:3(k+1),3l+1:3(l+1),bi] = modelPP[ij2k(k,l,2)].coeffs' * A # + modelPP.mean[k,l]
      end
      
      for k = 1:1, l=1:2
         Aval = evaluate(modelPD[ij2k(k,l,2)].basis, cfg)
         A = evaluateval_real(Aval)
         H[k+9:k+13,3l+1:3(l+1),bi] = modelPD[ij2k(k,l,2)].coeffs' * A
      end
      
      for k = 1:2, l=1:1
         # Aval = evaluate(modelDP[ij2k(k,l,1)].basis, cfg)
         # A = evaluateval_real(Aval)
         # H[3k+1:3(k+1),l+9:l+13,bi] = modelDP[ij2k(k,l,1)].coeffs' * A
         Aval = evaluate(modelPD[ij2k(k,l,1)].basis, cfg_dual)
         A = evaluateval_real(Aval)
         H[3k+1:3(k+1),l+9:l+13,bi] = (modelPD[ij2k(k,l,1)].coeffs' * A)'
      end

      for k = 1:1, l=1:1
         Aval = evaluate(modelDD[ij2k(k,l,1)].basis, cfg)
         A = evaluateval_real(Aval)
         # Aval_dual = evaluate(modelDD[ij2k(k,l,1)].basis, cfg_dual)
         # A = ( A + evaluateval_real(Aval_dual) )/2
         H[k+9:k+13,l+9:l+13,bi] = modelDD[ij2k(k,l,1)].coeffs' * A # + modelDD.mean[k,l]
      end

      # FIXME symmetrisation of i,j -> j,i removed: should be re-added in method below
   end
   return real(H)
end

function predict_offsite_HS_redundant(at::Atoms, model_whole::TBModelWhole) 
   natoms = length(at)
   blocks = CartesianIndices((1:natoms, 1:natoms)) # FIXME should be triangular
   H_blocks = predict_offsite_HS_redundant(at, model_whole, blocks)
   H_full = zeros(ComplexF64, 14, 14, natoms, natoms)
   for (bi, idx) in enumerate(blocks)
      i, j = Tuple(idx)
      i > j && continue
      H_full[:, :, i, j] = H_blocks[:, :, bi]
      H_full[:, :, j, i] = H_blocks[:, :, bi]'
   end
   return H_full
end

## Test on getting rid of redundant models + symmetrised homo-orbital
function predict_offsite_HS_redundant_sym(at::Atoms, model_whole::OffModelWhole, blocks)

   if !isoff(model_whole)
      @warn("the given model might not refer to offsite model")
   end

   modelSS = model_whole.ModelSS
   modelSP = model_whole.ModelSP
   modelSD = model_whole.ModelSD
   # modelPS = model_whole.ModelPS
   modelPP = model_whole.ModelPP
   modelPD = model_whole.ModelPD
   # modelDS = model_whole.ModelDS
   # modelDP = model_whole.ModelDP
   modelDD = model_whole.ModelDD

   H = zeros(ComplexF64,14,14,maximum(maximum(blocks)))   
   for (bi, idx) in enumerate(blocks)
      i, j = Tuple(idx)
      i == j && continue
      println("Predicting block $i $j")
      Rs = get_state(at,[(i,j)])
      cfg = ACEConfig(Rs[1])
      cfg_dual = ACEConfig(dual_state(Rs[1]))

      #  compute the hamiltonian
      #  -> H, S  as Norb x Norb x Nat × Nat tensor

      # TODO: Again, there must exist a more elegant way to do the same thing!!
      #       Thought: labeled the models (or maybe a dictionary for 0,1,2&spd?) and
      #       construct a function to do such thing.

      for k = 1:3, l=1:3
         Aval = evaluate(modelSS[ij2k(k,l,3)].basis, cfg)
         A = evaluateval_real(Aval)
         H[k,l,bi] += (modelSS[ij2k(k,l,3)].coeffs' * A)[1] # + modelSS.mean[1]
         Aval = evaluate(modelSS[ij2k(l,k,3)].basis, cfg_dual)
         A = evaluateval_real(Aval)
         H[k,l,bi] += ((modelSS[ij2k(l,k,3)].coeffs' * A)[1])'
         H[k,l,bi] = H[k,l,bi]./2
      end

      for k = 1:2, l=1:3
         Aval = evaluate(modelSP[ij2k(k,l,3)].basis, cfg)
         A = evaluateval_real(Aval)
         H[3k+1:3(k+1),l,bi] = modelSP[ij2k(k,l,3)].coeffs' * A
      end
      
      for k = 1:3, l=1:2
         # Aval = evaluate(modelPS[ij2k(k,l,2)].basis, cfg)
         # A = evaluateval_real(Aval)
         # H[k,3l+1:3(l+1),bi] = modelPS[ij2k(k,l,2)].coeffs' * A
         Aval = evaluate(modelSP[ij2k(l,k,3)].basis, cfg_dual)
         A = evaluateval_real(Aval)
         H[k,3l+1:3(l+1),bi] = (modelSP[ij2k(l,k,3)].coeffs' * A)'
      end
      
      for k = 1:1, l=1:3
         Aval = evaluate(modelSD[ij2k(k,l,3)].basis, cfg)
         A = evaluateval_real(Aval)
         H[k+9:k+13,l,bi] = modelSD[ij2k(k,l,3)].coeffs' * A
      end
      
      for k = 1:3, l=1:1
         # Aval = evaluate(modelDS[ij2k(k,l,1)].basis, cfg)
         # A = evaluateval_real(Aval)
         # H[k,l+9:l+13,bi] = modelDS[ij2k(k,l,1)].coeffs' * A
         Aval = evaluate(modelSD[ij2k(k,l,1)].basis, cfg_dual)
         A = evaluateval_real(Aval)
         H[k,l+9:l+13,bi] = (modelSD[ij2k(k,l,1)].coeffs' * A)'
      end

      for k = 1:2, l=1:2
         Aval = evaluate(modelPP[ij2k(k,l,2)].basis, cfg)
         A = evaluateval_real(Aval)
         H[3k+1:3(k+1),3l+1:3(l+1),bi] += modelPP[ij2k(k,l,2)].coeffs' * A # + modelPP.mean[k,l]
         Aval = evaluate(modelPP[ij2k(l,k,2)].basis, cfg_dual)
         A = evaluateval_real(Aval)
         H[3k+1:3(k+1),3l+1:3(l+1),bi] += (modelPP[ij2k(l,k,2)].coeffs' * A)' # + modelPP.mean[k,l]
         H[3k+1:3(k+1),3l+1:3(l+1),bi] = H[3k+1:3(k+1),3l+1:3(l+1),bi]./2
      end
      
      for k = 1:1, l=1:2
         Aval = evaluate(modelPD[ij2k(k,l,2)].basis, cfg)
         A = evaluateval_real(Aval)
         H[k+9:k+13,3l+1:3(l+1),bi] = modelPD[ij2k(k,l,2)].coeffs' * A
      end
      
      for k = 1:2, l=1:1
         # Aval = evaluate(modelDP[ij2k(k,l,1)].basis, cfg)
         # A = evaluateval_real(Aval)
         # H[3k+1:3(k+1),l+9:l+13,bi] = modelDP[ij2k(k,l,1)].coeffs' * A
         Aval = evaluate(modelPD[ij2k(k,l,1)].basis, cfg_dual)
         A = evaluateval_real(Aval)
         H[3k+1:3(k+1),l+9:l+13,bi] = (modelPD[ij2k(k,l,1)].coeffs' * A)'
      end

      for k = 1:1, l=1:1
         Aval = evaluate(modelDD[ij2k(k,l,1)].basis, cfg)
         A = evaluateval_real(Aval)
         H[k+9:k+13,l+9:l+13,bi] += modelDD[ij2k(k,l,1)].coeffs' * A # + modelDD.mean[k,l]
         Aval = evaluate(modelDD[ij2k(k,l,1)].basis, cfg_dual)
         A = evaluateval_real(Aval)
         H[k+9:k+13,l+9:l+13,bi] += (modelDD[ij2k(k,l,1)].coeffs' * A)' # + modelDD.mean[k,l]
         H[k+9:k+13,l+9:l+13,bi] = H[k+9:k+13,l+9:l+13,bi]./2
      end

      # FIXME symmetrisation of i,j -> j,i removed: should be re-added in method below
   end
   return real(H)
end

function predict_offsite_HS_redundant_sym(at::Atoms, model_whole::TBModelWhole) 
   natoms = length(at)
   blocks = CartesianIndices((1:natoms, 1:natoms)) # FIXME should be triangular
   H_blocks = predict_offsite_HS_redundant(at, model_whole, blocks)
   H_full = zeros(ComplexF64, 14, 14, natoms, natoms)
   for (bi, idx) in enumerate(blocks)
      i, j = Tuple(idx)
      i > j && continue
      H_full[:, :, i, j] = H_blocks[:, :, bi]
      H_full[:, :, j, i] = H_blocks[:, :, bi]'
   end
   return H_full
end

end

