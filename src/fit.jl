module Fitting

using LinearAlgebra, StaticArrays, Statistics, LowRankApprox, IterativeSolvers, Distributed, DistributedArrays
using ACE: evaluate, evaluate_d, PositionState, ACEConfig
using ACEhamiltonians.Structure, ACEhamiltonians.DataProcess
using ACE: PIBasis, get_spec
import ACE.scaling

export params2wmodels

## Assemble A, Y; read blocks type by data directly;
#  read all Y at one time (e.g., 9 s-s blocks, 6 s-p blocks etc)

function evaluateval_real(Aval)
   L1,L2 = size(Aval[1])
   L1 = Int((L1-1)/2)
   L2 = Int((L2-1)/2)

   # allocate Aval_real
   Aval_real = [zeros(ComplexF64,2L1+1,2L2+1) for i = 1:length(Aval)]
   # reconstruct real A
   # TODO: I believe that there must exist a matrix form... But I will keep it for now...
   for i=1:length(Aval)
      A = Aval[i].val
      for k=1:2L1+1, j=1:2L2+1
         if k < L1+1 && j < L2+1
            Aval_real[i][k,j] = 1/2 * ( A[k,j] - (-1)^(k-L1-1)*A[2+2L1-k,j]
                                         -(-1)^(j-L2-1)*A[k,2+2L2-j]
                                         +(-1)^(k+j-L1-L2)*A[2+2L1-k,2+2L2-j] )
         elseif k < L1+1 && j == L2+1
            Aval_real[i][k,j] = im/sqrt(2) * ( A[k,j] - (-1)^(k-L1-1)*A[2+2L1-k,j] )
         elseif k < L1+1 && j > L2+1
            Aval_real[i][k,j] = im/2 * ( A[k,j] - (-1)^(k-L1-1)*A[2+2L1-k,j]
                                          +(-1)^(j-L2-1)*A[k,2+2L2-j]
                                          -(-1)^(k+j-L1-L2)*A[2+2L1-k,2+2L2-j] )
         elseif k == L1+1 && j < L2+1
            Aval_real[i][k,j] = im/sqrt(2) * ( - A[k,j] + (-1)^(j-L2-1)*A[k,2+2L2-j] )
         elseif k == L1+1 && j == L2+1
            Aval_real[i][k,j] = A[k,j]
         elseif k == L1+1 && j > L2+1
            Aval_real[i][k,j] = 1/sqrt(2) * ( A[k,j] + (-1)^(j-L2-1)*A[k,2+2L2-j] )
         elseif k > L1+1 && j < L2+1
            Aval_real[i][k,j] = im/2 * ( - A[k,j] - (-1)^(k-L1-1)*A[2+2L1-k,j]
                                          +(-1)^(j-L2-1)*A[k,2+2L2-j]
                                          +(-1)^(k+j-L1-L2)*A[2+2L1-k,2+2L2-j] )
         elseif k > L1+1 && j == L2+1
            Aval_real[i][k,j] = 1/sqrt(2) * ( A[k,j] + (-1)^(k-L1-1)*A[2+2L1-k,j] )
         elseif k > L1+1 && j > L2+1
            Aval_real[i][k,j] = 1/2 * ( A[k,j] + (-1)^(k-L1-1)*A[2+2L1-k,j]
                                          +(-1)^(j-L2-1)*A[k,2+2L2-j]
                                          +(-1)^(k+j-L1-L2)*A[2+2L1-k,2+2L2-j] )
         end
      end
   end
   #return Aval_real
   if norm(Aval_real - real(Aval_real))<1e-12
      return real(Aval_real)
   else
      error("norm = $(norm(Aval_real - real(Aval_real))), please recheck...")
   end
end

function assemble_ls(bos, Hsubs, Rs, i, j, flag = "on"; type_mean = 1)
   L1,L2 = size(Hsubs[1][1])
   L1 = Int((L1 - 1)/2)
   L2 = Int((L2 - 1)/2)
   Y = [ zeros(2L1+1,2L2+1) for k=1:length(Hsubs) ]
   A = fill( zeros(2L1+1,2L2+1), length(Hsubs), size(bos.basis.A2Bmap)[1] )
   for idat = 1:length(Hsubs)
      Y[idat] = SMatrix{2L1+1,2L2+1,Float64}(Hsubs[idat][i,j])
      cfg = ACEConfig(Rs[idat])
      Aval = evaluate(bos.basis, cfg)
      A[idat,:] = evaluateval_real(Aval)
   end
   if L1 == L2
      if flag == "on"
         if type_mean == 0
            Mean = zeros(2L1+1,2L2+1)
         else
            Mean = mean(diag(mean(Y)))*I(2L1+1)
         end
      elseif flag == "off"
         Mean = zeros(2L1+1,2L2+1)
      else 
         error("Unexpected flag!")
      end
   else
      Mean = zeros(2L1+1,2L2+1)
   end
   Y = Y .- Ref(Mean)
   return A, Y, Mean
end

function basis2tikmat(bos)
   Γ = scaling(bos.basis,2)
   return Diagonal(Γ)
end

# reshape A and Y
function preprocessA(A)
   # S1: number of sites; S2: number of basis, SS1: 2L1+1, SS2: 2L2+1
   S1,S2 = size(A)
   SS1,SS2 = size(A[1])
   A_temp = zeros(ComplexF64,S1*SS1*SS2,S2)
   for i = 1:S1
      for j = 1:S2
         A_temp[SS1*SS2*(i-1)+1:SS1*SS2*i,j] = reshape(A[i,j],SS1*SS2,1)
      end
   end
   return real(A_temp)
end

function preprocessY(Y)
   Len = length(Y)
   SS1,SS2 = size(Y[1])
   Y_temp = zeros(ComplexF64,Len*SS1*SS2)
   for i = 1:Len
      Y_temp[SS1*SS2*(i-1)+1:SS1*SS2*i] = reshape(Y[i],SS1*SS2,1)
   end
   return real(Y_temp)
end

function solve_ls(A, Y, λ, Γ, Solver = "LSQR")

   

   A = preprocessA(A)
   Y = preprocessY(Y)
   
   # Due to the sampling, this part can be omitted at some point?
   # function remove_zero_rows(A,Y)
   #    @assert size(A)[1] == size(Y)[1]
   #    rows_num = size(A)[1]
   #    norm_A = [norm(A[i,:]) for i = 1:rows_num]
   #    norm_Y = [norm(Y[i]) for i = 1:rows_num]
   #    order = sortperm(norm_A)
   #    A = A[order,:]
   #    Y = Y[order]
   # 
   #    set = []
   #    idx = 1
   #    while norm_A[order][idx]<1e-7
   #       if norm_Y[order][idx]<1e-6
   #          append!(set,idx)
   #       end
   #       idx += 1
   #       if idx > rows_num
   #          break
   #       end
   #    end
   #    set = filter(x -> x∉set, Vector(1:rows_num))
   # 
   #    return A[set,:], Y[set]
   # end
   # 
   # A, Y = remove_zero_rows(A, Y)
   
   num = size(A)[2] # number of basis
   A = [A; λ*Γ]
   Y = [Y; zeros(num)]
   if Solver == "QR"
      return real(qr(A) \ Y)
   elseif Solver == "LSQR"
      return real(IterativeSolvers.lsqr(distribute(A), distribute(Y); atol = 1e-6, btol = 1e-6))
   elseif Solver == "NaiveSolver"
      return real((A'*A) \ (A'*Y))
   end

end

function blocks2model(Hsubs, Rs, rcut, maxdeg, ord, λ, regtype=1, solver="LSQR", flag = "on", λ_n=.5, λ_l=.5)
   L1,L2 = size(Hsubs[1][1])
   L1 = Int((L1 - 1)/2)
   L2 = Int((L2 - 1)/2)
   imax = 3 - L1
   jmax = 3 - L2
   submodel = Vector{TBModel}([])
   
   # TODO: Maybe we could consider the symmetry to reduce cost?
   for i = 1:imax, j = 1:jmax
      if flag == "on"
         bos = OnsiteBasis(rcut[i,j], maxdeg[i,j], ord[i,j], L1, L2)   
      elseif flag == "off"
         bos = OffsiteBasis(rcut[i,j], maxdeg[i,j], ord[i,j], L1, L2, λ_n, λ_l)
      end
      #C = zeros(Float64,size(bos.basis.A2Bmap)[1])
      if regtype == 1
         Γ = I
      elseif regtype == 2
         Γ = basis2tikmat(bos)
      end
      A, Y, Mean = assemble_ls(bos,Hsubs,Rs,i,j,flag)
      C = solve_ls(A,Y,λ[i,j],Γ,solver)
      push!(submodel, TBModel(bos.basis,C,Mean))      
   end
   return submodel
end

function data2model(data, rcut::Matrix{Float64}, maxdeg::Matrix{Int64}, ord::Matrix{Int64}, λ::Matrix{Float64}, regtype=1, solver="LSQR", flag = "on")
   Hsubs = data[1]
   Ssubs = data[2]
   Rs = data[3]
   return blocks2model(Hsubs,Rs,rcut,maxdeg,ord,λ,regtype,solver,flag), blocks2model(Ssubs,Rs,rcut,maxdeg,ord,λ,regtype,solver,flag,0,0)
end

function data2model(data, rcut::Vector{Matrix{Float64}}, maxdeg::Vector{Matrix{Int64}}, ord::Vector{Matrix{Int64}}, λ::Vector{Matrix{Float64}}, regtype=1, solver="LSQR", flag = "on")
   Hsubs = data[1]
   Ssubs = data[2]
   Rs = data[3]
   return blocks2model(Hsubs,Rs,rcut[1],maxdeg[1],ord[1],λ[1],regtype,solver,flag), blocks2model(Ssubs,Rs,rcut[2],maxdeg[2],ord[2],λ[2],regtype,solver,flag,0,0)
end

@doc raw"""
Input: dat::Data (specify the file and index to read), par_H, par_S::Params, parameters for H,S fitting

Output: TBModelWhole, including (1) basis, (2) coefficients, 
(3) Mean value for diagonal blocks (Onsite only, or the mean values for other block are zero)
"""

function params2wmodels(dat::Data, par_H::Params, par_S::Params)
   if ison(dat)
      flag = "on"
      num_model = 6
   elseif isoff(dat)
      flag = "off"
      num_model = 9
   end
   data_whole = data_read(dat.file_name,dat.index)
   # ModelWhole_H, ModelWhole_S = data2model(data_whole[1], [par_H.rcutset[1], par_S.rcutset[1]], [par_H.maxdegset[1], par_S.maxdegset[1]], [par_H.ordset[1], par_S.ordset[1]], [par_H.regset[1], par_S.regset[1]], par_H.regtype, par_H.solver, flag)
   ModelWhole_H = Vector{Vector{TBModel}}([])
   ModelWhole_S = Vector{Vector{TBModel}}([])
   for i = 1 : num_model
      mh, ms = data2model(data_whole[i], [par_H.rcutset[i], par_S.rcutset[i]], [par_H.maxdegset[i], par_S.maxdegset[i]], [par_H.ordset[i], par_S.ordset[i]], [par_H.regset[i], par_S.regset[i]], par_H.regtype, par_H.solver, flag)
      push!(ModelWhole_H, mh)
      push!(ModelWhole_S, ms)
   end
   if flag == "on"
      return OnModelWhole(dat,par_H,"H", ModelWhole_H[1],ModelWhole_H[2],ModelWhole_H[3],ModelWhole_H[4],ModelWhole_H[5],ModelWhole_H[6]), OnModelWhole(dat,par_S,"S",ModelWhole_S[1],ModelWhole_S[2],ModelWhole_S[3],ModelWhole_S[4],ModelWhole_S[5],ModelWhole_S[6]), data_whole
   elseif flag == "off"
      return OffModelWhole(dat,par_H,"H", ModelWhole_H[1],ModelWhole_H[2],ModelWhole_H[3],ModelWhole_H[4],ModelWhole_H[5],ModelWhole_H[6],ModelWhole_H[7],ModelWhole_H[8],ModelWhole_H[9]), OffModelWhole(dat,par_S,"S",ModelWhole_S[1],ModelWhole_S[2],ModelWhole_S[3],ModelWhole_S[4],ModelWhole_S[5],ModelWhole_S[6],ModelWhole_S[7],ModelWhole_S[8],ModelWhole_S[9]), data_whole
   end
end

# function scaling(pibasis::PIBasis, p)
#    ww = zeros(Float64, length(pibasis))
#    bspec = get_spec(pibasis)
#    for i = 1:length(pibasis)
#       for b in bspec[i]
#          # TODO: revisit how this should be implemented for a general basis
#          for bb in b
#             if typeof(values(bb))<:Number
#                ww[i] += sum( abs(bb).^p )
#             end
#          end
#       end
#    end
#    return ww
# end

end
