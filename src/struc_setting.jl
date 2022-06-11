module Structure

using ACE, SparseArrays, LinearAlgebra
using ACE: PolyTransform, SphericalMatrix, PIBasis, SymmetricBasis,
           SimpleSparseBasis, Utils.RnYlm_1pbasis, CylindricalBondEnvelope, 
           Categorical1pBasis, get_spec#, coco_dot

export Data, Params, OnsiteBasis, OffsiteBasis, TBModel, TBModelWhole, OnModelWhole, OffModelWhole, ison, isoff, get_sites
import LinearAlgebra.adjoint, LinearAlgebra.transpose, Base./

adjoint(A::SphericalMatrix{L1, L2, LEN1, LEN2, T, LL}) where {L1, L2, LEN1, LEN2, T, LL} = SphericalMatrix(A.val', Val{L2}(), Val{L1}())
transpose(A::SphericalMatrix{L1, L2, LEN1, LEN2, T, LL}) where {L1, L2, LEN1, LEN2, T, LL} = SphericalMatrix(transpose(A.val), Val{L2}(), Val{L1}())
/(A::SphericalMatrix{L1, L2, LEN1, LEN2, T, LL}, b::Number) where {L1, L2, LEN1, LEN2, T, LL} = SphericalMatrix(A.val/b, Val{L1}(), Val{L2}())

## Transfer block index and block number
const D_ind2L = Dict(  1 => [0,0],
                       2 => [1,0],
                       3 => [2,0],
                       4 => [1,1],
                       5 => [2,1],
                       6 => [2,2] )

const L2D_ind = Dict(  [0,0] => 1,
                       [1,0] => 2,
                       [2,0] => 3,
                       [1,1] => 4,
                       [2,1] => 5,
                       [2,2] => 6 )

## keys for data & parameters
struct Data{T}
   file_name::Vector{String}
   index::Vector{T}
end

get_length(data::Data) = length(data.file_name)
function get_sites(data::Data{T}) where {T}
   if T == Int64 || T == Tuple{Int64,Int64}
      return length(data.file_name) * length(data.index)
   else
      return sum(length.(data.index))
   end
end
ison(data::Data{T}) where {T} = (T == Int64 || T == Vector{Int64})
isoff(data::Data{T}) where {T} = (T == Tuple{Int64,Int64} || T == Vector{Tuple{Int64,Int64}})

"""
Params(rcutset, maxdegset, ordset, regset, regtype, solver)
*_set: Cutoff radii, max polynomial degree, correlation order, regularisation parameters
       for Hamiltonian and overlap blocks, respectively. They are vectors that contain: 
          for on-site:
          - 3*3 values for ss
          - 2*3 values for sp
          - 1*3 values for sd
          - 2*2 values for pp
          - 1*2 values for pd
          - 1*1 values for dd
          for off-site:
             - 3*3 values for ss
             - 2*3 values for sp
             - 1*3 values for sd
             - 3*2 values for ps
             - 2*2 values for pp
             - 1*2 values for pd
             - 3*1 values for ds
             - 2*1 values for dp
             - 1*1 values for dd
       all pars should be stored as matrix, if each element is given as a number only, it will be generated to a matrix with same value.
       # for on-site, rcut means the radii of the circle we concern about around the centre atom
       # for offsite, rcut means the bond cutoff, and the corresponding envcutoff is chosen to be rcut/2,
         more specifically, `env` is defined as ACE.CylindricalBondEnvelope(rcut,rcut/2,rcut/2)
regtype: parameter for choosing regularisation type, 1 => Standard regularasation, 2 => Ticknov regularasation
solver: Name of the solver used for LS problem - options: "QR", "LSQR", "NaiveSolver (A'AC = A'Y) "
"""
struct Params
    rcutset::Vector{Matrix{Float64}}
    maxdegset::Vector{Matrix{Int64}}
    ordset::Vector{Matrix{Int64}}
    regset::Vector{Matrix{Float64}}
    regtype::Int64
    solver::String
end

fill_par_on(set::Union{Vector{Matrix{Float64}},Vector{Matrix{Int64}}}) = set
fill_par_on(set::Union{Vector{Float64},Vector{Int64}}) = 
   [repeat([set[1]],3,3), repeat([set[2]],2,3), repeat([set[3]],1,3), repeat([set[4]],2,2), repeat([set[5]],1,2), repeat([set[6]],1,1)]
fill_par_off(set::Union{Vector{Matrix{Float64}},Vector{Matrix{Int64}}}) = set
fill_par_off(set::Union{Vector{Float64},Vector{Int64}}) = 
   [repeat([set[1]],3,3), repeat([set[2]],2,3), repeat([set[3]],1,3), repeat([set[4]],3,2), repeat([set[5]],2,2), repeat([set[6]],1,2), repeat([set[7]],3,1), repeat([set[8]],2,1), repeat([set[9]],1,1)]

function Params(rcutset::Union{Float64,Vector{Float64},Vector{Matrix{Float64}}}, maxdegset::Union{Int64,Vector{Int64},Vector{Matrix{Int64}}}, ordset::Union{Int64,Vector{Int64},Vector{Matrix{Int64}}},
       regset::Union{Float64,Vector{Float64},Vector{Matrix{Float64}}}, regtype::Int64, solver::String)
   if length(rcutset) == 6
      return Params(fill_par_on(rcutset), fill_par_on(maxdegset), fill_par_on(ordset), fill_par_on(regset), regtype, solver)
   elseif length(rcutset) == 9
      return Params(fill_par_off(rcutset), fill_par_off(maxdegset), fill_par_off(ordset), fill_par_off(regset), regtype, solver)
   else
      @error("Invalid length!")
   end
end

ison(par::Params) = (length(par.rcutset) == 6)
isoff(par::Params) = (length(par.rcutset) == 9)
## Onsite basis
struct OnsiteBasis
   rcut::Float64
   maxdeg::Int64
   ord::Int64
   basis::SymmetricBasis
end

function OnsiteBasis(rcut::Float64, maxdeg::Int64, ord::Int64, L1, L2)
   # Parameters for basis
   r0 = 2.5
   rin = 0.5 * r0

   B1p = RnYlm_1pbasis(; maxdeg = maxdeg, r0 = r0, trans = PolyTransform(1, 1/r0^2), rcut = rcut, rin = rin)
   Bsel = SimpleSparseBasis(ord, maxdeg)
   φ = SphericalMatrix(L1, L2; T = ComplexF64)
   pibasis = PIBasis(B1p, Bsel; property = φ, isreal = false)
   basis = SymmetricBasis(φ, pibasis)

   return OnsiteBasis(rcut,maxdeg,ord,basis)
end

## Offsite basis
struct OffsiteBasis
   rcut::Float64
   maxdeg::Int64
   ord::Int64
   basis::SymmetricBasis
end

function maxdeg2filterfun(maxdeg,λ_n=.5,λ_l=.5)
   return bb -> filter_offsite_be(bb,maxdeg,λ_n,λ_l)
end

function filter_offsite_be(bb,maxdeg,λ_n=.5,λ_l=.5)
   if length(bb) == 0; return false; end 
   deg_n = ceil(Int64,maxdeg * λ_n)
   deg_l = ceil(Int64,maxdeg * λ_l)
   for b in bb
      if (b.be == :env) && (b.n>deg_n || b.l>deg_l)
         return false
      end
      #if (b.be == :bond) && ( b.l>deg_l )
      #   return false
      #end
   end
   return ( sum( b.be == :bond for b in bb ) == 1 )
end

function OffsiteBasis_nonsym(rcut::Float64, maxdeg::Int64, ord::Int64, L1, L2, λ_n=.5, λ_l=.5)
   # Environment parameters for basis
   r0 = 2.5
   rin = 0.5 * r0
   #r0cut = 10.0
   renv = 10-rcut/2
   env = ACE.CylindricalBondEnvelope(rcut,renv,renv;λ=.0)

   Bsel = SimpleSparseBasis(ord+1, maxdeg)

   # oneparticle basis
   RnYlm = RnYlm_1pbasis(; maxdeg = maxdeg, r0 = r0, trans = PolyTransform(1, 1/r0^2), rcut = sqrt(renv^2+(rcut+renv)^2), rin = rin, pcut = 2)
   #ACE.init1pspec!(RnYlm, Bsel)
   B1p_env = env * RnYlm
   #ACE.init1pspec!(B1p_env, Bsel)
   B1p = ACE.Categorical1pBasis([:bond, :env]; varsym = :be, idxsym=:be) * B1p_env
   #ACE.init1pspec!(B1p, Bsel)
   
   # SymmetricBasis
   φ = SphericalMatrix(L1, L2; T = ComplexF64)
   basis = SymmetricBasis(φ, B1p, Bsel; filterfun = maxdeg2filterfun(maxdeg,λ_n,λ_l))

   return basis
end

function OffsiteBasis(rcut::Float64, maxdeg::Int64, ord::Int64, L1, L2, λ_n=.5, λ_l=.5)
   # construct non-symmetric offsite basis
   b = OffsiteBasis_nonsym(rcut,maxdeg,ord,L1,L2,λ_n,λ_l)
   if L1 == L2
      bb = b
   else
      bb = OffsiteBasis_nonsym(rcut,maxdeg,ord,L2,L1,λ_n,λ_l)
   end
   # symmetrise coefficients
   U = sym_coeffs(bb.A2Bmap, bb.pibasis)
   U_new = dropzeros((b.A2Bmap + U)./2)
                      
   G = [ length(notzero(U_new,a,b)) == 0 ? 0 : sum( coco_dot(U_new[a,i], U_new[b,i]) for i in notzero(U_new,a,b) ) 
           for a = 1:size(U_new)[1], b = 1:size(U_new)[1] ]
   svdC = svd(G)
   rk = rank(Diagonal(svdC.S), rtol = 1e-7)
   Ured = Diagonal(sqrt.(svdC.S[1:rk])) * svdC.U[:, 1:rk]'
   U_new = sparse(Ured * U_new)
   dropzeros!(U_new)
   
   # construct symmetric offsite basis
   basis = SymmetricBasis(b.pibasis,U_new,b.symgrp,b.real)
   
   return OffsiteBasis(rcut,maxdeg,ord,basis)
end
           
notzero(U,a,b) = intersect(U[a,:].nzind, U[b,:].nzind)

function sym_coeffs(U::SparseMatrixCSC{T,F},bpi) where {T,F}
   A = get_spec(bpi)
   D = Dict{Any, Int}()
   for (i, val) in enumerate(A)
       D[val]= i
   end
   UU = spzeros(eltype(transpose.(U)),size(U)[1],size(U)[2])
   for i = 1:size(UU)[1]
       for j = 1:size(UU)[2]
           if norm(U[i,j]) ≠ 0
               sgn = 0
               U_temp = copy(A[j])
               for (i,k) in enumerate(A[j])
                   #if k.be == :bond
                   U_temp[i] = (be = k.be, n = k.n, l = k.l, m = -k.m)
                   sgn += k.m
                   #end
               end
               if !(U_temp in A)
                   for UU_temp in A
                       if sort(UU_temp) == sort(U_temp)
                           U_temp = UU_temp
                       end
                   end
               end
               UU[i,j] = (-1)^(sgn+sort(U_temp)[1].l) * adjoint(U[i,D[U_temp]])
           end
       end
   end
   return UU
end
## Models for fitting TB Hamiltonian
"""
An ACE model usually has fields basis & coeffs, 
b in basis has the same size of target;
coeffs is a vector that has the same length of basis;
We introduce an extra field `mean` to normalise the data
"""
struct TBModel
   basis::SymmetricBasis
   coeffs
   mean
end

abstract type TBModelWhole end

struct OnModelWhole <: TBModelWhole
   data::Data
   params::Params
   type::String
   ModelSS::Vector{TBModel}
   ModelSP::Vector{TBModel}
   ModelSD::Vector{TBModel}
   ModelPP::Vector{TBModel}
   ModelPD::Vector{TBModel}
   ModelDD::Vector{TBModel}
end

struct OffModelWhole <: TBModelWhole
   data::Data
   params::Params
   type::String
   ModelSS::Vector{TBModel}
   ModelSP::Vector{TBModel}
   ModelSD::Vector{TBModel}
   ModelPS::Vector{TBModel}
   ModelPP::Vector{TBModel}
   ModelPD::Vector{TBModel}
   ModelDS::Vector{TBModel}
   ModelDP::Vector{TBModel}
   ModelDD::Vector{TBModel}
end

ison(model::TBModelWhole) = ison(model.params)
isoff(model::TBModelWhole) = isoff(model.params)

## Experiment code that hacked from ACE.jl

# import ACE._inner_evaluate
# using ACE: AbstractState, _evaluate_env, _evaluate_bond
# 
# function _inner_evaluate(env::CylindricalBondEnvelope, X::AbstractState)
#    if X.be == :env
#       return _evaluate_env(env, X) * _evaluate_bond(env, X)
#    elseif X.be == :bond
#       return _evaluate_bond(env, X)
#    else
#       error("invalid X.be value")
#    end
# end

end
