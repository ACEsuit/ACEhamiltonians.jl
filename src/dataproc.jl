module DataProcess

using HDF5, JuLIP, StaticArrays, LinearAlgebra, ACEatoms
using ACE: PositionState, BondEnvelope, filter, State, CylindricalBondEnvelope

export get_atoms, get_HSR, data_read

# TODO: must have existed somewhere...
const Z2Sym = Dict( 0 => :X, 1 => :H, 2 => :He, 6 => :C, 8 => :O, 9 => :F, 13 => :Al, 16 => :Si)
const Sym2Z = Dict( :X => 0, :H => 1, :He => 2, :C => 6, :O => 8, :F => 9, :Al => 13, :Si => 16)

## Data read

"""
`get_HSR(fname, index, L1, L2)`

Input: filename, index, L1, L2;
Output: Onsite L1-L2 blocks and the corresponding relative position;

If `typeof(index)<:Vector{Int}``, then it reads i-th (i in index) onsite L1-L2
blocks from a file called filename;

Elseif `typeof(index)<:Vector{Tuple}``, then it reads (i,j)-th ((i,j) in index)
offsite L1-L2 blocks from a file called filename;
"""

function get_atoms(fd::HDF5.File; groupname=nothing)
   groupname === nothing && (groupname = HDF5.name(first(fd)))
   positions = HDF5.read(fd, string(groupname,"/positions"))
   unitcell = HDF5.read(fd, string(groupname,"/unitcell"))
   species = HDF5.read(fd, string(groupname,"/species"))
   atoms = JuLIP.Atoms(; X = positions, Z = species,
                        cell = unitcell,
                        pbc = [true, true, true])
   return [unitcell, species, positions], atoms
end

function get_atoms(fname::String; groupname=nothing)
   HDF5.h5open(fname, "r") do fd
      get_atoms(fd; groupname=groupname)
   end
end

function get_HSR(fname::String,index,L1::Int64,L2::Int64)
   data = []
   HDF5.h5open(fname, "r") do fd
   groupnames = []
   for obj in fd
      push!(groupnames, HDF5.name(obj))
   end
   group_name = groupnames[1]
   dataset_names = []
   for obj in fd[group_name]
      push!(dataset_names, HDF5.name(obj))
   end
   HS_data = []
   H_str = string(group_name,"/H")
   S_str = string(group_name,"/S")
   if haskey(fd, H_str)
      H = HDF5.read(fd, string(group_name,"/H"))
      push!(HS_data, H)
   end
   if haskey(fd, S_str)
      S = HDF5.read(fd, string(group_name,"/S"))
      push!(HS_data, S)
   end
   push!(data, HS_data)
   atoms_data, atoms = get_atoms(fd; groupname=group_name)
   push!(data, atoms_data)
   push!(data, atoms)
   end
   return data_preprocess(data,L1,L2,index)
end

get_HSR(fname::String,index) = [ get_HSR(fname,index,L1,L2) for L2=0:2 for L1=L2:2 ]

## Data reshape
function hs2hssub(H,S,n,L1,L2,index::Vector{Int64})
   if length(index)>n | maximum(index)>n | minimum(index)<1
      error("index exceed the number of atom!")
   end
   sizehi = Int(size(H)[1]/n)
   # imax = 3-L1
   # jmax = 3-L2
   # tL1 = Int(sum([0,3,6][1:L1]))+1
   # tL2 = Int(sum([0,3,6][1:L2]))+1
   # hsub = [[SMatrix{2L1+1,2L2+1,ComplexF64}(H[sizehi*(k-1)+1:sizehi*k,sizehi*(k-1)+1:sizehi*k][(tL1+(2L1+1)*(i-1)):(tL1+(2L1+1)*i-1),(tL2+(2L2+1)*(j-1)):(tL2+(2L2+1)*j-1)]) for i=1:imax, j=1:jmax] for k = 1:n]
   # return hsub
   ## TODO: It might be better to save Hsub, Ssub as tensors so that they could be comparable to the predicted ones directly...
   if L1 == L2 == 0
      hss = [[SMatrix{2L1+1,2L2+1,Float64}(H[sizehi*(i-1)+1:sizehi*i,sizehi*(i-1)+1:sizehi*i][k,j]) for k=1:3, j=1:3] for i in index]
      sss = [[SMatrix{2L1+1,2L2+1,Float64}(S[sizehi*(i-1)+1:sizehi*i,sizehi*(i-1)+1:sizehi*i][k,j]) for k=1:3, j=1:3] for i in index]
      return hss, sss
   elseif L1 == L2 == 1
      hpp = [[SMatrix{2L1+1,2L2+1,Float64}(H[sizehi*(i-1)+1:sizehi*i,sizehi*(i-1)+1:sizehi*i][4+3(k-1):6+3(k-1),4+3(j-1):6+3(j-1)]) for k = 1:2, j=1:2] for i in index]
      spp = [[SMatrix{2L1+1,2L2+1,Float64}(S[sizehi*(i-1)+1:sizehi*i,sizehi*(i-1)+1:sizehi*i][4+3(k-1):6+3(k-1),4+3(j-1):6+3(j-1)]) for k = 1:2, j=1:2] for i in index]
      return hpp, spp
   elseif L1 == L2 == 2
      hdd = [[SMatrix{2L1+1,2L2+1,Float64}(H[sizehi*(i-1)+1:sizehi*i,sizehi*(i-1)+1:sizehi*i][10:14,10:14])] for i in index]
      sdd = [[SMatrix{2L1+1,2L2+1,Float64}(S[sizehi*(i-1)+1:sizehi*i,sizehi*(i-1)+1:sizehi*i][10:14,10:14])] for i in index]
      return hdd, sdd
   elseif L1 == 1 && L2 == 0
      hsp = [[SMatrix{2L1+1,2L2+1,Float64}(H[sizehi*(i-1)+1:sizehi*i,sizehi*(i-1)+1:sizehi*i][4+3(k-1):6+3(k-1),j]) for k=1:2, j=1:3] for i in index]
      ssp = [[SMatrix{2L1+1,2L2+1,Float64}(S[sizehi*(i-1)+1:sizehi*i,sizehi*(i-1)+1:sizehi*i][4+3(k-1):6+3(k-1),j]) for k=1:2, j=1:3] for i in index]
      return hsp, ssp
   elseif L1 == 2 && L2 == 1
      hpd = [[SMatrix{2L1+1,2L2+1,Float64}(H[sizehi*(i-1)+1:sizehi*i,sizehi*(i-1)+1:sizehi*i][10:14,4+3(j-1):6+3(j-1)]) for k=1:1, j=1:2] for i in index]
      spd = [[SMatrix{2L1+1,2L2+1,Float64}(S[sizehi*(i-1)+1:sizehi*i,sizehi*(i-1)+1:sizehi*i][10:14,4+3(j-1):6+3(j-1)]) for k=1:1, j=1:2] for i in index]
      return hpd, spd
   elseif L1 == 2 && L2 == 0
      hsd = [[SMatrix{2L1+1,2L2+1,Float64}(H[sizehi*(i-1)+1:sizehi*i,sizehi*(i-1)+1:sizehi*i][10+5(k-1):14+5(k-1),j]) for k=1:1, j=1:3] for i in index]
      ssd = [[SMatrix{2L1+1,2L2+1,Float64}(S[sizehi*(i-1)+1:sizehi*i,sizehi*(i-1)+1:sizehi*i][10+5(k-1):14+5(k-1),j]) for k=1:1, j=1:3] for i in index]
      return hsd, ssd
   end
end

function hs2hssub(H,S,n,L1,L2,index::Vector{Tuple{Int64,Int64}})
   if length(index)>n*(n-1)/2 #| maximum(index)>n | minimum(index)<1
      error("index exceed the number of atom!")
   end
   sizehi = Int(size(H)[1]/n)
   if L1 == L2 == 0
      hss = [[SMatrix{2L1+1,2L2+1,Float64}(H[sizehi*(i-1)+1:sizehi*i,sizehi*(j-1)+1:sizehi*j][k,l]) for k=1:3, l=1:3] for (i,j) in index]
      sss = [[SMatrix{2L1+1,2L2+1,Float64}(S[sizehi*(i-1)+1:sizehi*i,sizehi*(j-1)+1:sizehi*j][k,l]) for k=1:3, l=1:3] for (i,j) in index]
      return hss, sss
   elseif L1 == L2 == 1
      hpp = [[SMatrix{2L1+1,2L2+1,Float64}(H[sizehi*(i-1)+1:sizehi*i,sizehi*(j-1)+1:sizehi*j][4+3(k-1):6+3(k-1),4+3(l-1):6+3(l-1)]) for k = 1:2, l=1:2] for (i,j) in index]
      spp = [[SMatrix{2L1+1,2L2+1,Float64}(S[sizehi*(i-1)+1:sizehi*i,sizehi*(j-1)+1:sizehi*j][4+3(k-1):6+3(k-1),4+3(l-1):6+3(l-1)]) for k = 1:2, l=1:2] for (i,j) in index]
      return hpp, spp
   elseif L1 == L2 == 2
      hdd = [[SMatrix{2L1+1,2L2+1,Float64}(H[sizehi*(i-1)+1:sizehi*i,sizehi*(j-1)+1:sizehi*j][10:14,10:14])] for (i,j) in index]
      sdd = [[SMatrix{2L1+1,2L2+1,Float64}(S[sizehi*(i-1)+1:sizehi*i,sizehi*(j-1)+1:sizehi*j][10:14,10:14])] for (i,j) in index]
      return hdd, sdd
   elseif L1 == 1 && L2 == 0
      hsp = [[SMatrix{2L1+1,2L2+1,Float64}(H[sizehi*(i-1)+1:sizehi*i,sizehi*(j-1)+1:sizehi*j][4+3(k-1):6+3(k-1),l]) for k=1:2, l=1:3] for (i,j) in index]
      ssp = [[SMatrix{2L1+1,2L2+1,Float64}(S[sizehi*(i-1)+1:sizehi*i,sizehi*(j-1)+1:sizehi*j][4+3(k-1):6+3(k-1),l]) for k=1:2, l=1:3] for (i,j) in index]
      return hsp, ssp
   elseif L1 == 0 && L2 == 1
      hps = [[SMatrix{2L1+1,2L2+1,Float64}(H[sizehi*(i-1)+1:sizehi*i,sizehi*(j-1)+1:sizehi*j][k,4+3(l-1):6+3(l-1)]) for k=1:3, l=1:2] for (i,j) in index]
      sps = [[SMatrix{2L1+1,2L2+1,Float64}(S[sizehi*(i-1)+1:sizehi*i,sizehi*(j-1)+1:sizehi*j][k,4+3(l-1):6+3(l-1)]) for k=1:3, l=1:2] for (i,j) in index]
      return hps, sps   
   elseif L1 == 2 && L2 == 1
      hpd = [[SMatrix{2L1+1,2L2+1,Float64}(H[sizehi*(i-1)+1:sizehi*i,sizehi*(j-1)+1:sizehi*j][10:14,4+3(l-1):6+3(l-1)]) for k=1:1, l=1:2] for (i,j) in index]
      spd = [[SMatrix{2L1+1,2L2+1,Float64}(S[sizehi*(i-1)+1:sizehi*i,sizehi*(j-1)+1:sizehi*j][10:14,4+3(l-1):6+3(l-1)]) for k=1:1, l=1:2] for (i,j) in index]
      return hpd, spd
   elseif L1 == 1 && L2 == 2
      hdp = [[SMatrix{2L1+1,2L2+1,Float64}(H[sizehi*(i-1)+1:sizehi*i,sizehi*(j-1)+1:sizehi*j][4+3(k-1):6+3(k-1),10:14]) for k=1:2, l=1:1] for (i,j) in index]
      sdp = [[SMatrix{2L1+1,2L2+1,Float64}(S[sizehi*(i-1)+1:sizehi*i,sizehi*(j-1)+1:sizehi*j][4+3(k-1):6+3(k-1),10:14]) for k=1:2, l=1:1] for (i,j) in index]
      return hdp, sdp
   elseif L1 == 2 && L2 == 0
      hsd = [[SMatrix{2L1+1,2L2+1,Float64}(H[sizehi*(i-1)+1:sizehi*i,sizehi*(j-1)+1:sizehi*j][10+5(k-1):14+5(k-1),l]) for k=1:1, l=1:3] for (i,j) in index]
      ssd = [[SMatrix{2L1+1,2L2+1,Float64}(S[sizehi*(i-1)+1:sizehi*i,sizehi*(j-1)+1:sizehi*j][10+5(k-1):14+5(k-1),l]) for k=1:1, l=1:3] for (i,j) in index]
      return hsd, ssd
   elseif L1 == 0 && L2 == 2
      hds = [[SMatrix{2L1+1,2L2+1,Float64}(H[sizehi*(i-1)+1:sizehi*i,sizehi*(j-1)+1:sizehi*j][10+5(l-1):14+5(l-1),k]) for k=1:3, l=1:1] for (i,j) in index]
      sds = [[SMatrix{2L1+1,2L2+1,Float64}(S[sizehi*(i-1)+1:sizehi*i,sizehi*(j-1)+1:sizehi*j][10+5(l-1):14+5(l-1),k]) for k=1:3, l=1:1] for (i,j) in index]
      return hds, sds
   end
end

## Onsite data_preprocess

function data_preprocess(data,L1,L2,index::Vector{Int64})
   H = data[1][1][:,:]
   S = data[1][2][:,:]
   at = data[3]
   n_atom = length(at.X)

   Hsub, Ssub = hs2hssub(H,S,n_atom,L1,L2,index)
   nlist = JuLIP.neighbourlist(at,20.0)
   Rs = [ get_state_on(nlist,at,i) for i = 1:length(index) ]
   
   return Hsub,Ssub,Rs
end

function get_state_on(nlist,at::JuLIP.Atoms,i::Int64)
   neigh_i = JuLIP.Potentials.neigsz(nlist,at,i)
   return [ ACEatoms.AtomState{Float64}(mu = neigh_i[3][j], mu0 = at.Z[i], rr = neigh_i[2][j]) for j = 1:length(neigh_i[2]) ]
end

## Offsite data_preprocess - NOTE: MIC is used here
# const AtomOffsiteState = State{NamedTuple{(:mu, :mu0, :mu1, :rr, :rr0, :be), Tuple{AtomicNumber, AtomicNumber, AtomicNumber, SVector{3, Float64}, SVector{3, Float64}, Symbol}}}
const AtomOffsiteState = State{(:mu, :mu0, :mu1, :rr, :rr0, :be), Tuple{AtomicNumber, AtomicNumber, AtomicNumber, SVector{3, Float64}, SVector{3, Float64}, Symbol}}
# const AtomOffsiteState{T} = State{NamedTuple{(:mu, :mu01, :rr, :rr0, :be), Tuple{AtomicNumber, Tuple{AtomicNumber, AtomicNumber}, SVector{3, T}, SVector{3, T}, Symbol}}}
## TODO: for mu0 and mu1, is it better to change them to (mu0,mu1) <: Tuple{AtomicNumber,AtomicNumber}? Also, for env state, this is not needed?

function get_bond(at,i,j;MIC=true)
   nlist = JuLIP.neighbourlist(at,20.0)
   neigh_i = JuLIP.Potentials.neigsz(nlist,at,i)
   idx = findall(isequal(j),neigh_i[1])
   if MIC
      idx = idx[findmin(norm.(neigh_i[2][idx]))[2]]
   end
   return idx
end

function get_state(at,i::Int64,j::Int64,env::BondEnvelope)
   if i==j
      error("i,j should be distinct")
   end
   nlist = JuLIP.neighbourlist(at,20.0)
   neigh_i = JuLIP.Potentials.neigsz(nlist,at,i)
   idx = get_bond(at,i,j)
   rr = neigh_i[2][idx]
   st = AtomOffsiteState(mu = at.Z[j], mu0 = at.Z[i], mu1 = at.Z[j], rr = rr, rr0 = rr, be=:bond)
   #st = [ State(rr = neigh_i[2][j], rr0 = neigh_i[2][j], be=:bond) for j in set]
   for (jj, rj) in enumerate(neigh_i[2])
      st_temp = AtomOffsiteState(mu = neigh_i[3][jj], mu0 = at.Z[i], mu1 = at.Z[j], rr = rj, rr0 = rr, be=:env)
      if rj≠rr && filter(env,st_temp)
         st = [st; st_temp]
      end
   end
   return st
end

get_state(at,index,env=CylindricalBondEnvelope(18.0,10.0,10.0)) = [ get_state(at,i,j,env) for (i,j) in index ]

function data_preprocess(data,L1,L2,index::Vector{Tuple{Int64, Int64}},env=CylindricalBondEnvelope(18.0,10.0,10.0))
   H = data[1][1][:,:]
   S = data[1][2][:,:]
   at = data[3]
   n_atom = length(at.X)

   Hsub, Ssub = hs2hssub(H,S,n_atom,L1,L2,index)
   Rs = get_state(at,index,env)
   return Hsub,Ssub,Rs
end

# Input: filename; Output: Exact onsite H&S
## TODO: to be removed...
function get_HS(fname)
   data = []
   HDF5.h5open(fname, "r") do fd
   groupnames = []
   for obj in fd
      push!(groupnames, HDF5.name(obj))
   end
   group_name = groupnames[1]
   dataset_names = []
   for obj in fd[group_name]
      push!(dataset_names, HDF5.name(obj))
   end
   HS_data = []
   H_str = string(group_name,"/H")
   S_str = string(group_name,"/S")
   if haskey(fd, H_str)
      H = HDF5.read(fd, string(group_name,"/H"))
      push!(HS_data, H)
   end
   if haskey(fd, S_str)
      S = HDF5.read(fd, string(group_name,"/S"))
      push!(HS_data, S)
   end
   push!(data, HS_data)
   end
   return data
end


"""
`data_read(fname, index, L1, L2)`

Input: filenames, index, L1, L2;
Output: Onsite L1-L2 blocks and the corresponding relative position;

If `typeof(index)<:Vector{Int}`` or `Vector{Vector{Int}}``, then it reads i-th
(i in index) onsite L1-L2 blocks from files whose name is in filenames;
Elseif `typeof(index)<:Vector{Tuple}`` or `Vector{Vector{Tuple}}``, then it reads
(i,j)-th ((i,j) in index) offsite L1-L2 blocks from files whose name is in filename;

When L1, L2 are omitted, it will read all blocks (ordered as ss, sp, sd, pd, pp, dd)
"""
# Multiple data reading: read data from H5 files named as `name in file_name`
function data_read(file_name::Vector{String},index,L1::Int64,L2::Int64)

   if typeof(index[1])<:Vector

      if length(index) != length(file_name)
         error("Index set(s) should either be a vector or have the same length as fime_name")
      end

      data = get_HSR(file_name[1],index[1],L1,L2)

      for L = 2:length(file_name)
         data_temp = get_HSR(file_name[L],index[L],L1,L2)
         data = ([data[1]; data_temp[1]], [data[2]; data_temp[2]], [data[3]; data_temp[3]])
      end

   else

      data = get_HSR(file_name[1],index,L1,L2)

      for L = 2:length(file_name)
         data_temp = get_HSR(file_name[L],index,L1,L2)
         data = ([data[1]; data_temp[1]], [data[2]; data_temp[2]], [data[3]; data_temp[3]])
      end
   end

   return data
end

function data_read(file_name::Vector{String}, index::Vector{T}) where{T}
   if T <: Union{Int64,Vector{Int64}}
      return [ data_read(file_name,index,L1,L2) for L2=0:2 for L1=L2:2 ]
   elseif T <: Union{Tuple{Int64,Int64},Vector{Tuple{Int64,Int64}}}
      return [ data_read(file_name,index,L1,L2) for L2=0:2 for L1=0:2 ]
   else
      @warn("Possibly wrong index type")
   end
end

## map from a whole configuration to an atom list -> showing what types of atoms are there in this config
at2zlist(at::JuLIP.Atoms) = unique(at.Z)
# NOTE: If we need Integer rather than atomic number then 
# [ Z2Sym[unique(at.Z)[i].z] for i = 1:length(unique(at.Z)) ]

## map from an atom list to model list - showing types of onsite models needed
zlist2mlist_on(zlist) = Dict([(zlist[i],i) for i = 1:length(zlist)])

## map from an atom list to model list - showing types of offsite models needed 
zlist2mlist_off(zlist) = Dict([((zlist[i],zlist[j]),(i,j)) for i = 1:length(zlist) for j = 1:length(zlist)])

end