module Dictionary

using ACE, LinearAlgebra, JuLIP, StaticArrays
using ACEhamiltonians.Structure
using ACE: CylindricalBondEnvelope, SphericalMatrix
using ACEhamiltonians.Predict:ij2k
import ACE.write_dict, ACE.read_dict

export write_dict, read_dict

write_dict(A::Matrix{T}) where {T} =
             Dict("__id__" => "ACE_Matrix",
                    "vals" => write_dict.(A) )

read_dict(::Val{:ACE_Matrix}, D::Dict) = read_dict.(D["vals"])

write_dict(D::Diagonal{T}) where {T} =
             Dict("__id__" => "Diag_Matrix",
                    "vals" => write_dict(D.diag),
                       "T" => write_dict(T) )

read_dict(::Val{:Diag_Matrix}, D::Dict) = Diagonal(read_dict(D["vals"]))

write_dict(φ::SphericalMatrix{L1, L2, Len1, Len2, T}) where {L1,L2,Len1,Len2,T} =
             Dict("__id__" => "SphericalMatrix",
                      "L1" => L1,
                      "L2" => L2,
                     "val" => write_dict(φ.val),
                       "T" => write_dict(T) )

function read_dict(::Val{:SphericalMatrix}, D::Dict)
   #T = D["T"]
   #L1 = D["L1"]
   #L2 = D["L2"]
   #val = read_dict(D["val"])
   return SphericalMatrix{D["L1"], D["L2"], 2D["L1"]+1, 2D["L2"]+1, read_dict(D["T"]), (2D["L1"]+1)*(2D["L2"]+1)}( SMatrix{2D["L1"]+1, 2D["L2"]+1, read_dict(D["T"])}(read_dict(D["val"])), Val(D["L1"]), Val(D["L2"]) )
end

write_dict(sm::SMatrix{Len1, Len2, T, D}) where {Len1, Len2, T, D} = 
   Dict("__id__" => "SMatrix",
          "Len1" => Len1,
          "Len2" => Len2,
           "val" => sm.data,
             "T" => write_dict(T) )
read_dict(::Val{:SMatrix}, D::Dict) = SMatrix{D["Len1"],D["Len2"],read_dict(D["T"]),D["Len1"]*D["Len2"]}(absvec2valvec(D["val"]))

function absvec2valvec(val)
   if typeof(val[1]) <: Dict
      return [ val[i]["re"] + val[i]["im"]*im for i = 1:length(val) ]
   else
      return val
   end
end

write_dict(dat::Data) =
      Dict(   "__id__" => "Data_key",
            "file_name" => dat.file_name,
               "index" => dat.index)

function read_dict(::Val{:Data_key}, D::Dict)
   fname = D["file_name"]
   fname = Vector{String}(fname)
   idx = D["index"]
   if typeof(idx[1])<:Int64 || typeof(idx[1])<:Tuple{Int64,Int64}
      index = Vector{typeof(idx[1])}(idx)
   elseif typeof(idx[1])<:Vector{Any}
      if typeof(idx[1][1])<:Int64 || typeof(idx[1][1])<:Tuple{Int64,Int64}
         idx = [ Vector{typeof(idx[1][1])}(idx[i]) for i = 1:length(idx) ] 
      end
   end
   return Data(fname, idx)
end

write_dict(par::Params) =
      Dict(   "__id__" => "Paramset",
                "rcut" => par.rcutset,
              "maxdeg" => par.maxdegset,
               "order" => par.ordset,
             "reg_par" => par.regset,
            "reg_type" => par.regtype,
              "Solver" => par.solver )

function read_dict(::Val{:Paramset}, D::Dict)
   if typeof(D["rcut"])<:Vector{Any}
      return read_dict_fromjson(D)
   else
      return Params(D["rcut"], D["maxdeg"], D["order"], D["reg_par"], D["reg_type"], D["Solver"])
   end
end

function read_dict_fromjson(D::Dict)
   rcut = vecvec2vecmat(D["rcut"])
   maxdeg = vecvec2vecmat(D["maxdeg"])
   order = vecvec2vecmat(D["order"])
   reg_par = vecvec2vecmat(D["order"])
   return Params(rcut, maxdeg, order, reg_par, D["reg_type"], D["Solver"])
end

function vecvec2vecmat(target::Vector{Any})
    out = Vector{Matrix{Float64}}([])
    if length(target) == 6
      push!(out, [ target[1][1] target[1][2] target[1][3] ])
      push!(out, [ target[2][1] target[2][2] target[2][3] ])
      push!(out, [ target[3][1] target[3][2] target[3][3] ])
      push!(out, [ target[4][1] target[4][2] ])
      push!(out, [ target[5][1] target[5][2] ])
      push!(out, [ target[6][1][1] for i = 1:1,j=1:1])
   elseif length(target) == 9
      push!(out, [ target[1][1] target[1][2] target[1][3] ])
      push!(out, [ target[2][1] target[2][2] target[2][3] ])
      push!(out, [ target[3][1] target[3][2] target[3][3] ])
      push!(out, [ target[4][1] target[4][2] ])
      push!(out, [ target[5][1] target[5][2] ])
      push!(out, [ target[6][1] target[6][2] ])
      push!(out, [ target[7][1][i] for i = 1:3, j = 1:1 ])
      push!(out, [ target[8][1][i] for i = 1:2,j=1:1 ])
      push!(out, [ target[9][1][i] for i = 1:1,j=1:1 ])
   end   
   return out
end

write_dict(V::TBModel) =
      Dict( "__id__" => "ACEhamiltonians_TBModel",
             "basis" => write_dict(V.basis),
            "coeffs" => V.coeffs,
              "mean" => V.mean )

function read_dict(::Val{:ACEhamiltonians_TBModel}, D::Dict)

   basis = read_dict(D["basis"])
   coeffs = D["coeffs"]
   mean = deal_mean(D["mean"])

   return TBModel(basis, coeffs, mean)

end
function deal_mean(mean)
   if typeof(mean)<:Matrix
      return mean
   else
      Mean = [ mean[j][i] for i = 1:length(mean[1]), j = 1:length(mean) ]
   end
end

write_dict(env::CylindricalBondEnvelope) =
      Dict( "__id__" => "CylindricalBondEnvelope",
       "bond_cutoff" => env.r0cut,
      "radii_cutoff" => env.rcut,
       "axis_cutoff" => env.zcut,
            "p_bond" => env.p0,
           "p_radii" => env.pr,
            "p_axis" => env.pz,
            "floppy" => env.floppy,
                 "λ" => env.λ )

read_dict(::Val{:CylindricalBondEnvelope}, D::Dict) =
   CylindricalBondEnvelope( D["bond_cutoff"], D["radii_cutoff"], D["axis_cutoff"], D["p_bond"], D["p_radii"], D["p_axis"], D["floppy"], D["λ"] )

# TODO: I am considering to add L1 L2 parameter to TBModel so that this
#       could be done far more compact!!

write_dict(ModelW::OnModelWhole) =
      Dict( "__id__" => "ACEhamiltonians_OnModelWhole",
              "data" => write_dict(ModelW.data),
            "params" => write_dict(ModelW.params),
              "Type" => ModelW.type,
           "ModelSS" => write_dict.(ModelW.ModelSS),
           "ModelSP" => write_dict.(ModelW.ModelSP),
           "ModelSD" => write_dict.(ModelW.ModelSD),
           "ModelPP" => write_dict.(ModelW.ModelPP),
           "ModelPD" => write_dict.(ModelW.ModelPD),
           "ModelDD" => write_dict.(ModelW.ModelDD) )

function read_dict(::Val{:ACEhamiltonians_OnModelWhole}, D::Dict)

          data = read_dict(D["data"])
      paramset = read_dict(D["params"])
          type = D["Type"]
       ModelSS = read_dict.(D["ModelSS"])
       ModelSP = read_dict.(D["ModelSP"])
       ModelSD = read_dict.(D["ModelSD"])
       ModelPP = read_dict.(D["ModelPP"])
       ModelPD = read_dict.(D["ModelPD"])
       ModelDD = read_dict.(D["ModelDD"])

   return OnModelWhole(data,paramset,type,ModelSS,ModelSP,ModelSD,ModelPP,ModelPD,ModelDD)

end

write_dict(ModelW::OffModelWhole) =
      Dict( "__id__" => "ACEhamiltonians_OffModelWhole",
              "data" => write_dict(ModelW.data),
            "params" => write_dict(ModelW.params),
              "Type" => ModelW.type,
           "ModelSS" => write_dict.(ModelW.ModelSS),
           "ModelSP" => write_dict.(ModelW.ModelSP),
           "ModelSD" => write_dict.(ModelW.ModelSD),
           "ModelPS" => write_dict.(ModelW.ModelPS),
           "ModelPP" => write_dict.(ModelW.ModelPP),
           "ModelPD" => write_dict.(ModelW.ModelPD),
           "ModelDS" => write_dict.(ModelW.ModelDS),
           "ModelDP" => write_dict.(ModelW.ModelDP),
           "ModelDD" => write_dict.(ModelW.ModelDD) )

function read_dict(::Val{:ACEhamiltonians_OffModelWhole}, D::Dict)

          data = read_dict(D["data"])
      paramset = read_dict(D["params"])
          type = D["Type"]
       ModelSS = read_dict.(D["ModelSS"])
       ModelSP = read_dict.(D["ModelSP"])
       ModelSD = read_dict.(D["ModelSD"])
       ModelPS = read_dict.(D["ModelPS"])
       ModelPP = read_dict.(D["ModelPP"])
       ModelPD = read_dict.(D["ModelPD"])
       ModelDS = read_dict.(D["ModelDS"])
       ModelDP = read_dict.(D["ModelDP"])
       ModelDD = read_dict.(D["ModelDD"])

   return OffModelWhole(data,paramset,type,ModelSS,ModelSP,ModelSD,ModelPS,ModelPP,ModelPD,ModelDS,ModelDP,ModelDD)

end

## Model recover
#  Recover model/atoms from json file - TODO: there must be a more elegant way for model reconstruction - will consider label the model (by apping orbital symbol to number)!!

function recover_model(Dm::Dict)

   paramset = read_dict_fromjson(Dm["params"])
   data = read_dict(Dm["data"])
   idx = data.index

   if ison(paramset)#typeof(idx[1])<:Int64 || (typeof(idx[1])<:Union{Vector{Any},Vector{Int64}} && typeof(idx[1][1])<:Int64)
      return OnModelWhole( data, paramset, Dm["Type"],
       [ TBModel(OnsiteBasis(paramset.rcutset[1][i,j],paramset.maxdegset[1][i,j],paramset.ordset[1][i,j],0,0).basis, Vector{Float64}(Dm["ModelSS"][ij2k(i,j,3)]["coeffs"]), reshape(Vector{Float64}(Dm["ModelSS"][ij2k(i,j,3)]["mean"][1]),1,1)) for i = 1:3 for j = 1:3 ],
       [ TBModel(OnsiteBasis(paramset.rcutset[2][i,j],paramset.maxdegset[2][i,j],paramset.ordset[2][i,j],1,0).basis, Vector{Float64}(Dm["ModelSP"][ij2k(i,j,3)]["coeffs"]), reshape(Vector{Float64}(Dm["ModelSP"][ij2k(i,j,3)]["mean"][1]),3,1)) for i = 1:2 for j = 1:3 ],
       [ TBModel(OnsiteBasis(paramset.rcutset[3][i,j],paramset.maxdegset[3][i,j],paramset.ordset[3][i,j],2,0).basis, Vector{Float64}(Dm["ModelSD"][ij2k(i,j,3)]["coeffs"]), reshape(Vector{Float64}(Dm["ModelSD"][ij2k(i,j,3)]["mean"][1]),5,1)) for i = 1:1 for j = 1:3 ],
       [ TBModel(OnsiteBasis(paramset.rcutset[4][i,j],paramset.maxdegset[4][i,j],paramset.ordset[4][i,j],1,1).basis, Vector{Float64}(Dm["ModelPP"][ij2k(i,j,2)]["coeffs"]), [ Vector{Float64}.(Dm["ModelPP"][ij2k(i,j,2)]["mean"])[1] Vector{Float64}.(Dm["ModelPP"][ij2k(i,j,2)]["mean"])[2] Vector{Float64}.(Dm["ModelPP"][ij2k(i,j,2)]["mean"])[3] ]) for i = 1:2 for j = 1:2 ],
       [ TBModel(OnsiteBasis(paramset.rcutset[5][i,j],paramset.maxdegset[5][i,j],paramset.ordset[5][i,j],2,1).basis, Vector{Float64}(Dm["ModelPD"][ij2k(i,j,2)]["coeffs"]), [ Vector{Float64}.(Dm["ModelPD"][ij2k(i,j,2)]["mean"])[1] Vector{Float64}.(Dm["ModelPD"][ij2k(i,j,2)]["mean"])[2] Vector{Float64}.(Dm["ModelPD"][ij2k(i,j,2)]["mean"])[3] ]) for i = 1:1 for j = 1:2 ],
       [ TBModel(OnsiteBasis(paramset.rcutset[6][i,j],paramset.maxdegset[6][i,j],paramset.ordset[6][i,j],2,2).basis, Vector{Float64}(Dm["ModelDD"][ij2k(i,j,1)]["coeffs"]), [ Vector{Float64}.(Dm["ModelDD"][ij2k(i,j,1)]["mean"])[1] Vector{Float64}.(Dm["ModelDD"][ij2k(i,j,1)]["mean"])[2] Vector{Float64}.(Dm["ModelDD"][ij2k(i,j,1)]["mean"])[3] Vector{Float64}.(Dm["ModelDD"][ij2k(i,j,1)]["mean"])[4] Vector{Float64}.(Dm["ModelDD"][ij2k(i,j,1)]["mean"])[5]]) for i = 1:1 for j = 1:1 ] )
   elseif isoff(paramset)#typeof(idx[1])<:Tuple{Int64,Int64} || (typeof(idx[1])<:Vector{Any} && typeof(idx[1][1])<:Vector{Any})
      if Dm["Type"] == "H"
         return OffModelWhole( data, paramset, Dm["Type"],
         [ TBModel(OffsiteBasis(paramset.rcutset[1][i,j],paramset.maxdegset[1][i,j],paramset.ordset[1][i,j],0,0).basis, Vector{Float64}(Dm["ModelSS"][ij2k(i,j,3)]["coeffs"]), reshape(Vector{Float64}(Dm["ModelSS"][ij2k(i,j,3)]["mean"][1]),1,1)) for i = 1:3 for j = 1:3 ],
         [ TBModel(OffsiteBasis(paramset.rcutset[2][i,j],paramset.maxdegset[2][i,j],paramset.ordset[2][i,j],1,0).basis, Vector{Float64}(Dm["ModelSP"][ij2k(i,j,3)]["coeffs"]), reshape(Vector{Float64}(Dm["ModelSP"][ij2k(i,j,3)]["mean"][1]),3,1)) for i = 1:2 for j = 1:3 ],
         [ TBModel(OffsiteBasis(paramset.rcutset[3][i,j],paramset.maxdegset[3][i,j],paramset.ordset[3][i,j],2,0).basis, Vector{Float64}(Dm["ModelSD"][ij2k(i,j,3)]["coeffs"]), reshape(Vector{Float64}(Dm["ModelSD"][ij2k(i,j,3)]["mean"][1]),5,1)) for i = 1:1 for j = 1:3 ],
         [ TBModel(OffsiteBasis(paramset.rcutset[4][i,j],paramset.maxdegset[4][i,j],paramset.ordset[4][i,j],0,1).basis, Vector{Float64}(Dm["ModelPS"][ij2k(i,j,2)]["coeffs"]), [ Vector{Float64}.(Dm["ModelPS"][ij2k(i,j,2)]["mean"])[1] Vector{Float64}.(Dm["ModelPS"][ij2k(i,j,2)]["mean"])[2] Vector{Float64}.(Dm["ModelPS"][ij2k(i,j,2)]["mean"])[3] ]) for i = 1:3 for j = 1:2 ],
         [ TBModel(OffsiteBasis(paramset.rcutset[5][i,j],paramset.maxdegset[5][i,j],paramset.ordset[5][i,j],1,1).basis, Vector{Float64}(Dm["ModelPP"][ij2k(i,j,2)]["coeffs"]), [ Vector{Float64}.(Dm["ModelPP"][ij2k(i,j,2)]["mean"])[1] Vector{Float64}.(Dm["ModelPP"][ij2k(i,j,2)]["mean"])[2] Vector{Float64}.(Dm["ModelPP"][ij2k(i,j,2)]["mean"])[3] ]) for i = 1:2 for j = 1:2 ],
         [ TBModel(OffsiteBasis(paramset.rcutset[6][i,j],paramset.maxdegset[6][i,j],paramset.ordset[6][i,j],2,1).basis, Vector{Float64}(Dm["ModelPD"][ij2k(i,j,2)]["coeffs"]), [ Vector{Float64}.(Dm["ModelPD"][ij2k(i,j,2)]["mean"])[1] Vector{Float64}.(Dm["ModelPD"][ij2k(i,j,2)]["mean"])[2] Vector{Float64}.(Dm["ModelPD"][ij2k(i,j,2)]["mean"])[3] ]) for i = 1:1 for j = 1:2 ],
         [ TBModel(OffsiteBasis(paramset.rcutset[7][i,j],paramset.maxdegset[7][i,j],paramset.ordset[7][i,j],0,2).basis, Vector{Float64}(Dm["ModelDS"][ij2k(i,j,1)]["coeffs"]), [ Vector{Float64}.(Dm["ModelDS"][ij2k(i,j,1)]["mean"])[1] Vector{Float64}.(Dm["ModelDS"][ij2k(i,j,1)]["mean"])[2] Vector{Float64}.(Dm["ModelDS"][ij2k(i,j,1)]["mean"])[3] Vector{Float64}.(Dm["ModelDS"][ij2k(i,j,1)]["mean"])[4] Vector{Float64}.(Dm["ModelDS"][ij2k(i,j,1)]["mean"])[5]]) for i = 1:3 for j = 1:1 ],
         [ TBModel(OffsiteBasis(paramset.rcutset[8][i,j],paramset.maxdegset[8][i,j],paramset.ordset[8][i,j],1,2).basis, Vector{Float64}(Dm["ModelDP"][ij2k(i,j,1)]["coeffs"]), [ Vector{Float64}.(Dm["ModelDP"][ij2k(i,j,1)]["mean"])[1] Vector{Float64}.(Dm["ModelDP"][ij2k(i,j,1)]["mean"])[2] Vector{Float64}.(Dm["ModelDP"][ij2k(i,j,1)]["mean"])[3] Vector{Float64}.(Dm["ModelDP"][ij2k(i,j,1)]["mean"])[4] Vector{Float64}.(Dm["ModelDP"][ij2k(i,j,1)]["mean"])[5]]) for i = 1:2 for j = 1:1 ],
         [ TBModel(OffsiteBasis(paramset.rcutset[9][i,j],paramset.maxdegset[9][i,j],paramset.ordset[9][i,j],2,2).basis, Vector{Float64}(Dm["ModelDD"][ij2k(i,j,1)]["coeffs"]), [ Vector{Float64}.(Dm["ModelDD"][ij2k(i,j,1)]["mean"])[1] Vector{Float64}.(Dm["ModelDD"][ij2k(i,j,1)]["mean"])[2] Vector{Float64}.(Dm["ModelDD"][ij2k(i,j,1)]["mean"])[3] Vector{Float64}.(Dm["ModelDD"][ij2k(i,j,1)]["mean"])[4] Vector{Float64}.(Dm["ModelDD"][ij2k(i,j,1)]["mean"])[5]]) for i = 1:1 for j = 1:1 ] )
      elseif Dm["Type"] == "S"
         return OffModelWhole( data, paramset, Dm["Type"],
         [ TBModel(OffsiteBasis(paramset.rcutset[1][i,j],paramset.maxdegset[1][i,j],paramset.ordset[1][i,j],0,0,0,0).basis, Vector{Float64}(Dm["ModelSS"][ij2k(i,j,3)]["coeffs"]), reshape(Vector{Float64}(Dm["ModelSS"][ij2k(i,j,3)]["mean"][1]),1,1)) for i = 1:3 for j = 1:3 ],
         [ TBModel(OffsiteBasis(paramset.rcutset[2][i,j],paramset.maxdegset[2][i,j],paramset.ordset[2][i,j],1,0,0,0).basis, Vector{Float64}(Dm["ModelSP"][ij2k(i,j,3)]["coeffs"]), reshape(Vector{Float64}(Dm["ModelSP"][ij2k(i,j,3)]["mean"][1]),3,1)) for i = 1:2 for j = 1:3 ],
         [ TBModel(OffsiteBasis(paramset.rcutset[3][i,j],paramset.maxdegset[3][i,j],paramset.ordset[3][i,j],2,0,0,0).basis, Vector{Float64}(Dm["ModelSD"][ij2k(i,j,3)]["coeffs"]), reshape(Vector{Float64}(Dm["ModelSD"][ij2k(i,j,3)]["mean"][1]),5,1)) for i = 1:1 for j = 1:3 ],
         [ TBModel(OffsiteBasis(paramset.rcutset[4][i,j],paramset.maxdegset[4][i,j],paramset.ordset[4][i,j],0,1,0,0).basis, Vector{Float64}(Dm["ModelPS"][ij2k(i,j,2)]["coeffs"]), [ Vector{Float64}.(Dm["ModelPS"][ij2k(i,j,2)]["mean"])[1] Vector{Float64}.(Dm["ModelPS"][ij2k(i,j,2)]["mean"])[2] Vector{Float64}.(Dm["ModelPS"][ij2k(i,j,2)]["mean"])[3] ]) for i = 1:3 for j = 1:2 ],
         [ TBModel(OffsiteBasis(paramset.rcutset[5][i,j],paramset.maxdegset[5][i,j],paramset.ordset[5][i,j],1,1,0,0).basis, Vector{Float64}(Dm["ModelPP"][ij2k(i,j,2)]["coeffs"]), [ Vector{Float64}.(Dm["ModelPP"][ij2k(i,j,2)]["mean"])[1] Vector{Float64}.(Dm["ModelPP"][ij2k(i,j,2)]["mean"])[2] Vector{Float64}.(Dm["ModelPP"][ij2k(i,j,2)]["mean"])[3] ]) for i = 1:2 for j = 1:2 ],
         [ TBModel(OffsiteBasis(paramset.rcutset[6][i,j],paramset.maxdegset[6][i,j],paramset.ordset[6][i,j],2,1,0,0).basis, Vector{Float64}(Dm["ModelPD"][ij2k(i,j,2)]["coeffs"]), [ Vector{Float64}.(Dm["ModelPD"][ij2k(i,j,2)]["mean"])[1] Vector{Float64}.(Dm["ModelPD"][ij2k(i,j,2)]["mean"])[2] Vector{Float64}.(Dm["ModelPD"][ij2k(i,j,2)]["mean"])[3] ]) for i = 1:1 for j = 1:2 ],
         [ TBModel(OffsiteBasis(paramset.rcutset[7][i,j],paramset.maxdegset[7][i,j],paramset.ordset[7][i,j],0,2,0,0).basis, Vector{Float64}(Dm["ModelDS"][ij2k(i,j,1)]["coeffs"]), [ Vector{Float64}.(Dm["ModelDS"][ij2k(i,j,1)]["mean"])[1] Vector{Float64}.(Dm["ModelDS"][ij2k(i,j,1)]["mean"])[2] Vector{Float64}.(Dm["ModelDS"][ij2k(i,j,1)]["mean"])[3] Vector{Float64}.(Dm["ModelDS"][ij2k(i,j,1)]["mean"])[4] Vector{Float64}.(Dm["ModelDS"][ij2k(i,j,1)]["mean"])[5]]) for i = 1:3 for j = 1:1 ],
         [ TBModel(OffsiteBasis(paramset.rcutset[8][i,j],paramset.maxdegset[8][i,j],paramset.ordset[8][i,j],1,2,0,0).basis, Vector{Float64}(Dm["ModelDP"][ij2k(i,j,1)]["coeffs"]), [ Vector{Float64}.(Dm["ModelDP"][ij2k(i,j,1)]["mean"])[1] Vector{Float64}.(Dm["ModelDP"][ij2k(i,j,1)]["mean"])[2] Vector{Float64}.(Dm["ModelDP"][ij2k(i,j,1)]["mean"])[3] Vector{Float64}.(Dm["ModelDP"][ij2k(i,j,1)]["mean"])[4] Vector{Float64}.(Dm["ModelDP"][ij2k(i,j,1)]["mean"])[5]]) for i = 1:2 for j = 1:1 ],
         [ TBModel(OffsiteBasis(paramset.rcutset[9][i,j],paramset.maxdegset[9][i,j],paramset.ordset[9][i,j],2,2,0,0).basis, Vector{Float64}(Dm["ModelDD"][ij2k(i,j,1)]["coeffs"]), [ Vector{Float64}.(Dm["ModelDD"][ij2k(i,j,1)]["mean"])[1] Vector{Float64}.(Dm["ModelDD"][ij2k(i,j,1)]["mean"])[2] Vector{Float64}.(Dm["ModelDD"][ij2k(i,j,1)]["mean"])[3] Vector{Float64}.(Dm["ModelDD"][ij2k(i,j,1)]["mean"])[4] Vector{Float64}.(Dm["ModelDD"][ij2k(i,j,1)]["mean"])[5]]) for i = 1:1 for j = 1:1 ] )
      end
   else 
      error("Unexpected type $(typeof(idx[1]))")
   end
end

recover_error(Dm::Dict) = [vecvec2vecmat(Dm["RMSE_train"]), vecvec2vecmat(Dm["RMSE_test"])]

recover_at(D::Dict) = JuLIP.Atoms(; X = D["X"], Z = D["Z"], cell = D["cell"], pbc = D["pbc"])

end
