using ACEhamiltonians, ACE, JSON, JuLIP, HDF5
using ArgParse

using ACEhamiltonians: Data, Params, write_modeljson, fname2index
using ACEhamiltonians.Predict: predict_main

args = ArgParseSettings()
@add_arg_table args begin
    "--name", "-N"
        help = "Name of data files to include in fit"
        arg_type = String
        nargs = '+'
        default = ["data/FCC-MD-500K/SK-supercell-001.h5", 
                   "data/FCC-MD-500K/SK-supercell-002.h5"]
    "--index", "-i"
        help = "Atoms from each file to include. Default is 1:20:729, i.e. every 20th atom"
        arg_type = Vector{Tuple{Int64,Int64}}
        default = vec([ k for k in Iterators.product(Vector(1:20:729),Vector(1:20:729).+1)])
        eval_arg = true
    "--indextest", "-t"
        help = "Atoms from each file to include. Default is 1:20:729.+1, i.e. every 20th atom"
        arg_type = Vector{Tuple{Int64,Int64}}
        default = vec([ k for k in Iterators.product(Vector(1:20:729).+1,Vector(1:20:729).+2)])
        eval_arg = true
    "--rcut", "-r"
        help = """Cutoff radii for Hamiltonian and overlap blocks. 
                  Either 1, 2 or 12 arguments required. If a single
                  cutoff is given it is used for all blocks. If
                  two cutoffs are given, the first is for H and the 
                  second for S. If 12 are present they must be in the order:
                  H_SS, H_SP, H_SD, H_PP, H_PD, H_DD,
                  S_SS, S_SP, S_SD, S_PP, S_PD, S_DD."""
        nargs = '*'
        arg_type = Float64
    "--degree", "-d"
        help = """Maximum polynomial degree for the ACE basis set.
                  As for --rcut, 1, 2 or 12 arguments can be supplied.
                  If one argument is supplied """
        nargs = '*'
        arg_type = Int64
        default = [6]
    "--order", "-o"
        help = """Body order for the ACE basis set. 
                  As for --rcut, 1, 2 or 12 arguments can be supplied."""
        nargs = '*'
        arg_type = Int64
        default = [2]
    "--reg", "-λ"
        help = """Regularisation applied to linear system. As for --rcut,
                  1, 2 or 12 arguments can be supplied"""
        nargs = '*'
        arg_type = Float64
        default = [1e-7]
    "--regtype", "-R"
        help = "Type of regularisation to apply when solving linear system."
        arg_type = Int64
        default = 2
end

function expand_arg(key::String, value::AbstractVector)
    if length(value) == 1
        if key == "degree"
            return repeat([value[1], value[1]+2], 1, 9)
        elseif key == "order"
            return repeat([value[1], 1], 1, 9)
        else    
            return repeat(value, 2, 9)
        end
    elseif length(value) == 2
        return repeat(value, 1, 9)
    elseif length(value) == 12
        return reshape(value, 2, 9)
    else
        error("unexpected length(value) of $(length(value))")
    end
end

parsed_args = parse_args(ARGS, args)

for (key, value) in parsed_args
    if key ∈ ["rcut", "degree", "order", "reg"]
        parsed_args[key] = expand_arg(key, value)
    end
end

data = Data(parsed_args["name"], fname2index.(parsed_args["name"],1000))
# @show data

data_test = Data(parsed_args["name"], parsed_args["indextest"])
# @show data_test

H_params = Params(parsed_args["rcut"][1,:],
                  parsed_args["degree"][1,:],
                  parsed_args["order"][1,:],
                  parsed_args["reg"][1,:],
                  parsed_args["regtype"],
                  "LSQR")
@show H_params

S_params = Params(parsed_args["rcut"][2,:],
                  parsed_args["degree"][2,:],
                  parsed_args["order"][2,:],
                  parsed_args["reg"][2,:],
                  parsed_args["regtype"],
                  "LSQR")
@show S_params

println("Performing fit...")

# Step 2. Models training, generating corresponding errors
OMW_H, OMW_S, errset = predict_main(data, data_test, H_params, S_params)

@show errset
println("Saving model...")

# Step 3. Save the models/errors/parameters/etc. in a json file
write_modeljson(OMW_H, OMW_S, errset)

println("Done.")

