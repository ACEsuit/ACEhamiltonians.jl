using JuLIP: load_json
using ACEhamiltonians.Dictionary: recover_error
using JSON

modelfile = ARGS[1]

@show modelfile
H_data = load_json(modelfile)

errors = Dict()

errors["H_params"] = H_data["params"]
errors["H_error"] = recover_error(H_data)

S_file = replace(modelfile, "_H" => "_S")

if isfile(S_file)
    S_data = load_json(S_file)
    errors["S_params"] = S_data["params"]
    errors["S_error"] = recover_error(S_data)
end

outfile = replace(modelfile, "_H.json" => "_errors.json")
@show outfile

open(outfile, "w") do fd
    write(fd, JSON.json(errors))
end
