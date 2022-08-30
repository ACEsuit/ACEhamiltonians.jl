using PyCall, Serialization, Printf
using JuLIP; Atoms
using ACEbase: read_dict, load_json
using ACEhamiltonians.Bases: Model
using ACEhamiltonians.Predicting: predict, cell_translations

ase = pyimport("ase")
pyimport("ase.io")

############################
# General IO Functionality #
############################

format_string(i::AbstractFloat) = @sprintf("%15.8E", i)
format_string(i::Integer) = @sprintf("%5i", i)

function write_array(io::String, array::AbstractArray)
    open(io, "w") do io
        write_array(io, array)
    end
    nothing
end

function write_array(io::IO, array::AbstractArray{<:Any, 3})
    write(io, "SHAPE:", join(size(array), "×"))
    for (n_cell, cell) in enumerate(eachslice(array, dims=3))
        write(io, "\n", "CELL:$n_cell\n", join(join.(eachrow(format_string.(cell)), " "), "\n"))
    end
    nothing
end

function write_array(io::IO, array::AbstractArray{<:Any, 2})
    write(io, "SHAPE:", join(size(array), "×"))
    write(io, "\n", join(join.(eachrow(format_string.(array)), " "), "\n"))
    nothing
end

function write_basis_info(io::String, basis_def)
    open(io, "w") do io
        write_basis_info(io, basis_def)
    end
    nothing
end

function write_basis_info(io::IO, basis_def)
    # Write out the header
    @printf(io, "%5s %3s %3s %3s", "z", "i", "l", "m")
    # Loop over species
    for (z, shells) in sort(basis_def)
        # Then over shell numbers and their azimuthal quantum number
        for (n, l) in enumerate(shells)
            # Then over the various magnetic quantum numbers
            for m in -l:l
                # Then write out the "orbital" information
                @printf(io, "\n%5i %3i %3i %3i", z, n, l, m)
            end
        end
    end
    nothing
end


function _load_model(path::String)
    if endswith(lowercase(path), ".json")
        return read_dict(load_json(path))
    elseif endswith(lowercase(path), ".bin")
        return deserialize(path)
    else
        error("Unknown file extension used; only \"json\" & \"bin\" are supported")
    end
end


##################
# Main Code Loop #
##################
function main(geometry_path, H_model_path, S_model_path)
    # Results are placed into the directory where the geometry file is located.
    results_dir = dirname(geometry_path)
    H_rsm_output_path = joinpath(results_dir, "H_rsm.out")
    S_rsm_output_path = joinpath(results_dir, "S_rsm.out")
    cell_indices_output_path = joinpath(results_dir, "cell_indices.out")
    basis_output_path = joinpath(results_dir, "basis_indices.out")
    
    # Ensure any residue results files are purged before the calculation starts
    # this helps to guard against accidentally reading the results from a past
    # calculation if the current one fails.
    for path in [H_rsm_output_path, S_rsm_output_path, cell_indices_output_path]
        rm(path, force=true)
    end
    
    # Load Hamiltonian and overlap models from the supplied files
    H_model, S_model = _load_model(H_model_path), _load_model(S_model_path)
    # Repeat the process for the geometry file
    atoms = let ase_atoms = ase.io.read(geometry_path)
        JuLIP.Atoms(
            ;Z=ase_atoms.get_atomic_numbers(), X=ase_atoms.positions',
            cell=ase_atoms.cell.array, pbc=ase_atoms.pbc)
    end

    # Select an appropriate cut of distance
    cutoff = max(
        maximum(values(H_model.off_site_parameters.b_cut)),
        maximum(values(S_model.off_site_parameters.b_cut))
    )

    # Identify all images that lie within the selected cutoff distance 
    cell_indices = cell_translations(atoms, cutoff)

    # Predict the Hamiltonian and overlap matrices
    H = predict(H_model, atoms, cell_indices)
    S = predict(S_model, atoms, cell_indices; oai=true)

    # Save the contents of the reals space Hamiltonian and overlap matrices along
    # with the cell indices into appropriately named data files. 
    write_array(H_rsm_output_path, H)
    write_array(S_rsm_output_path, S)
    write_array(cell_indices_output_path, cell_indices)
    
    # Write out the basis set definition
    let zs = getfield.(atoms.Z, :z)
        basis_def = filter(i->i[1]∈zs, H_model.basis_definition)
        write_basis_info(basis_output_path, basis_def)
    end

    nothing
end


main(ARGS[1], ARGS[2], ARGS[3])
