using ACEhamiltonians
using BlockArrays, Serialization, StaticArrays
using LinearAlgebra: norm, diagind
using ACEbase: read_dict, load_json

using ACEhamiltonians.States: _guard_position, _inner_evaluate

# Todo:
#   - Remove hard coding.
#   - identify best place for occupancies to be specified.
#   - resolve unit mess.

_Bohr2Angstrom = 1/0.188972598857892E+01
_F64SV = SVector{3, Float64}
_FORCE_SHOW_ERROR = true


macro FSE(func)
    # Force show error
    if _FORCE_SHOW_ERROR
        func.args[2] = quote
            try
                $(func.args[2].args...)
            catch e
                println("Error encountered in Julia-DFTB+ API")
                for (exc, bt) in current_exceptions()
                    showerror(stdout, exc, bt)
                    println(stdout)
                end
                rethrow(e)
            end
        end
    end
    return func
end


_sub_block_sizes(species, basis_def) = 2basis_def[species] .+ 1

function _reshape_to_block(array, species_1, species_2, basis_definition)
    return PseudoBlockMatrix(
        reshape(
            array,
            number_of_orbitals(species_1, basis_definition),
            number_of_orbitals(species_2, basis_definition)
            ),
        _sub_block_sizes(species_1, basis_definition),
        _sub_block_sizes(species_2, basis_definition)
        )
end

# Setup related function

_s2n = Dict(
    "H"=>1, "He"=>2, "Li"=>3, "Be"=>4, "B"=>5, "C"=>6, "N"=>7, "O"=>8, "F"=>9,"Ne"=>10,
    "Na"=>11, "Mg"=>12, "Al"=>13, "Si"=>14, "P"=>15, "S"=>16, "Cl"=>17, "Ar"=>18,
    "K"=>19, "Ca"=>20, "Sc"=>21, "Ti"=>22, "V"=>23, "Cr"=>24, "Mn"=>25, "Fe"=>26,
    "Co"=>27, "Ni"=>28, "Cu"=>29, "Zn"=>30, "Ga"=>31, "Ge"=>32, "As"=>33, "Se"=>34,
    "Br"=>35, "Kr"=>36, "Rb"=>37, "Sr"=>38, "Y"=>39, "Zr"=>40, "Nb"=>41, "Mo"=>42,
    "Tc"=>43, "Ru"=>44, "Rh"=>45, "Pd"=>46, "Ag"=>47, "Cd"=>48, "In"=>49, "Sn"=>50,
    "Sb"=>51, "Te"=>52, "I"=>53, "Xe"=>54, "Cs"=>55, "Ba"=>56, "La"=>57, "Ce"=>58,
    "Pr"=>59, "Nd"=>60, "Pm"=>61, "Sm"=>62, "Eu"=>63, "Gd"=>64, "Tb"=>65, "Dy"=>66,
    "Ho"=>67, "Er"=>68, "Tm"=>69, "Yb"=>70, "Lu"=>71, "Hf"=>72, "Ta"=>73, "W"=>74,
    "Re"=>75, "Os"=>76, "Ir"=>77, "Pt"=>78, "Au"=>79, "Hg"=>80, "Tl"=>81, "Pb"=>82,
    "Bi"=>83, "Po"=>84, "At"=>85, "Rn"=>86, "Fr"=>87, "Ra"=>88, "Ac"=>89, "Th"=>90,
    "Pa"=>91, "U"=>92, "Np"=>93, "Pu"=>94, "Am"=>95, "Cm"=>96, "Bk"=>97, "Cf"=>98,
    "Es"=>99, "Fm"=>10, "Md"=>10, "No"=>10, "Lr"=>10, "Rf"=>10, "Db"=>10, "Sg"=>10,
    "Bh"=>10, "Hs"=>10, "Mt"=>10, "Ds"=>11, "Rg"=>111)


"""Return the unique id number associated with species of a given name."""
@FSE function species_name_to_id(name, model)
    # The current implementation of this function is only a temporary measure. 
    # Once atom names have been replaced with enums and the atomic number in "Basis.id"
    # is replaced with a species id then this function will be able interrogate the
    # model and its bases for information on what species it supports. Furthermore
    # this will allow for multiple species of the same atomic number to be used.
    return Int32(_s2n[name])

end

"""Returns true of the supplied model supports the provided species"""
@FSE function offers_species(name, model)
    if haskey(_s2n, name)
        return _s2n[name] ∈ Set([id[1] for id in keys(model.on_site_bases)])
    else
        return false
    end
end

@FSE function n_orbs_per_atom(species_id, model)
    return Int32(sum(2model.basis_definition[species_id].+1))
end

"""Maximum environmental cutoff distance"""
@FSE function max_environment_cutoff(model)
    max_on = maximum(values(model.on_site_parameters.e_cut_out))
    max_off = maximum(values(model.off_site_parameters.e_cut_out))
    distance = max(max_on, max_off)
    return distance / _Bohr2Angstrom
end


"""Maximum interaction cutoff distance"""
@FSE function max_interaction_cutoff(model)
    distance = maximum([env.r0cut for env in envelope.(values(model.off_site_bases))])
    return distance / _Bohr2Angstrom
end


@FSE function n_shells_on_species(species, model)
    return Int32(length(model.basis_definition[species]))
end

@FSE function shells_on_species!(array, species, model)
    shells = model.basis_definition[species]
    if length(array) ≠ length(shells)
        println("shells_on_species!: Provided array is of incorrect length.")
        throw(BoundsError("shells_on_species!: Provided array is of incorrect length."))
    end
    array[:] = shells
    nothing
end

@FSE function shell_occupancies!(array, species, model)
    ########### DEBUGGING ###########
    occupancies = Dict(
        1=>[1.0],
        6=>[1.0, 0.0],
        14=>[2.0, 2.0, 2.0, 6.0, 2.0, 0.0, 0.0])[species]
    #################################
    if length(array) ≠ length(occupancies)
        println("shell_occupancies!: Provided array is of incorrect length.")
        println("$(length(array)), $(length(occupancies))")
        throw(BoundsError("shell_occupancies!: Provided array is of incorrect length."))
    end
    array[:] = occupancies

end


@FSE function load_model(path::String)
    if endswith(lowercase(path), ".json")
        return read_dict(load_json(path))
    elseif endswith(lowercase(path), ".bin")
        return deserialize(path)
    else
        error("Unknown file extension used; only \"json\" & \"bin\" are supported")
    end
end


@FSE function _build_atom_state(coordinates, cutoff)
    # Build a list of static coordinate vectors, excluding the origin. 
    positions = map(_F64SV, eachcol(coordinates[:, 2:end]))

    # Exclude positions that lie outside of the cutoff allowed by the model.
    positions_in_range = positions[norm.(positions) .≤ cutoff]
    
    # Construct the associated state object vector
    return map(AtomState, positions_in_range)
end


function _build_bond_state(coordinates, envelope)
    # Build a list of static coordinate vectors, excluding the two bonding
    # atoms.
    positions = map(_F64SV, eachcol(coordinates[:, 3:end]))

    # The rest of this function copies code directly from `states.get_state`.
    # Here, rr0 is multiplied by two as vectors provided by DFTB+ point to
    # the midpoint of the bond. ACEhamiltonians expects the bond vector and
    # to be inverted, hence the second position is taken.
    rr0 = _F64SV(coordinates[:, 2] * 2)
    states = Vector{BondState{_F64SV, Bool}}(undef, length(positions) + 1)
    states[1] = BondState(_F64SV(coordinates[:, 2]), rr0, true)
    
    for k=1:length(positions)
        states[k+1] = BondState{_F64SV, Bool}(positions[k], rr0, false)
    end

    @views mask = _inner_evaluate.(Ref(envelope), states[2:end]) .!= 0.0
    @views n = sum(mask) + 1
    @views states[2:n] = states[2:end][mask]
    return states[1:n]
end


function build_on_site_atom_block!(block::Vector{Float64}, coordinates::Vector{Float64}, species, model)
    
    basis_def = model.basis_definition
    n_shells = length(basis_def[species[1]])

    # Unflatten the coordinates array
    coordinates = reshape(coordinates, 3, :) * _Bohr2Angstrom
    
    # Unflatten the atom-block array and convert it into a PseudoBlockMatrix
    block = _reshape_to_block(block, species[1], species[1], basis_def)


    # On-site atom block of the overlap matrix are just an identify matrix
    if model.label == "S"
        block .= 0.0
        block[diagind(block)] .= 1.0
        return nothing
    end
    
    # Loop over all shell pairs
    for i_shell=1:n_shells
        for j_shell=i_shell:n_shells

            # Pull out the associated sub-block as a view
            @views sub_block = block[Block(i_shell, j_shell)]

            # Select the appropriate model
            basis = model.on_site_bases[(species[1], i_shell, j_shell)]

            # Construct the on-site state taking into account the required cutoff
            state = _build_atom_state(coordinates, radial(basis).R.ru)
            
            # Make the prediction
            predict!(sub_block, basis, state)

            # if i_shell ≡ j_shell ≡ 7 && model.label ≡ "H"
            #     serialize("subblocks_on.dat", collect(sub_block))
            # end

            # Set the symmetrically equivalent block when appropriate
            if i_shell ≠ j_shell
                @views block[Block(j_shell, i_shell)] = sub_block'
            end
        end
    end

end


@FSE function build_off_site_atom_block!(block::Vector{Float64}, coordinates::Vector{Float64}, species, model)
    # Need to deal with situation where Z₁ > Z₂
    basis_def = model.basis_definition
    species_i, species_j = species[1:2]
    n_shells_i = number_of_shells(species_i, basis_def)
    n_shells_j = number_of_shells(species_j, basis_def)

    # Unflatten the coordinates array
    coordinates = reshape(coordinates, 3, :) * _Bohr2Angstrom
    
    # Unflatten the atom-block array and convert it into a PseudoBlockMatrix
    block = _reshape_to_block(block, species_i, species_j, basis_def)

    # By default only interactions where species-i ≥ species-j are defined as
    # adding interactions for species-i < species-j would be redundant.
    if species_i > species_j
        block = block'
        n_shells_i, n_shells_j = n_shells_j, n_shells_i
        reflect_state = true
    else
        reflect_state = false
    end

    # Loop over all shell pairs
    for i_shell=1:n_shells_i
        for j_shell=1:n_shells_j
            
            # Skip over i_shell > j_shell homo-atomic interactions
            species_i ≡ species_j && i_shell > j_shell && continue
            
            # Pull out the associated sub-block as a view
            @views sub_block = block[Block(i_shell, j_shell)]

            # Select the appropriate model
            basis = model.off_site_bases[(species[1], species[2], i_shell, j_shell)]

            # Construct the on-site state taking into account the required cutoff
            state = _build_bond_state(coordinates, envelope(basis))
            
            if reflect_state
                state = reflect.(state)
            end

            # Make the prediction
            predict!(sub_block, basis, state)

            # if i_shell ≡ 5 && j_shell ≡ 6 && model.label ≡ "H"                    
            #     _dump("subblocks_off.dat", collect(sub_block))
            #     _dump("bond_vectors.dat", Vector(state[1].rr0))
            #     _dump("states.dat", Vector(state))
            #     _dump("coordinates.dat", coordinates)
            # end

            if species_i ≡ species_j
                @views predict!(block[Block(j_shell, i_shell)]', basis, reflect.(state))
            end

        end
    end

end


function _dump(file_name, data)
    file_data = isfile(file_name) ? deserialize(file_name) : Any[]
    push!(file_data, data)
    serialize(file_name, file_data)
end

coordinates = [0.5, 0, 0, -0.5, 0, 0, 0, 0.5, 0]
block = zeros(17 * 17)
species = [14, 14, 14]
model = deserialize("Working/H_model_Si_003_fitted.bin")
build_off_site_atom_block!(block, coordinates, species, model)

s = [1, 1, 1, 3, 3, 3, 5]
ref = PseudoBlockArray(reshape(block, 17, 17), s, s)

println("Finished!")
