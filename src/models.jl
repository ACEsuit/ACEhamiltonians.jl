module Models

using ACEhamiltonians, ACE, ACEbase
import ACEbase: read_dict, write_dict
using ACEhamiltonians.Parameters: OnSiteParaSet, OffSiteParaSet
using ACEhamiltonians.Bases: AHBasis, is_fitted
using ACEhamiltonians: DUAL_BASIS_MODEL


export Model


# ╔═══════╗
# ║ Model ║
# ╚═══════╝


# Todo:
#   - On-site and off-site components should be optional.
#   - Document
#   - Clean up 
struct Model

    on_site_bases
    off_site_bases
    on_site_parameters
    off_site_parameters
    basis_definition

    label::String

    meta_data::Dict{String, Any}

    function Model(
        on_site_bases, off_site_bases, on_site_parameters::OnSiteParaSet,
        off_site_parameters::OffSiteParaSet, basis_definition, label::String,
        meta_data::Union{Dict, Nothing}=nothing)
        
        # If no meta-data is supplied then just default to a blank dictionary
        meta_data = isnothing(meta_data) ? Dict{String, Any}() : meta_data  
        
        new(on_site_bases, off_site_bases, on_site_parameters, off_site_parameters,
            basis_definition, label, meta_data)
    end
    
    function Model(
        basis_definition::BasisDef, on_site_parameters::OnSiteParaSet,
        off_site_parameters::OffSiteParaSet, label::String,
        meta_data::Union{Dict, Nothing}=nothing)

        # Developers Notes
        # This makes the assumption that all z₁-z₂-ℓ₁-ℓ₂ interactions are represented
        # by the same model.
        # Discuss use of the on/off_site_cache entities

        on_sites = Dict{NTuple{3, keytype(basis_definition)}, AHBasis}()
        off_sites = Dict{NTuple{4, keytype(basis_definition)}, AHBasis}()
        
        # Caching the basis functions of the functions is faster and allows ust to reuse
        # the same basis function for similar interactions.
        ace_basis_on = with_cache(on_site_ace_basis)
        ace_basis_off = with_cache(off_site_ace_basis)

        # Sorting the basis definition makes avoiding interaction doubling easier.
        # That is to say, we don't create models for both H-C and C-H interactions
        # as they represent the same thing.
        basis_definition_sorted = sort(collect(basis_definition), by=first)
        
        @debug "Building model"
        # Loop over all unique species pairs then over all combinations of their shells. 
        for (zₙ, (zᵢ, shellsᵢ)) in enumerate(basis_definition_sorted)
            for (zⱼ, shellsⱼ) in basis_definition_sorted[zₙ:end]
                homo_atomic = zᵢ == zⱼ
                for (n₁, ℓ₁) in enumerate(shellsᵢ), (n₂, ℓ₂) in enumerate(shellsⱼ)

                    # Skip symmetrically equivalent interactions. 
                    homo_atomic && n₁ > n₂ && continue
                    
                    if homo_atomic
                        id = (zᵢ, n₁, n₂)
                        @debug "Building on-site model : $id"
                        ace_basis = ace_basis_on( # On-site bases
                            ℓ₁, ℓ₂, on_site_parameters[id]...)
                        #TODO: add species to the above line

                        on_sites[(zᵢ, n₁, n₂)] = AHBasis(ace_basis, id)
                    end

                    id = (zᵢ, zⱼ, n₁, n₂)
                    @debug "Building off-site model: $id"

                    ace_basis = ace_basis_off( # Off-site bases
                        ℓ₁, ℓ₂, off_site_parameters[id]...)
                    #TODO: add species to the above line
                    
                    @static if DUAL_BASIS_MODEL
                        if homo_atomic && n₁ == n₂
                            off_sites[(zᵢ, zⱼ, n₁, n₂)] = AHBasis(ace_basis, id)
                        else
                            ace_basis_i = ace_basis_off(
                                ℓ₂, ℓ₁, off_site_parameters[(zⱼ, zᵢ, n₂, n₁)]...)
                            off_sites[(zᵢ, zⱼ, n₁, n₂)] = AHBasis(ace_basis, ace_basis_i, id)
                        end
                    else
                        off_sites[(zᵢ, zⱼ, n₁, n₂)] = AHBasis(ace_basis, id)
                    end     
                end
            end
        end

    # If no meta-data is supplied then just default to a blank dictionary
    meta_data = isnothing(meta_data) ? Dict{String, Any}() : meta_data  
    new(on_sites, off_sites, on_site_parameters, off_site_parameters, basis_definition, label, meta_data)
    end

end

# Associated methods

Base.:(==)(x::Model, y::Model) = (
    x.on_site_bases == y.on_site_bases && x.off_site_bases == y.off_site_bases
    && x.on_site_parameters == y.on_site_parameters && x.off_site_parameters == y.off_site_parameters)


# ╭───────┬──────────────────╮
# │ Model │ IO Functionality │
# ╰───────┴──────────────────╯

function ACEbase.write_dict(m::Model)
    # ACE bases are stored as hash values which are checked against the "bases_hashes"
    # dictionary during reading. This avoids saving multiple copies of the same object;
    # which is common as `AHBasis` objects tend to share basis functions.


    bases_hashes = Dict{String, Any}()

    function add_basis(basis)
        # Store the hash/basis pair in the bases_hashes dictionary. As the `write_dict`
        # method can be quite costly to evaluate it is best to only call it when strictly
        # necessary; hence this function exists.
        basis_hash = string(hash(basis))
        if !haskey(bases_hashes, basis_hash)
            bases_hashes[basis_hash] = write_dict(basis)
        end
    end

    for basis in union(values(m.on_site_bases), values(m.off_site_bases))        
        add_basis(basis.basis)
    end

    # Serialise the meta-data
    meta_data = Dict{String, Any}(
        # Invoke the `read_dict` method on values as and where appropriate
        k => hasmethod(write_dict, (typeof(v),)) ? write_dict(v) : v
        for (k, v) in m.meta_data
    )

    dict =  Dict(
        "__id__"=>"HModel",
        "on_site_bases"=>Dict(k=>write_dict(v, true) for (k, v) in m.on_site_bases),
        "off_site_bases"=>Dict(k=>write_dict(v, true) for (k, v) in m.off_site_bases),
        "on_site_parameters"=>write_dict(m.on_site_parameters),
        "off_site_parameters"=>write_dict(m.off_site_parameters),
        "basis_definition"=>Dict(k=>write_dict(v) for (k, v) in m.basis_definition),
        "bases_hashes"=>bases_hashes,
        "label"=>m.label,
        "meta_data"=>meta_data)
    
    return dict
end


function ACEbase.read_dict(::Val{:HModel}, dict::Dict)::Model

    function set_bases(target, basis_functions)
        for v in values(target)
            v["basis"] = basis_functions[v["basis"]]
        end
    end

    # Replace basis object hashs with the appropriate object. 
    set_bases(dict["on_site_bases"], dict["bases_hashes"])
    set_bases(dict["off_site_bases"], dict["bases_hashes"])

    ensure_int(v) = v isa String ? parse(Int, v) : v
    
    # Parse meta-data
    if haskey(dict, "meta_data")
        meta_data = Dict{String, Any}()
        for (k, v) in dict["meta_data"]
            if typeof(v) <: Dict && haskey(v, "__id__")
                meta_data[k] = read_dict(v)
            else
                meta_data[k] = v
            end
        end
    else
        meta_data = nothing
    end

    # One of the important entries present in the meta-data dictionary is the `occupancy`
    # data. This should be keyed by integers; however the serialisation/de-serialisation
    # process converts this into a string. A hard-coded fix is implemented here, but it
    # would be better to create a more general way of handling this later on.
    if !isnothing(meta_data) && haskey(meta_data, "occupancy") && (keytype(meta_data["occupancy"]) ≡ String)
        meta_data["occupancy"] = Dict(parse(Int, k)=>v for (k, v) in meta_data["occupancy"])
    end

    return Model(
        Dict(parse_key(k)=>read_dict(v) for (k, v) in dict["on_site_bases"]),
        Dict(parse_key(k)=>read_dict(v) for (k, v) in dict["off_site_bases"]),
        read_dict(dict["on_site_parameters"]),
        read_dict(dict["off_site_parameters"]),
        Dict(ensure_int(k)=>read_dict(v) for (k, v) in dict["basis_definition"]),
        dict["label"],
        meta_data)
end


# Todo: this is mostly to stop terminal spam and should be updated
# with more meaningful information later on.
function Base.show(io::IO, model::Model)

    # Work out if the on/off site bases are fully, partially or un-fitted.
    f = b -> if all(b) "no" elseif all(!, b) "yes" else "partially" end
    on = f([!is_fitted(i) for i in values(model.on_site_bases)])
    off = f([!is_fitted(i) for i in values(model.off_site_bases)])
    
    # Identify the species present
    species = join(sort(unique(getindex.(collect(keys(model.on_site_bases)), 1))), ", ", " & ")

    print(io, "Model(fitted=(on: $on, off: $off), species: ($species))")
end


end
