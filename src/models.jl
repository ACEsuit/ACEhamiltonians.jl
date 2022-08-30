module Models


using ACEhamiltonians, ACE, ACEbase
import ACEbase: read_dict, write_dict
using ACEhamiltonians.Parameters: OnSiteParaSet, OffSiteParaSet
using ACEhamiltonians.Bases: Basis, IsoBasis, AnisoBasis, is_fitted



export Model


# ╔═══════╗
# ║ Model ║
# ╚═══════╝

struct Model
    on_site_bases
    off_site_bases

    on_site_parameters::OnSiteParaSet
    off_site_parameters::OffSiteParaSet
    basis_definition

    function Model(on_site_bases, off_site_bases,
        on_site_parameters::OnSiteParaSet, off_site_parameters::OffSiteParaSet, basis_definition)
        new(on_site_bases, off_site_bases, on_site_parameters, off_site_parameters, basis_definition)
    end
    
    function Model(basis_definition::BasisDef, on_site_parameters::OnSiteParaSet,
                   off_site_parameters::OffSiteParaSet)
        # Developers Notes
        # This makes the assumption that all z₁-z₂-ℓ₁-ℓ₂ interactions are represented
        # by the same model.
        # Discuss use of the on/off_site_cache entities

        on_sites = Dict{NTuple{3, keytype(basis_definition)}, Basis}()
        off_sites = Dict{NTuple{4, keytype(basis_definition)}, Basis}()
        
        # Caching the basis functions of the functions is faster and allows ust to reuse
        # the same basis function for similar interactions.
        ace_basis_on = with_cache(on_site_ace_basis)
        ace_basis_off = with_cache(off_site_ace_basis)

        # Sorting the basis definition makes avoiding interaction doubling easier.
        # That is to say, we don't create models for both H-C and C-H interactions
        # as they represent the same thing.
        basis_definition_sorted = sort(collect(basis_definition), by=first)
        
        # Loop over all unique species pairs then over all combinations of their shells. 
        for (zₙ, (zᵢ, shellsᵢ)) in enumerate(basis_definition_sorted)
            for (zⱼ, shellsⱼ) in basis_definition_sorted[zₙ:end]
                homo_atomic = zᵢ == zⱼ
                for (n₁, ℓ₁) in enumerate(shellsᵢ), (n₂, ℓ₂) in enumerate(shellsⱼ)
                    homo_orbital = n₁ == n₂

                    # Skip symmetrically equivalent interactions. 
                    homo_atomic && n₁ > n₂ && continue
                    
                    if homo_atomic
                        id = (zᵢ, n₁, n₂)
                        ace_basis = ace_basis_on( # On-site bases
                            ℓ₁, ℓ₂, on_site_parameters[id]...)

                        on_sites[(zᵢ, n₁, n₂)] = Basis(ace_basis, id)
                    end

                    id = (zᵢ, zⱼ, n₁, n₂)
                    ace_basis = ace_basis_off( # Off-site bases
                        ℓ₁, ℓ₂, off_site_parameters[id]...)
                    
                    # Unless this is a homo-atomic homo-orbital interaction a double basis
                    # is needed.
                    if homo_atomic && homo_orbital
                        off_sites[(zᵢ, zⱼ, n₁, n₂)] = Basis(ace_basis, id)
                    else
                        ace_basis_i = ace_basis_off(
                            ℓ₂, ℓ₁, off_site_parameters[(zⱼ, zᵢ, n₂, n₁)]...)
                        off_sites[(zᵢ, zⱼ, n₁, n₂)] = Basis(ace_basis, ace_basis_i, id)
                    end
                end
            end
        end

    new(on_sites, off_sites, on_site_parameters, off_site_parameters, basis_definition)
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
    # which is common as `Basis` objects tend to share basis functions.


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
        if basis isa AnisoBasis
            add_basis(basis.basis_i)
        end
    end

    dict =  Dict(
        "__id__"=>"HModel",
        "on_site_bases"=>Dict(k=>write_dict(v, true) for (k, v) in m.on_site_bases),
        "off_site_bases"=>Dict(k=>write_dict(v, true) for (k, v) in m.off_site_bases),
        "on_site_parameters"=>write_dict(m.on_site_parameters),
        "off_site_parameters"=>write_dict(m.off_site_parameters),
        "basis_definition"=>Dict(k=>write_dict(v) for (k, v) in m.basis_definition),
        "bases_hashes"=>bases_hashes)
    
    return dict
end


function ACEbase.read_dict(::Val{:HModel}, dict::Dict)::Model

    function set_bases(target, basis_functions)
        for v in values(target)
            v["basis"] = basis_functions[v["basis"]]
            if v["__id__"] == "AnisoBasis"
                v["basis_i"] = basis_functions[v["basis_i"]]
            end
        end
    end

    # Replace basis object hashs with the appropriate object. 
    
    set_bases(dict["on_site_bases"], dict["bases_hashes"])
    set_bases(dict["off_site_bases"], dict["bases_hashes"])

    ensure_int(v) = v isa String ? parse(Int, v) : v
    
    return Model(
        Dict(parse_key(k)=>read_dict(v) for (k, v) in dict["on_site_bases"]),
        Dict(parse_key(k)=>read_dict(v) for (k, v) in dict["off_site_bases"]),
        read_dict(dict["on_site_parameters"]),
        read_dict(dict["off_site_parameters"]),
        Dict(ensure_int(k)=>read_dict(v) for (k, v) in dict["basis_definition"]))
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