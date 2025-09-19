module J2P

using Serialization
using Distributed
using JuLIP: Atoms
using LinearAlgebra
using Statistics
using PyCall

addprocs(28)
@everywhere begin
    using ACEhamiltonians
    import ACEhamiltonians: predict
end


function predict(atoms::Vector{PyObject}, model::Model) 
    atoms = atoms_p2j(atoms)
    images = cell_translations.(atoms, Ref(model))
    predicted = predict.(Ref(model), atoms, images)
    return predicted    
end


function atoms_p2j(atoms::Vector{PyObject})
    return [Atoms(Z=atom.get_atomic_numbers(), X=transpose(atom.positions), cell=collect(Float64.(I(3) * 100)), pbc=true) for atom in atoms]
end

end
