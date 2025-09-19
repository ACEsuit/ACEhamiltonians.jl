using Distributed
addprocs(28)
@everywhere begin
 using JuLIP: Atoms
 using LinearAlgebra
 using Statistics
 using PyCall
 using ACEhamiltonians
 import ACEhamiltonians: predict, Model
end

@everywhere function predictpp(atoms::Vector{PyObject}, model::Model)
    atoms = [Atoms(Z=atom.get_atomic_numbers(), X=transpose(atom.positions), cell=collect(Float64.(I(3) * 100)), pbc=true) for atom in atoms] 
    n = nworkers()
    len = length(atoms)
    chunk_size = ceil(Int, len / n)
    chunks_atoms = [atoms[(i-1)*chunk_size+1:min(i*chunk_size, len)] for i in 1:n]

    images = cell_translations.(atoms, Ref(model))
    chunks_images = [images[(i-1)*chunk_size+1:min(i*chunk_size, len)] for i in 1:n]
    
    predicted = Any[]
    for (chunk_atoms, chunk_images) in zip(chunks_atoms, chunks_images)
        task = @spawn predict.(Ref(model), chunk_atoms, chunk_images)
        push!(predicted, task)
    end
    predicted = fetch.(predicted)
    
    predicted = vcat(predicted...)
    return predicted    
end
