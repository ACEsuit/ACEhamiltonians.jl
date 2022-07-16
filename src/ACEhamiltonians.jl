module ACEhamiltonians

using JuLIP, JSON, HDF5, Reexport


export BasisDef

"""
    BasisDef(atomic_number => [ℓ₁, ..., ℓᵢ], ...)

Provides information about the basis set by specifying the azimuthal quantum numbers (ℓ)
of each shell on each species. Dictionary is keyed by atomic numbers & valued by vectors
of ℓs i.e. `Dict{atomic_number, [ℓ₁, ..., ℓᵢ]}`. 

A minimal basis set for hydrocarbon systems would be `BasisDef(1=>[0], 6=>[0, 0, 1])`.
This declares hydrogen atoms as having only a single s-shell and carbon atoms as having
two s-shells and one p-shell.
"""
BasisDef = Dict{I, Vector{I}} where I<:Integer


include("io.jl")
@reexport using ACEhamiltonians.DatabaseIO

include("parameters.jl")
@reexport using ACEhamiltonians.Parameters

include("data.jl")
@reexport using ACEhamiltonians.MatrixManipulation

include("states.jl")
@reexport using ACEhamiltonians.States

include("struc_setting.jl")
@reexport using ACEhamiltonians.Structure

include("dataproc.jl")
@reexport using ACEhamiltonians.DataProcess

include("fit.jl")
@reexport using ACEhamiltonians.Fitting

include("predict.jl")
@reexport using ACEhamiltonians.Predict

include("dictionary.jl")
@reexport using ACEhamiltonians.Dictionary

include("tools.jl")

end
