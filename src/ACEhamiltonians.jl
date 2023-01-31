module ACEhamiltonians

using JuLIP, JSON, HDF5, Reexport

using ACE.SphericalHarmonics: SphericalCoords
import ACE.SphericalHarmonics: cart2spher

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


# function cart2spher(r⃗::AbstractVector{T}) where T
#     @assert length(r⃗) == 3
#     # When the length of the vector `r⃗` is zero then the signs can have a
#     # destabalising effect on the results; i.e.
#     #   cart2spher([0., 0., 0.]) ≠ cart2spher([0. 0., -0.])
#     # Hense the following catch:
#     φ = atan(r⃗[2], r⃗[1])
#     θ = atan(hypot(r⃗[1], r⃗[2]), r⃗[3])
#     sinφ, cosφ = sincos(φ)
#     sinθ, cosθ = sincos(θ)
#     return SphericalCoords{T}(norm(r⃗), cosφ, sinφ, cosθ, sinθ)
# end

function cart2spher(r⃗::AbstractVector{T}) where T
    @assert length(r⃗) == 3
    # When the length of the vector `r⃗` is zero then the signs can have a
    # destabalising effect on the results; i.e.
    #   cart2spher([0., 0., 0.]) ≠ cart2spher([0. 0., -0.])
    # Hense the following catch:
    if norm(r⃗) ≠ 0.0
        φ = atan(r⃗[2], r⃗[1])
        θ = atan(hypot(r⃗[1], r⃗[2]), r⃗[3])
        sinφ, cosφ = sincos(φ)
        sinθ, cosθ = sincos(θ)
        return SphericalCoords{T}(norm(r⃗), cosφ, sinφ, cosθ, sinθ)
    else
        return SphericalCoords{T}(0.0, 1.0, 0.0, 1.0, 0.0)
    end
end


include("common.jl")
@reexport using ACEhamiltonians.Common

include("io.jl")
@reexport using ACEhamiltonians.DatabaseIO

include("parameters.jl")
@reexport using ACEhamiltonians.Parameters

include("data.jl")
@reexport using ACEhamiltonians.MatrixManipulation

include("states.jl")
@reexport using ACEhamiltonians.States

include("basis.jl")
@reexport using ACEhamiltonians.Bases

include("models.jl")
@reexport using ACEhamiltonians.Models

include("datastructs.jl")
@reexport using ACEhamiltonians.DataSets

include("struc_setting.jl")
@reexport using ACEhamiltonians.Structure

include("dataproc.jl")
@reexport using ACEhamiltonians.DataProcess

include("fit.jl")
@reexport using ACEhamiltonians.Fitting

include("fitting.jl")
@reexport using ACEhamiltonians.Fitting2

include("predicting.jl")
@reexport using ACEhamiltonians.Predicting

include("predict.jl")
@reexport using ACEhamiltonians.Predict

include("dictionary.jl")
@reexport using ACEhamiltonians.Dictionary

include("properties.jl")
@reexport using ACEhamiltonians.Properties

include("tools.jl")

include("api/dftbp_api.jl")

end
