

# Warning this module is not release ready 
module Properties

using ACEhamiltonians, LinearAlgebra

export real_to_complex!, real_to_complex, band_structure, density_of_states, eigenvalue_confidence_interval


const _π2im = -2.0π * im

function eigvals_at_k(H::A, S::A, T, k_point; kws...) where A<:AbstractArray{<:AbstractFloat, 3}
    return real(eigvals(real_to_complex(H, T, k_point), real_to_complex(S, T, k_point); kws...))
end


phase(k::AbstractVector, T::AbstractVector) = exp(_π2im * (k ⋅ T))
phase(k::AbstractVector, T::AbstractMatrix) = exp.(_π2im * (k' * T))

function real_to_complex!(A_complex::AbstractMatrix{C}, A_real::AbstractArray{F, 3}, T::AbstractMatrix, k_point; sym=false) where {C<:Complex, F<:AbstractFloat}
    for i=1:size(T, 2)
        @views A_complex .+= A_real[:, :, i] .* phase(k_point, T[:, i])
    end
    if sym
        A_complex .= (A_complex + A_complex') * 0.5
    end
    nothing
end


"""
    real_to_complex(A_real, T, k_point[; sym=false])

Compute the complex matrix at a given k-point for a given real-space matrix.

# Arguments
 - `A_real`: real-space matrix of size N×N×T, where N is the number of atomic
    orbitals and T the number of cell translation vectors.
 - `T`: cell translation vector matrix of size 3×T.
 - `k_point`: the k-points for which the complex matrix should be returned.
 - `sym`: if true the resulting matrix will be symmetrised prior to its return.

# Returns
 - `A_complex`: the real space matrix evaluated at the requested k-point.

"""
function real_to_complex(A_real::AbstractArray{F, 3}, T, k_point::Vector; sym=false) where F<:AbstractFloat
    A_complex = zeros(Complex{F}, size(A_real, 2), size(A_real, 2))
    real_to_complex!(A_complex, A_real, T, k_point; sym=sym)
    return A_complex
end


function eigenvalue_confidence_interval(H, H̃, S, S̃, T, k_points, posterior=false)
    n = size(H, 1)
    C = complex(valtype(H))

    H_k = Matrix{C}(undef, n, n)
    S_k = Matrix{C}(undef, n, n)
    H̃_k = Matrix{C}(undef, n, n)
    S̃_k = Matrix{C}(undef, n, n)
    ΔH = Matrix{C}(undef, n, n)
    ΔS = Matrix{C}(undef, n, n)

    results = Matrix{valtype(H)}(undef, n, size(k_points, 2))

    for (i, k_point) in enumerate(eachcol(k_points))
        fill!(H_k, zero(C))
        fill!(S_k, zero(C))
        fill!(H̃_k, zero(C))
        fill!(S̃_k, zero(C))

        real_to_complex!(H_k, H, T, k_point)
        real_to_complex!(S_k, S, T, k_point)
        real_to_complex!(H̃_k, H̃, T, k_point)
        real_to_complex!(S̃_k, S̃, T, k_point)

        ΔH[:, :] = H̃_k - H_k
        ΔS[:, :] = S̃_k - S_k
        
        ϵ, φ = eigen!(H_k, S_k);

        ϵₜ = let
            if !posterior
                ϵ
            else
                eigen!(H̃_k, S̃_k).values()
            end
        end 

        for (j, (ϵᵢ, φᵢ)) in enumerate(zip(ϵₜ, eachcol(φ)))
   
            results[j, i] = real.(φᵢ' * ((ΔH - ϵᵢ * ΔS) * φᵢ))
        end

    end

    return results
end


function gaussian_broadening(E, ϵ, σ)
    return exp(-((E - ϵ) / σ)^2) / (sqrt(π) * σ)
end

# function gaussian_broadening(E, dE, ϵ, σ)
#     # Broadens in an identical manner to FHI-aims; not that this wil require
#     # SpecialFunctions.erf to work. While the results returned by this method
#     # match with FHI-aims the final DoS is off by a factor of 0.5 for some
#     # reason; so double counting is happening somewhere.
#     ga = erf((E - ϵ + (dE/2)) / (sqrt(2.0)σ))
#     gb = erf((E - ϵ - (dE/2)) / (sqrt(2.0)σ))
#     return (ga - gb) / 2dE
# end


"""
Density of states (k-point independent)
"""
function density_of_states(E::T, ϵ::T, σ) where T<:Vector{<:AbstractFloat}
    dos = T(undef, length(E))
    for i=1:length(E)
        dos[i] = sum(gaussian_broadening.(E[i], ϵ, σ))
    end
    return dos
end



"""
Density of states (k-point dependant)
"""
function density_of_states(E::V, ϵ::Matrix{F}, k_weights::V, σ; fermi=0.0) where V<:Vector{F} where F<:AbstractFloat
    # A non-zero fermi value indicates that the energies `E` are relative to the fermi
    # level. The most efficient and less user intrusive way to deal with this is create
    # and operate on an offset copy of `E`.  
    if fermi ≠ 0.0
        E = E .+ fermi
    end

    dos = zeros(F, length(E))
    let temp_array = zeros(F, size(ϵ)...)
        for i=1:length(E)
            temp_array .= gaussian_broadening.(E[i], ϵ, σ)
            temp_array .*= k_weights'
            dos[i] = sum(temp_array)
        end
    end
    
    return dos
end


function density_of_states(E::Vector, H::M, S::M, σ) where {M<:AbstractMatrix}
    return density_of_states(E, eigvals(H, S), σ)
end

function density_of_states(E::V, H::A, S::A, k_point::V, T, σ) where {V<:Vector{<:AbstractFloat}, A<:AbstractArray{<:AbstractFloat,3}}
    return density_of_states(E, eigvals_at_k(H, S, T, k_point), σ)
end

function density_of_states(E::Vector, H::A, S::A, k_points::AbstractMatrix, T, k_weights, σ; fermi=0.0) where {A<:AbstractArray{F, 3}} where F<:AbstractFloat
    return density_of_states(E, band_structure(H, S, T, k_points), k_weights, σ; fermi)
end




"""
    band_structure(H_real, S_real, T, k_points)

# Arguments
 - `H_real`: real space Hamiltonian matrix of size N×N×T, where N is the number of atomic
    orbitals and T the number of cell translation vectors.
 - `S_real`: real space overlap matrix of size N×N×T.
 - `T`: cell translation vector matrix of size 3×T.
 - `k_points`: a matrix specifying the k-points at which the eigenvalues should be evaluated.

# Returns
 - `eigenvalues`: eigenvalues evaluated along the specified k-point path. The columns of
    this matrix loop over k-points and rows over states.

"""
function band_structure(H_real::A, S_real::A, T, k_points) where A<:AbstractArray{F, 3} where F<:AbstractFloat

    C = Complex{F}
    nₒ, nₖ = size(H_real, 2), size(k_points, 2)
    
    # Final results array.
    ϵ = Matrix{F}(undef, nₒ, nₖ)

    # Construct the transient storage arrays
    let H_complex = Matrix{C}(undef, nₒ, nₒ), S_complex = Matrix{C}(undef, nₒ, nₒ)

        # Loop over each k-point
        for i=1:nₖ
            # Clear the transient storage arrays
            fill!(H_complex, zero(C))
            fill!(S_complex, zero(C))

            # Evaluate the Hamiltonian and overlap matrices at the iᵗʰ k-point
            real_to_complex!(H_complex, H_real, T, k_points[:, i])
            real_to_complex!(S_complex, S_real, T, k_points[:, i])
            
            # Calculate the eigenvalues
            ϵ[:, i] .= real(eigvals(H_complex, S_complex))
        end

    end
    
    return ϵ
end


end