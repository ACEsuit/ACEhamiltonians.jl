using LinearAlgebra
using Printf
using BlockArrays
using JSON
using HDF5
using LaTeXStrings
using JuLIP: load_json
using ACEhamiltonians.Dictionary: recover_model, recover_at
using Roots

const hartree2ev = 27.211386024367243

struct KPath
    phase::String
    kpoints
    special_points
end

const DATA_DIR = joinpath(@__DIR__, "../data/")

function KPath(phase)
    @assert phase in [:FCC, :BCC]
    phase = String(phase)    
    bandpath = JSON.parsefile(joinpath(DATA_DIR, "$(phase)_tb_bandpath.json"))
    kpoints = bandpath["kpoints"]
    special_points = bandpath["special_points"]
    return KPath(phase, kpoints, special_points)
end

function read_HkSk(filename, N_orb)
    data = JSON.parsefile(filename)

    k = zeros(length(data), 3)
    Hk = zeros(ComplexF64, length(data), N_orb, N_orb)
    Sk = zeros(ComplexF64, length(data), N_orb, N_orb)
    bands = zeros(length(data), N_orb)

    for (i, row) in enumerate(data)
        k[i, :] = row["k"]
        Hk[i, :, :] = Hermitian(complex.(hcat(row["Re_H_k"]...), hcat(row["Im_H_k"]...)))
        Sk[i, :, :] = Hermitian(complex.(hcat(row["Re_S_k"]...), hcat(row["Im_S_k"]...)))

        bands[i, :] = real.(eigvals(Hk[i, :, :], Sk[i, :, :]))
    end

    return k, Hk, Sk, bands
end

function tb_cells(kmesh)
    Nx, Ny, Nz = kmesh
    R = CartesianIndices((-(Nx÷2):(Nx÷2), 
                          -(Ny÷2):(Ny÷2),
                          -(Nz÷2):(Nz÷2)))
    return getindex.(reshape(R, :), [1 2 3])
end

function full_to_block(H, S, N_orb)
    natoms = Int(size(H, 1)/N_orb)
    Hb = BlockArray(H, repeat([N_orb], natoms), repeat([N_orb], natoms))
    Sb = BlockArray(S, repeat([N_orb], natoms), repeat([N_orb], natoms))
    return Hb, Sb
end

function split_HS(Hb, Sb, R)
    N_orb = blocksizes(Hb)[1][1]
    N_blocks = blocksize(Hb,1)
    H_NMM = zeros(N_blocks, N_orb, N_orb)
    S_NMM = zeros(N_blocks, N_orb, N_orb)
    ia = findfirst(all(R .== [0 0 0], dims=2))[1]
    blocks = Tuple{Int64,Int64}[]
    for ja = 1:N_blocks
        push!(blocks, (ia, ja))
        H_NMM[ja, :, :] = view(Hb, Block(ia, ja))
        S_NMM[ja, :, :] = view(Sb, Block(ia, ja))
    end
    return H_NMM, S_NMM, blocks
end

# copied from DFTK.jl -- could add this to dependencies, or (better) use Brilloiun.jl instead

"""Bring ``k``-point coordinates into the range [-0.5, 0.5)"""
function normalize_kpoint_coordinate(x::Real)
    x = x - round(Int, x, RoundNearestTiesUp)
    @assert -0.5 ≤ x < 0.5
    x
end
normalize_kpoint_coordinate(k::AbstractVector) = normalize_kpoint_coordinate.(k)

using StaticArrays
const Vec3{T} = SVector{3, T} where T

"""Construct the coordinates of the ``k``-points in a (shifted) Monkorst-Pack grid"""
function kgrid_monkhorst_pack(kgrid_size; kshift=[0, 0, 0])
    kgrid_size = Vec3{Int}(kgrid_size)
    start = -floor.(Int, (kgrid_size .- 1) ./ 2)
    stop  = ceil.(Int, (kgrid_size .- 1) ./ 2)
    kshift = Vec3{Rational{Int}}(kshift)
    kcoords = [(kshift .+ Vec3([i, j, k])) .// kgrid_size
               for i=start[1]:stop[1], j=start[2]:stop[2], k=start[3]:stop[3]]
    vec(normalize_kpoint_coordinate.(kcoords))
end

function real_to_bloch(kpoints, H_NMM, S_NMM, R)
    N_orb = size(H_NMM, 2)
    N_k = length(kpoints)
    H_kMM = zeros(ComplexF64, N_k, N_orb, N_orb)
    S_kMM = zeros(ComplexF64, N_k, N_orb, N_orb)
    for i = 1:N_k
        phase_N = exp.(-2im * π * (R * kpoints[i]))
        H_kMM[i, :, :] = Hermitian(dropdims(sum(phase_N .* H_NMM, dims=1), dims=1))
        S_kMM[i, :, :] = Hermitian(dropdims(sum(phase_N .* S_NMM, dims=1), dims=1))
    end
    return H_kMM, S_kMM
end

function eigensolve(H, S; reg=0)
    S = copy(S) + reg * I
    return eigvals(H, S)
end

function eigvals_error(ϵ, H, H0, S, S0)
    # Error estimate from λᵢ - λ⁰ᵢ = <x⁰ᵢ | dH - λ⁰ᵢ dS | x⁰ᵢ >
    ϵ0, ψ0 = eigen(H0, S0)
    dH = H - H0
    dS = S - S0
    return [ real.(ψ0[:, j]' * ((dH - ϵ[j] * dS) * ψ0[:, j])) for j=1:length(ϵ0) ]
end

function kspace_eigensolve(kpoints, H_NMM, S_NMM, R; reg=0, return_HS=false)
    H_kMM, S_kMM = real_to_bloch(kpoints, H_NMM, S_NMM, R)
    N_k = length(kpoints)
    N_orb = size(H_NMM, 2)
    ϵ_kM = zeros(N_k, N_orb)
    for k = 1:N_k
        ϵ_kM[k, :] = real.(eigensolve(H_kMM[k,:,:], S_kMM[k,:,:], reg=reg))
    end
    if return_HS
        return ϵ_kM, H_kMM, S_kMM
    else
        return ϵ_kM
    end
end

function bandstructure(kpath::KPath, H_NMM, S_NMM, R; H0_NMM=nothing, S0_NMM=nothing, reg=0)
    N_k, N_orb = length(kpath.kpoints), size(H_NMM, 2)
    ϵ_kM, H_kMM, S_kMM = kspace_eigensolve(kpath.kpoints, H_NMM, S_NMM, R; reg=reg, return_HS=true)
    if (H0_NMM !== nothing && S0_NMM !== nothing)
        H0_kMM, S0_kMM = real_to_bloch(kpath.kpoints, H0_NMM, S0_NMM, R)
        ϵerr_kM = zeros(N_k, N_orb)
        for i = 1:N_k
            ϵerr_kM[i, :] = eigvals_error(ϵ_kM[i, :], H_kMM[i, :, :], H0_kMM[i, :, :], 
                                                      S_kMM[i, :, :], S0_kMM[i, :, :])
        end
        return ϵ_kM, ϵerr_kM
    end
    return ϵ_kM
end

function density_of_states(ϵ_kM::Matrix, w_k::Vector; σ=0.01, egrid=nothing, emin=nothing, emax=nothing, N=1000)
    N_k, N_orb = size(ϵ_kM)
    emin === nothing && (emin = minimum(ϵ_kM) - 10σ)
    emax === nothing && (emax = maximum(ϵ_kM) + 10σ)
    egrid === nothing && (egrid = range(emin, emax, length=N))
    δ(ϵ) = exp.(-((egrid .- ϵ) ./ σ).^2) ./ (sqrt(π) * σ)
    return ϵ_kM, w_k, egrid, sum([w_k[k] * δ(ϵ_kM[k, n]) for k=1:N_k, n=1:N_orb])
end

function density_of_states(kmesh, H_NMM::Array{Float64,3}, S_NMM::Array{Float64, 3}, R::Matrix{Int}; kshift=[0, 0, 0], kwargs...)
    kpoints = kgrid_monkhorst_pack(kmesh; kshift=kshift)
    ϵ_kM = kspace_eigensolve(kpoints, H_NMM, S_NMM, R)
    w_k = repeat([1.0/length(kpoints)], length(kpoints)) # check if Gamma point is included, if so should be half weighted
                                                         # use time-reversal symmetry
    return density_of_states(ϵ_kM, w_k; kwargs...)
end

fermi(e, ef, σ) = 1.0 / (1.0 + exp((e - ef) / σ))

function fermi_level(ϵ_kM, w_k, nelectrons, σ=1e-3)
    count_electrons(ef) = 2 * sum(w_k .* fermi.(ϵ_kM, ef, σ)) # 2 electrons per orbital
    f(ef) = count_electrons(ef) - nelectrons
    return find_zero(f,  (minimum(ϵ_kM), maximum(ϵ_kM)))
end

function plot_bands(kpath::KPath, bands::Matrix; ϵ=nothing, fermi_level=0.0, 
                    plotobj=nothing, label=nothing, color=:black, ylabel=L"Energy $E - \epsilon_F$ (eV)")
    k0 = 0.0
    x_k = []
    tick_p = [k0]
    tick_k = kpath.phase == "FCC" ? ["W"] : tick_k = ["Γ"]
    push!(x_k, k0)
    for xi=2:size(kpath.kpoints, 1)
        k0 += norm(kpath.kpoints[xi] .- kpath.kpoints[xi-1])
        push!(x_k, k0)
        indict = [k for (k, v) in kpath.special_points if v==kpath.kpoints[xi] ]
        if length(indict) > 0
            push!(tick_p, k0)
            push!(tick_k, replace(indict[1], "G" => "Γ"))
        end
    end

    plotobj === nothing && (plotobj = plot())
    for i = 1:size(bands, 2)
        thislabel = i == 1 ? label : false
        ribbon = nothing
        (ϵ !== nothing) && (ribbon = ϵ[:, i])
        if ribbon≠nothing
            rib_new_lb = [ribbon[i]>0 ? ribbon[i] : 0 for i = 1:length(ribbon)]
            rib_new_ub = [ribbon[i]>0 ? 0 : -ribbon[i] for i = 1:length(ribbon)]
            ribbon = (rib_new_lb,rib_new_ub)
        end
        plot!(plotobj, x_k, bands[:, i] .- fermi_level, color=color, lw=1, label=thislabel,
                  ribbon=ribbon, fillalpha=0.3)
    end
    for p=1:length(tick_p)
        plot!(plotobj, [tick_p[p]], lw=0.5, c=:gray, seriestype=:vline, label=false)
    end    
    plot!(plotobj, [0.0], lw=1, c=:black, ls=:dash, seriestype=:hline, label=false)
    plot!(plotobj; fg_legend=:transparent, bg_legend=:transparent, 
          grid=false, framestyle = :box, legend=nothing)
    xticks!(plotobj, ([t for t in tick_p],[k for k in tick_k]), fontsize=14)
    ylabel!(plotobj, ylabel)
    
    xlims!(plotobj, 0.0, maximum(x_k))
    
    plot(plotobj, dpi=120)
    return plotobj
end

