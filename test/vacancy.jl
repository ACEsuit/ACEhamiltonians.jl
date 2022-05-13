# path under which Zenodo zip file has been unpacked
DATA_PATH = abspath(joinpath(@__DIR__, "../../"))

include("bands.jl")
using CairoMakie
using ASE
using JuLIP
using JLD2

CairoMakie.activate!(type = "svg")

##

atoms = Atoms(ASE.read_xyz(joinpath(DATA_PATH, "reference_data", "vacancy", "geometry.in")))
N_atoms = length(atoms)

n = neighbourlist(atoms, 4.0)
undercoord = findall([sum(n.i .== i) for i=1:length(atoms)] .== 11)

##

N_orb = 14
N_eig = N_atoms * N_orb

vacancy_jld = joinpath(DATA_PATH, "predicted_data", "vacancy", "vacancy.jld2")
if isfile(vacancy_jld)
    println("Loading vacancy eigenvalues and eigenvectors from JLD2 file")
    ϵf_fcc, ϵf_pred, ϵf_vac,
    ϵ_fcc, ϵ_pred, ϵ_vac,
    ϕ_fcc, ϕ_pred, ϕ_vac = load(vacancy_jld, "ϵf_fcc", "ϵf_pred", "ϵf_vac",
                                "ϵ_fcc", "ϵ_pred", "ϵ_vac",
                                "ϕ_fcc", "ϕ_pred", "ϕ_vac")
else
    println("Loading vacancy H and S matrices and solving eigenproblem. This will take some time.")
    H_vac, S_vac = h5open(joinpath(DATA_PATH, "reference_data", "vacancy", "vacancy.h5")) do h5file
        read(h5file, "H"), read(h5file, "S")
    end
    
    Hb, Sb = full_to_block(H_vac, S_vac, N_orb)
    for bi=1:N_atoms
        Sb[Block(bi, bi)] = I(N_orb)
    end

    H_vac = Hermitian(Matrix(Hb))
    S_vac = Hermitian(Matrix(Sb))
    inv_sqrt_S_vac = inv(sqrt(S_vac))
    Ht_vac = Hermitian(inv_sqrt_S_vac * H_vac * inv_sqrt_S_vac)
    ϵ_vac, ϕ_vac = eigen(Ht_vac)

    H_p, S_p = h5open(joinpath(DATA_PATH, "predicted_data", "vacancy", "vacancy.h5")) do h5file
        read(h5file, "H"), read(h5file, "S")
    end

    H_p .+= H_p'
    H_p[diagind(H_p)] ./= 2
    S_p .+= S_p'
    S_p[diagind(S_p)] ./= 2

    Hb_pred, Sb_pred = full_to_block(H_p, S_p, N_orb)

    for bi=1:N_atoms
        Sb_pred[Block(bi, bi)] = I(N_orb)
    end

    H_pred = Hermitian(Matrix(Hb_pred))
    S_pred = Hermitian(Matrix(Sb_pred))
    inv_sqrt_S_pred = inv(sqrt(S_pred))
    Ht_pred = Hermitian(inv_sqrt_S_pred * H_pred * inv_sqrt_S_pred)
    ϵ_pred, ϕ_pred = eigen(Ht_pred)

    H_fcc, S_fcc = h5open(joinpath(DATA_PATH, "reference_data", "FCC", "FCC-supercell-000.h5")) do h5file
        dropdims(read(h5file, "aitb/H"), dims=3), dropdims(read(h5file, "aitb/S"), dims=3)
    end
    
    H_fcc = Hermitian(H_fcc)
    S_fcc = Hermitian(S_fcc)
    inv_sqrt_S_fcc = inv(sqrt(S_fcc))
    Ht_fcc = Hermitian(inv_sqrt_S_fcc * H_fcc * inv_sqrt_S_fcc)
    ϵ_fcc, ϕ_fcc = eigen(Ht_fcc)

    ϵf_fcc = fermi_level(ϵ_fcc, [1], 13 * 729, 0.01)
    ϵf_vac = fermi_level(ϵ_vac, [1], 13 * 728, 0.01)
    ϵf_pred = fermi_level(ϵ_pred, [1], 13 * 728, 0.01)    

    jldsave("vacancy.jld2";
        ϵf_fcc, ϵf_pred, ϵf_vac,
        ϵ_fcc, ϵ_pred, ϵ_vac,
        ϕ_fcc, ϕ_pred, ϕ_vac)
end

# fig = Figure(resolution=(800,600))

# i, j = 1, 2

# ax, hm1 = heatmap(fig[1,1], (Hb[Block(i,j)]) )
# hidedecorations!(ax)
# ax.title = "DFT H / Ha"
# ax.aspect = 1
# ylims!(N_orb, 0.5)
# Colorbar(fig[1,2], hm1)

# ax, hm2 = heatmap(fig[1,3], (Hb_pred[Block(i,j)]))
# hidedecorations!(ax)
# ax.title = "ACE H / Ha"
# ax.aspect = 1
# ylims!(N_orb, 0.5)
# Colorbar(fig[1,4], hm2)

# ax, hm3 = heatmap(fig[1,5], log10.(abs.(Hb[Block(i,j)] - Hb_pred[Block(i,j)])))
# hidedecorations!(ax)
# ax.title = "log10(error / Ha)"
# ax.aspect = 1
# ylims!(N_orb, 0.5)
# Colorbar(fig[1,6], hm3)

# ax, hm4 = heatmap(fig[2,1], (Sb[Block(i,j)]))
# hidedecorations!(ax)
# ax.title = "DFT S"
# ax.aspect = 1
# ylims!(N_orb, 0.5)
# Colorbar(fig[2,2], hm4)

# ax, hm5 = heatmap(fig[2,3], (Sb_pred[Block(i,j)]))
# hidedecorations!(ax)
# ax.title = "ACE S"
# ax.aspect = 1
# ylims!(N_orb, 0.5)
# Colorbar(fig[2,4], hm5)

# if i != j
#     ax, hm6 = heatmap(fig[2,5], log10.(abs.(Sb[Block(i,j)] - Sb_pred[Block(i,j)])))
#     hidedecorations!(ax)
#     ax.title = "log10(error)"
#     ax.aspect = 1
#     ylims!(N_orb, 0.5)
#     Colorbar(fig[2,6], hm6)
# end

# resize_to_layout!(fig)
# fig
# ##

# errs = zeros(N_atoms, N_atoms)
# for j=1:N_atoms, i=1:N_atoms
#     errs[j, i] = maximum(abs.(Hb[Block(i, j)] - Hb_pred[Block(i, j)]))
# end

# fig = Figure()
# ax, hm = heatmap(fig[1,1], log10.(errs))
# #hidedecorations!(ax)
# ax.title = "log10(MAE block_ij / Ha)"
# ax.aspect = 1
# ylims!(N_atoms, 0.5)
# Colorbar(fig[1,2], hm)
# fig

##

# dϵ_vac = eigvals_error(ϵ_pred, H_pred, H_vac, S_pred, S_vac)

# fig = Figure()
# ax = Axis(fig[1,1])
# l1 = lines!(ax, 1..N_eig, ϵ_vac, label="DFT spectrum")
# l2 = lines!(ax, 1..N_eig, ϵ_pred, label="ACE spectrum")
# fill_between!(ax, 1:N_eig, ϵ_pred, ϵ_pred + dϵ_vac, color=l2.attributes.color)
# axislegend(position=:lt)
# ax.xlabel = "Eigenvalue index"
# ax.ylabel = "Eigenvalue / Ha"
# ylims!(ax, -20, 20)
# fig
##

σ = 0.01
egrid = range(-1,1,length=2001)

δ(ϵ) = exp.(-((egrid .- ϵ) ./ σ).^2) ./ (sqrt(π) * σ)

fulldos(ϵ) = sum([δ(ϵ[i]) for i=1:length(ϵ)])

function pdos(ϵ, ϕ, atoms)
    dos = zeros(length(egrid))
    for i=1:length(ϵ)
        for a in atoms # list of 1, 8, 9, 17, 72, 73, 81, 89, 153, 649, 657
            for j = 1:N_orb
                dos .+= abs(ϕ[(a-1) * N_orb + j, i])^2 * δ(ϵ[i])
            end
        end
    end
    dos
end


##

dos_DFT = fulldos(ϵ_fcc)
dos_vac = fulldos(ϵ_vac)
pdos_vac = pdos(ϵ_vac, ϕ_vac, undercoord)
dos_ACE = fulldos(ϵ_pred)
pdos_ACE = pdos(ϵ_pred, ϕ_pred, undercoord)

##

fig = Figure()

scale =  length(atoms) / length(undercoord)

# ax1 = Axis(fig[1, 1], grid=false, title="(a) Density of states")
# hidexdecorations!(ax1, ticks=false, label=false, ticklabels=false)
# hideydecorations!(ax1)

# # lines!(ax1, (egrid .- ϵf_fcc).* hartree2ev, dos_DFT, label="DFT FCC", linewidth=3, color=:green)
# lines!(ax1, (egrid .- ϵf_fcc).* hartree2ev, dos_DFT .+ 8e3, label="DFT FCC", linewidth=3, color=(:green, 0.5), linestyle=:solid)
# lines!(ax1, (egrid .- ϵf_vac).* hartree2ev, dos_vac .+ 8e3, label="DFT vacancy", linewidth=3, color=:red)
# lines!(ax1, (egrid .- ϵf_vac).* hartree2ev, dos_ACE .+ 16e3, label="ACE vacancy", linewidth=3, color=:blue)
# lines!(ax1, (egrid .- ϵf_fcc).* hartree2ev, dos_DFT .+ 16e3, linewidth=3, color=(:green, 0.5), linestyle=:solid)

# xlims!(ax1, -12, 5)
# ax1.xlabel = "Energy E - ϵF / eV"
# vlines!(ax1, [0], color=:black, linestyle=:dot)
# ylims!(8e3, 26e3)

ax2 = Axis(fig[1, 1], grid=false) #title="(b) Projected density of states (vac. neighbs.)", legend=true)
hidexdecorations!(ax2, ticks=false, label=false, ticklabels=false)
hideydecorations!(ax2)

# lines!(ax2, (egrid .- ϵf_fcc).* hartree2ev, dos_DFT, linewidth=3, color=:green, alpha=0.8)
lines!(ax2, (egrid .- ϵf_fcc).* hartree2ev, dos_DFT .+ 4e3, label="DFT FCC DoS (rescaled)", linewidth=3, color=(:green, 0.5), linestyle=:solid)
lines!(ax2, (egrid .- ϵf_pred).* hartree2ev, scale * pdos_ACE .+ 4e3, label="ACE vacancy PDoS", linewidth=3, color=:blue)
lines!(ax2, (egrid .- ϵf_pred).* hartree2ev, scale * pdos_vac .+ 4e3, label="DFT vacancy PDoS", linewidth=3, color=:red)
# lines!(ax2, (egrid .- ϵf_fcc).* hartree2ev, dos_DFT .+ 16e3, linewidth=3, color=(:green, 0.5), linestyle=:solid)
# lines!(ax2, (egrid .- ϵf_pred).* hartree2ev, scale * (pdos_vac - pdos_ACE), linewidth=3, color=(:black, 0.5), linestyle=:solid, label="(ACE - DFT) vacancy PDoS error")


xlims!(ax2, -12, 5)
vlines!(ax2, [0], color=(:black, 1.0), linestyle=:dot)
hlines!(ax2, [0, 4e3], color=(:black, 1.0), linestyle=:dot)
ax2.xlabel = "Energy E - ϵF / eV"

ylims!(4e3, 14e3)

# fig[2, 1] = Legend(fig, fig.content[1], framevisible=false, orientation=:horizontal)

axislegend(ax2, position=:lt)

save("vacancy.pdf", fig)
fig

##

function charge_density(ϵ, ϕ, ϵf; σ=1e-3)
    N_orb = size(ϕ, 1)
    occ = fermi.(ϵ, ϵf, σ)    
    ρ = zeros(N_orb)
    for i=1:N_orb
        ρ .+= occ[i] * abs2.(@view ϕ[:, i])
    end
    ρ
end

ρ_vac = charge_density(ϵ_vac, ϕ_vac, ϵf_vac)
ρ_pred = charge_density(ϵ_pred, ϕ_pred, ϵf_pred)

##

charge_DFT = vec(sum(reshape(ρ_vac, 14, 728), dims=1))
charge_ACE = vec(sum(reshape(ρ_pred, 14, 728), dims=1))

set_data!(atoms, "charge_DFT", charge_DFT)
set_data!(atoms, "charge_ACE", charge_ACE)
write_extxyz("vacancy.xyz", atoms)
