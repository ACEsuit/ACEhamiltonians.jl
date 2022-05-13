# path under which Zenodo zip file has been unpacked
DATA_PATH = abspath(joinpath(@__DIR__, "../../"))

include("bands.jl")

using CairoMakie
using LaTeXStrings
CairoMakie.activate!(type = "svg")

##

N_orb = 14
kmesh = [9, 9, 9]

H, S = h5open(joinpath(DATA_PATH, "reference_data", "BCC", "BCC-supercell-000.h5")) do h5file
    dropdims(read(h5file, "aitb/H"), dims=3), dropdims(read(h5file, "aitb/S"), dims=3)
end
Hb, Sb = full_to_block(H, S, N_orb)
R = tb_cells(kmesh)
H_NMM_bcc, S_NMM_bcc, blocks = split_HS(Hb, Sb, R)  # BCC reference HS

# set onsite S to identity - remove numerical noise
S_NMM_bcc[365, :, :] = I(14)

H, S = h5open(joinpath(DATA_PATH, "reference_data", "FCC", "FCC-supercell-000.h5")) do h5file
    dropdims(read(h5file, "aitb/H"), dims=3), dropdims(read(h5file, "aitb/S"), dims=3)
end
Hb, Sb = full_to_block(H, S, N_orb)
R = tb_cells(kmesh)
H_NMM_fcc, S_NMM_fcc, blocks = split_HS(Hb, Sb, R) # FCC reference HS

# set onsite S to identity - remove numerical noise
S_NMM_fcc[365, :, :] = I(14)

fcc_kpath = KPath(:FCC)
bcc_kpath = KPath(:BCC)

## 

# density of states and Fermi level
# BCC
ϵ_kM_bcc, w_k, ϵgrid, d_bcc_fine = density_of_states(kmesh, H_NMM_bcc, S_NMM_bcc, R, 
                                        egrid=range(-1,1,length=1000), σ=0.02)

ϵf_bcc = fermi_level(ϵ_kM_bcc, w_k, 13, 0.02)

# FCC
ϵ_kM_fcc, w_k_fcc, ϵgrid_fcc, d_fcc_fine = density_of_states(kmesh, H_NMM_fcc, S_NMM_fcc, R, 
                                        egrid=range(-1,1,length=1000), σ=0.02)

ϵf_fcc = fermi_level(ϵ_kM_fcc, w_k_fcc, 13, 0.02)


# Exact DFT band structure

# BCC
bands_bcc = bandstructure(bcc_kpath, H_NMM_bcc, S_NMM_bcc, R)

# FCC
bands_fcc = bandstructure(fcc_kpath, H_NMM_fcc, S_NMM_fcc, R)

##

dos = []
efermi = []
allbands = []
allbands_err = []
eigs = []
E_band = []

dos_ref = []
efermi_ref = []
allbands_ref = []
eigs_ref = []
E_band_ref = []

egrid = range(-1,1,length=1000)

for i = 0:15
    println("Reading config $i")
    idx = @sprintf "%03d" i
    filename = joinpath(DATA_PATH, "predicted_data", "FCC-to-BCC", "out_$(idx).h5")
    H, S = h5open(filename) do h5file
        real.(permutedims(read(h5file, "H"), (3, 1, 2))), real.(permutedims(read(h5file, "S"), (3, 1, 2)))
    end
    S[365, :, :] = I(14)
    ϵp_kM, w_k, ϵgrid, d_p = density_of_states(kmesh, H, S, R,
                                               egrid=egrid, σ=0.02)
    ϵf_p = fermi_level(ϵp_kM, w_k, 13, 0.01)

    println("Reading ref config $i")
    ref_filename = joinpath(DATA_PATH, "reference_data",  "FCC-to-BCC", "SK-supercell-$(idx).h5")
    H_ref, S_ref = h5open(ref_filename) do h5file
        dropdims(read(h5file, "aitb/H"), dims=3), dropdims(read(h5file, "aitb/S"), dims=3)
    end
    Hb, Sb = full_to_block(H_ref, S_ref, N_orb)
    H_ref, S_ref, blocks = split_HS(Hb, Sb, R)
    S_ref[365, :, :] = I(14)

    ϵ_kM_ref, w_k, ϵgrid, d = density_of_states(kmesh, H_ref, S_ref, R,
                                                   egrid=egrid, σ=0.02)
    ϵf_ref = fermi_level(ϵp_kM, w_k, 13, 0.01)


    push!(eigs, ϵp_kM)
    push!(E_band, sum(fermi.(ϵp_kM, ϵf_p, 0.01) .* ϵp_kM .* w_k))
    push!(dos, d_p)
    push!(efermi, ϵf_p)
    bands, bands_err = bandstructure(bcc_kpath, H, S, R, H0_NMM=H_ref, S0_NMM=S_ref)
    push!(allbands, bands)
    push!(allbands_err, bands_err)

    push!(eigs_ref, ϵ_kM_ref)
    push!(E_band_ref, sum(fermi.(ϵ_kM_ref, ϵf_ref, 0.01) .* ϵ_kM_ref .* w_k))
    push!(dos_ref, d)
    push!(efermi_ref, ϵf_ref)
    bands_ref = bandstructure(bcc_kpath, H_ref, S_ref, R)
    push!(allbands_ref, bands_ref)
end

##

fcc_atoms = load_json(joinpath(DATA_PATH, "reference_data", "FCC", "FCC-999-supercell.json"))
bcc_atoms = load_json(joinpath(DATA_PATH, "reference_data", "BCC", "BCC-999-supercell.json"))

atoms = [ load_json(joinpath(DATA_PATH, "reference_data", "FCC-to-BCC", "geometry-$(@sprintf "%03d" i).json")) for i=0:15 ]

cell(i) = hcat(hcat(atoms[i]["cell"])...)

function c_over_a(i) 
    L = cell(i)
    a = norm(L[:, 3])
    c = norm(L[:, 1])
    return c/a
end

ca = c_over_a.(1:16)

##

function read_matrix(filename; dim=729*14)
    M = zeros(dim, dim)
    open(filename) do file
        for line in eachline(file)
            i, j, v = split(line)
            i, j, v = parse(Int, i), parse(Int, j), parse(Float64, v)
            M[i, j] = v
        end
    end
    return M
end

function read_FHIaims_HS_out(dirname; dim=729*14)
    H_filename = joinpath(dirname, "hamiltonian.out")
    S_filename = joinpath(dirname, "overlap-matrix.out")
    H = read_matrix(H_filename; dim=dim)
    S = read_matrix(S_filename; dim=dim)
    return H, S
end

##

using FileIO

egrid = range(-1,1,length=1000)

function plot_fcc_bcc!(ax, i; color=:black, ylabel="", shift=5.0, ylims=(-10, 20))
    ax.ylabel = ylabel
    lines!(ax, (egrid .- efermi_ref[i+1]) .* hartree2ev, dos_ref[i+1], color=:red, label="DFT", linewidth=2)
    lines!(ax, (egrid .- efermi[i+1]) .* hartree2ev, dos[i+1] .+ shift, color=:blue, linewidth=2)
    vlines!(ax, [0.0] * hartree2ev, color=color, linestyle=:dash, linewidth=2, label=nothing)

    ax.xticks = [-10, 0, 10, 20]
    ax.xticklabelrotation = π/4
    hideydecorations!(ax, ticklabels=false, ticks=true)
    xlims!(ax, -15.0, 30.0)
    ylims!(ax, ylims...)
end

function image_fcc_bcc!(ax, i)
    idx = @sprintf "%04d" i
    img = FileIO.load(joinpath(DATA_PATH, "reference_data", "FCC-to-BCC", "bain-path$idx.png"))
    ax.aspect = DataAspect()
    ax.yreversed = true
    image!(ax, img')
    hidedecorations!(ax)
    # hidespines!(ax)
end

using PyCall
@pyimport scipy.stats as st


function band_error(bands, bands_ref, ϵf, ϵf_ref; σ=0.003166790852369942, range=2:14)  # equivalent to 1000 K
    norm([norm((bands[:, i] .* fermi.(bands[:, i], ϵf, σ)) - 
               (bands_ref[:, i] .* fermi.(bands_ref[:, i], ϵf_ref, σ))) for i ∈ range]) ./ sqrt(length(bands))
end

dos_error(dos, dos_ref, fermi_level) = (st.wasserstein_distance(dos, dos_ref), 
                                        st.wasserstein_distance(dos[egrid .< fermi_level], dos_ref[egrid .< fermi_level]))

dos_err = [dos_error(dos[i], dos_ref[i], efermi_ref[i]) for i= 1:16]

band_err = [ hartree2ev * band_error(allbands[i], allbands_ref[i], efermi[i], efermi_ref[i]) for i=1:16]

##

# dos_diff = plot(ca, [dot(d[egrid .< ϵf_fcc], d_fcc_fine[egrid .< ϵf_fcc])/norm(d[egrid .< ϵf_fcc])/norm(d_fcc_fine[egrid .< ϵf_fcc]) for d in dos], 
#         label=raw"$D \cdot D_\mathrm{FCC}$", color=3, lw=2, xlabel=raw"$c/a$", ylabel="DoS overlap")
# plot!(ca, [dot(d[egrid .< ϵf_fcc], d_bcc_fine[egrid .< ϵf_fcc])/norm(d[egrid .< ϵf_fcc])/norm(d_bcc_fine[egrid .< ϵf_fcc]) for d in dos], 
#         label=raw"$D \cdot D_\mathrm{BCC}$", color=4, lw=2)

noto_sans = assetpath("fonts", "NotoSans-Regular.ttf")
noto_sans_bold = assetpath("fonts", "NotoSans-Bold.ttf")

fig = Figure(resolution=(600, 700), font=noto_sans)
for (i, (err, label)) in enumerate(zip([dos_err, band_err], 
                                       ["DOS error (all)", "Band error / eV"]))
    ax = Axis(fig[i+1, 1], 
              xlabel = i == 1 ? "" : "Bain path reaction coordinate c/a", 
              ylabel=label, grid=false)
    if label == "DOS error (all)"
        lines!(ax, ca, getindex.(err, 1), label="ACE", color=:blue, linewidth=2)
        # lines!(ax, [-1], [-1], label="DFT", color=:red, linewidth=2)
        ylims!(ax, 0, 0.65)

        ax2 = Axis(fig[i+1, 1], ylabel="DOS error (occupied)", grid=false)
        hidexdecorations!(ax2, grid=true, label=false, ticklabels=true, ticks=false)
        hideydecorations!(ax2, grid=true, label=false, ticklabels=true, ticks=false)
        ax2.yaxisposition = :right
        ax2.xticksvisible = false
        ax2.xticklabelsvisible = false
        ax2.yticklabelsvisible = true
        ax2.yticklabelalign = (:left, :center)
        xlims!(ax2, 0.66, 1.05)
                
        lines!(ax2, ca, getindex.(err, 2), label="Error in ACE model with respect to DFT", color=:blue, linewidth=2, linestyle=:dash)
        annotations!(ax, ["Full DOS", "Occupied DOS"], [Point(0.75, 0.15), Point(0.75, 0.40)], textsize=13, align=(:center, :baseline))
    else
        lines!(ax, ca, err, label="Error in ACE model with respect to DFT", color=:blue, linewidth=2)
    end
    vlines!(ax, [1.0], color=:green, linestyle=:dot, linewidth=2, label="FCC")
    vlines!(ax, [1.0/sqrt(2)], color=:purple, linestyle=:dot, linewidth=2, label="BCC")
    vlines!(ax, [ca[5+1], ca[9+1]], color=(:black, 0.5), linestyle=:dot, linewidth=2)
    xlims!(ax, 0.66, 1.05)
    ax.xticks =  [1/sqrt(2), ca[5+1], ca[9+1], 1.0]
    ax.xtickformat = xs -> [@sprintf "%.2f" x for x in xs]
    hidexdecorations!(ax, grid=true, label=false, ticklabels=i == 1, ticks=false)
    hideydecorations!(ax, grid=true, label=false, ticklabels=false, ticks=false)
end
fig[1, 1] = Legend(fig, fig.content[1], framevisible=false, orientation=:horizontal)

f = fig[4, 1] = GridLayout()
# g = fig[5, 1]

ha = [-0.08, 0.25, 0.73, 1.05]
va = [0.82, 0.5, 0.7, 0.9]

for (j, i) in enumerate([2, 5, 9, 12])
    color = :black
    i == 2 && (color = :purple)
    i == 12 && (color = :green)
    ax = Axis(f[1, j])
    plot_fcc_bcc!(ax, i; color=color, ylabel=j == 1 ? "DoS" : "", shift=15.0, ylims=(-30, 30))
    hidexdecorations!(ax, grid=true, label=false, ticklabels=false, ticks=false)
    hideydecorations!(ax, grid=true, label=false, ticklabels=true, ticks=false)

    img_ax = Axis(fig[3, 1], width=Relative(0.35), height=Relative(0.35), halign=ha[j], valign=va[j])
    image_fcc_bcc!(img_ax, i)
end

for (label, layout) in zip(["(a)", "(b)", "(c)"], [fig[2,1], fig[3,1], fig[4,1]])
    Label(layout[1, 1, TopLeft()], label,
        textsize = 18,
        # font = noto_sans_bold,
        padding = (0, 5, 10, 0),
        halign = :right)
end

Label(fig[4, 1, Bottom()], "Energy relative to Fermi level / eV", halign=:center, valign=:top, padding=(0, 0, 0, 30))

rowsize!(fig.layout, 1, 20)
rowsize!(fig.layout, 4, Auto(0.8))
rowgap!(fig.layout, 1, 0)
rowgap!(fig.layout, 0)
colgap!(f, 10)

save("BCC_to_FCC.pdf", fig)
fig
##

function read_dos(filename)
    H, S = h5open(filename) do h5file
        real.(permutedims(read(h5file, "H"), (3, 1, 2))), real.(permutedims(read(h5file, "S"), (3, 1, 2)))
    end
    S[365, :, :] = I(14)
    ϵp_kM, w_k, ϵgrid, d_p = density_of_states(kmesh, H, S, R,
                                               egrid=egrid, σ=0.02)
    return d_p
end

##

prefix = joinpath(DATA_PATH, "predicted_data", "restricted-models")
filenames = [ joinpath(prefix, "bcc_fit_predict_bcc.h5"),
              joinpath(prefix, "bcc_fit_predict_fcc.h5"),
              joinpath(prefix, "fcc_fit_predict_bcc.h5"),
              joinpath(prefix, "fcc_fit_predict_fcc.h5")]

crossover_dos = []             

for filename in filenames
    d_p = read_dos(filename)
    push!(crossover_dos, d_p)
end

##

d_ACE_bcc = read_dos(joinpath(DATA_PATH, "predicted_data", "BCC", "optimised_model_bcc.h5"))
d_ACE_fcc = read_dos(joinpath(DATA_PATH, "predicted_data", "FCC", "optimised_model_fcc.h5"))

##

function plot_dos!(ax, dos, efermi; color=:black, ylabel="", shift=0.0, ylims=(0, 20))
    ax.ylabel = ylabel
    lines!(ax, (egrid .- efermi) .* hartree2ev, dos .+ shift, color=color, linewidth=2)
    # vline!([efermi[i+1]] * hartree2ev, color=:red, ls=:dash, label=nothing)
end

fig = Figure()
ax = Axis(fig[1, 1], grid=false, title="FCC Density of States", titlesize=22)


plot_dos!(ax, d_fcc_fine, ϵf_fcc; color=:red)
plot_dos!(ax, d_ACE_fcc, ϵf_fcc; color=:blue, shift=12)
@printf("FCC & FCC+BCC & %.3f & %.3f \\\\ \n", dos_error(d_ACE_fcc, d_fcc_fine, ϵf_fcc)...)
plot_dos!(ax, crossover_dos[2], ϵf_fcc; color=:purple, shift=24)
@printf("FCC & BCC & %.2f & %.3f \\\\ \n", dos_error(crossover_dos[2], d_fcc_fine, ϵf_fcc)...)
plot_dos!(ax, crossover_dos[4], ϵf_fcc; color=:green, shift=36)
@printf("FCC & FCC & %.3f & %.3f \\\\ \n", dos_error(crossover_dos[4], d_fcc_fine, ϵf_fcc)...)

ax.xticks = [-10, 0, 10, 20]
ax.xlabel = "Energy E - ϵF / eV"
ax.xlabelsize = 22
ax.xticksize = 22
vlines!(ax, [0.0] * hartree2ev, color=:black, linestyle=:dash, linewidth=2, label=nothing)
hidexdecorations!(ax, label=false, ticklabels=false, ticks=true)
hideydecorations!(ax) #, ticklabels=false, ticks=true)
xlims!(ax, -15.0, 30.0)
ylims!(ax, 0, 52)

annotations!(ax, ["DFT reference", "ACE: full", "ACE: BCC", "ACE: FCC"], 
            color=[:red, :blue, :purple, :green],
            [Point(1, 1), Point(1, 14), Point(1, 25), Point(1, 38)], 
            textsize=20, align=(:left, :baseline))


ax = Axis(fig[1, 2], grid=false, title="BCC Density of States", titlesize=22)

plot_dos!(ax, d_bcc_fine, ϵf_bcc; color=:red)
plot_dos!(ax, d_ACE_bcc, ϵf_bcc; color=:blue, shift=12)
@printf("BCC & FCC+BCC & & %.3f & %.3f \\\\ \n", dos_error(d_ACE_bcc, d_bcc_fine, ϵf_bcc)...)
plot_dos!(ax, crossover_dos[1], ϵf_bcc; color=:purple, shift=24)
@printf("BCC & BCC& %.2f & %.3f \\\\ \n", dos_error(crossover_dos[1], d_bcc_fine, ϵf_bcc)...)
plot_dos!(ax, crossover_dos[3], ϵf_bcc; color=:green, shift=36)
@printf("BCC & FCC & %.3f & %.3f \\\\ \n", dos_error(crossover_dos[3], d_bcc_fine, ϵf_bcc)...)


ax.xticks = [-10, 0, 10, 20]
ax.xlabel = "Energy E - ϵF / eV"
ax.xlabelsize = 22
ax.xticksize = 22
vlines!(ax, [0.0] * hartree2ev, color=:black, linestyle=:dash, linewidth=2, label=nothing)
hidexdecorations!(ax, label=false, ticklabels=false, ticks=true)
hideydecorations!(ax) #, ticklabels=false, ticks=true)
xlims!(ax, -15.0, 30.0)
ylims!(ax, 0, 52)

# annotations!(ax, ["DFT reference", "ACE: BCC only", "ACE: FCC only"], 
#             color=[:red, :purple, :green],
#             [Point(1, 1), Point(1, 13), Point(1, 23)], 
#             textsize=20, align=(:left, :baseline))

save("crossover.pdf", fig)
fig
