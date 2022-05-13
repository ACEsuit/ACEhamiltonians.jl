# path under which Zenodo zip file has been unpacked
DATA_PATH = abspath(joinpath(@__DIR__, "../../"))

using HDF5
using Plots
using Plots.PlotMeasures
gr()

using PyCall
@pyimport scipy.stats as st


include("bands.jl")

plot_bands!(plotobj::Plots.Plot, args...; kwargs...) = plot_bands(args...; plotobj=plotobj, kwargs...)

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
ϵ_kM_bcc, w_k, ϵgrid, d_bcc = density_of_states(kmesh, H_NMM_bcc, S_NMM_bcc, R, 
                                        egrid=range(-1,1,length=1000), σ=0.02)

ϵf_bcc = fermi_level(ϵ_kM_bcc, w_k, 13, 0.01)

# FCC
ϵ_kM_fcc, w_k_fcc, ϵgrid_fcc, d_fcc = density_of_states(kmesh, H_NMM_fcc, S_NMM_fcc, R, 
                                        egrid=range(-1,1,length=1000), σ=0.02)

ϵf_fcc = fermi_level(ϵ_kM_fcc, w_k_fcc, 13, 0.01)

##                                

# Exact DFT band structure

# BCC
bands_bcc = bandstructure(bcc_kpath, H_NMM_bcc, S_NMM_bcc, R)
plt = plot_bands(bcc_kpath, hartree2ev * bands_bcc, color=:red, fermi_level=hartree2ev * ϵf_bcc)
ylims!(-15, 30)

##

# FCC
bands_fcc = bandstructure(fcc_kpath, H_NMM_fcc, S_NMM_fcc, R)
plt = plot_bands(fcc_kpath, hartree2ev * bands_fcc, color=:red, fermi_level=hartree2ev * ϵf_fcc)
ylims!(-20, 30)

##

function labels(phase::Symbol)
    if phase == :BCC
        onsite_label = "_bcc"
        offsite_label = "_bcc"
    elseif  phase == :FCC
        onsite_label = ""
        offsite_label = "_1114"
    else
        error("bad phase $phase")
    end
    return onsite_label, offsite_label
end

# Approximate band structure
# ord-2 offsite
function band_plot(i, phase::Symbol; ylabel=L"Energy $E - \epsilon_F$ (eV)", ylims=(-10, 30), Hp_NMM=nothing, Sp_NMM=nothing)
    onsite_label, offsite_label = labels(phase)
    base = phase == :FCC ? "hybridon" : "hybridonoff"
    
    if Hp_NMM === nothing && Sp_NMM === nothing
        Hp_NMM, Sp_NMM = h5open(joinpath(DATA_PATH, "predicted_data", uppercase(string(phase)), "out_hyboff_2_6_2_$(i)_1_$(i+2)$onsite_label.h5")) do h5file
            real.(permutedims(read(h5file, "H"), (3, 1, 2))), real.(permutedims(read(h5file, "S"), (3, 1, 2)))
        end
        Hp2_NMM, Sp2_NMM = h5open(joinpath(DATA_PATH, "predicted_data", uppercase(string(phase)), "out_$(base)_2_6_2_6_1_8$offsite_label.h5")) do h5file
            real.(permutedims(read(h5file, "H"), (3, 1, 2))), real.(permutedims(read(h5file, "S"), (3, 1, 2)))
        end
        Hp_NMM[365,:,:] = Hp2_NMM[365,:,:]
        Sp_NMM[365,:,:] = I(14) # on site overlap
    end

    ϵp_kM, w_k, ϵgrid, d_p_fcc = density_of_states(kmesh, Hp_NMM, Sp_NMM, R, 
    egrid=range(-1,1,length=1000), σ=0.02)
    ϵf_p = fermi_level(ϵp_kM, w_k, 13, 0.01)

    if phase == :FCC
        kpath = fcc_kpath
        bands = bands_fcc
        H_NMM, S_NMM = H_NMM_fcc, S_NMM_fcc
        ϵf = ϵf_fcc
    else
        kpath = bcc_kpath
        bands = bands_bcc
        H_NMM, S_NMM = H_NMM_bcc, S_NMM_bcc
        ϵf = ϵf_bcc
    end

    bands_p, err_p = bandstructure(kpath, Hp_NMM, Sp_NMM, R; H0_NMM=H_NMM, S0_NMM=S_NMM)
    plt = plot_bands(kpath, hartree2ev * bands, color=:red, fermi_level=hartree2ev * ϵf, label="DFT", ylabel=ylabel)
    plot_bands!(plt, kpath, hartree2ev * bands_p, ϵ = hartree2ev * err_p, fermi_level=hartree2ev * ϵf, color=:blue, label="hybrid_onsite(2,6) + hybrid_offsite(2,$i)")
    return (ylims!(ylims...), bands_p, ϵf_p)
end

# ord-1 offsite
function band_plot_new(onsite_order, onsite_degree, offsite_order, offsite_degree, phase::Symbol; ylabel="Energy (eV)", ylims=(-20, 25), Hp_NMM=nothing, Sp_NMM=nothing)
    if Hp_NMM === nothing && Sp_NMM === nothing
        phase_label = labels(phase)[1]
        Hp_NMM, Sp_NMM = h5open(joinpath(DATA_PATH, "predicted_data", uppercase(string(phase)), "out_hyboff_2_6_$(offsite_order)_$(offsite_degree)_1_$(offsite_degree+2)$phase_label.h5")) do h5file
            real.(permutedims(read(h5file, "H"), (3, 1, 2))), real.(permutedims(read(h5file, "S"), (3, 1, 2)))
        end
        Hp2_NMM, Sp2_NMM = h5open(joinpath(DATA_PATH, "predicted_data", uppercase(string(phase)), "out_hyb_$(onsite_order)_$(onsite_degree)_1_6$phase_label.h5")) do h5file
            real.(permutedims(read(h5file, "H"), (3, 1, 2))), real.(permutedims(read(h5file, "S"), (3, 1, 2)))
        end
        Hp_NMM[365,:,:] = Hp2_NMM[365,:,:]
        Sp_NMM[365,:,:] = I(14) # on site overlap
    end

    ϵp_kM, w_k, ϵgrid, d_p_fcc = density_of_states(kmesh, Hp_NMM, Sp_NMM, R, 
    egrid = range(-1,1,length=1000), σ=0.02)
    ϵf_p = fermi_level(ϵp_kM, w_k, 13, 0.01)

    if phase == :FCC
        kpath = fcc_kpath
        bands = bands_fcc
        H_NMM, S_NMM = H_NMM_fcc, S_NMM_fcc
        ϵf = ϵf_fcc
    else
        kpath = bcc_kpath
        bands = bands_bcc
        H_NMM, S_NMM = H_NMM_bcc, S_NMM_bcc
        ϵf = ϵf_bcc
    end

    bands_p, err_p = bandstructure(kpath, Hp_NMM, Sp_NMM, R; H0_NMM=H_NMM, S0_NMM=S_NMM)
    plt = plot_bands(kpath, hartree2ev * bands, color=:red, fermi_level=hartree2ev * ϵf, label="DFT", ylabel=ylabel)
    plot_bands!(plt, kpath, hartree2ev * bands_p,  fermi_level=hartree2ev * ϵf, ϵ = hartree2ev * err_p, color=:blue, label="hybrid_onsite($onsite_order,$onsite_degree) + hybrid_offsite($offsite_order,$offsite_degree)")
    return (ylims!(ylims...), bands_p, ϵf_p)
end

function band_error(bands, bands_ref, ϵf_ref; σ=0.003166790852369942, range=2:14)  # equivalent to 1000 K
    norm([norm((bands[:, i] .* fermi.(bands[:, i], ϵf_ref, σ)) - 
               (bands_ref[:, i] .* fermi.(bands_ref[:, i], ϵf_ref, σ))) for i ∈ range]) ./ sqrt(length(bands))
end

##

band_error_bcc_order2 = []
anim = @animate for i ∈ 6:12
    plt, bands, ϵf = band_plot(i, :BCC)
    title!(plt, "BCC Order 2 Degree $i")
    push!(band_error_bcc_order2, band_error(bands, bands_bcc, ϵf_bcc))
    plt
end
gif(anim, "hybrid_band_convergence_2_6_2_maxdeg_bcc.gif", fps = 1)

##

band_error_fcc_order2 = []
anim_fcc = @animate for i ∈ 6:12
    plt, bands, ϵf = band_plot(i, :FCC)
    title!(plt, "FCC Order 2 Degree $i")
    push!(band_error_fcc_order2, band_error(bands, bands_fcc, ϵf_fcc))
    plt
end
gif(anim_fcc, "hybrid_band_convergence_2_6_2_maxdeg_fcc.gif", fps = 1)


##

band_error_bcc_order1 = []
anim = @animate for i ∈ 6:14
    plt, bands, ϵf = band_plot_new(2, 4, 1, i, :BCC)
    title!(plt, "BCC Order 1 Degree $i")
    push!(band_error_bcc_order1, band_error(bands, bands_bcc, ϵf_bcc))
    plt
end
gif(anim, "hybrid_band_convergence_2_4_1_maxdeg_bcc.gif", fps = 1)

##

band_error_fcc_order1 = []
anim_fcc = @animate for i ∈ 6:14
    plt, bands, ϵf = band_plot_new(2, 4, 1, i, :FCC)
    title!(plt, "FCC Order 1 Degree $i")
    push!(band_error_fcc_order1, band_error(bands, bands_fcc, ϵf_fcc))
    plt
end
gif(anim_fcc, "hybrid_band_convergence_2_4_1_maxdeg_fcc.gif", fps = 1)

##

# value plot

function block!(y1, y2, x1, x2, label, color; transparent=true, do_transpose=true, labels=true)
    color isa Int && (color = palette(:default)[color])
    fillcolor = color
    transparent && (fillcolor = nothing)
    @show y1, y2, x1, x2
    plot!(Shape([x1, x1, (x2+1), (x2+1), x1], 
                [y1, (y2+1), (y2+1), y1, y1]), 
                opacity=0.5, color=fillcolor, linecolor=color, lw=2, legend=nothing)

    revlabel = isa(label, LaTeXString) ? label : Plots.text(reverse(label), 18) 
    label = isa(label, LaTeXString) ? label :  Plots.text(label, 18)

    labels && annotate!((x1+x2+1)/2, (y1+y2+1)/2, label, halign=:center, valign=:center)

    # add transposed block
    if do_transpose && ([x1, x2, y1, y2] != [y1, y2, x1, x2])
        x1, x2, y1, y2 = y1, y2, x1, x2
        plot!(Shape([x1, x1, (x2+1), (x2+1), x1], 
                    [y1, (y2+1), (y2+1), y1, y1]), 
                    opacity=0.5, color=fillcolor, linecolor=color, lw=2, legend=nothing,size=(400,400))
        labels && annotate!((x1+x2+1)/2, (y1+y2+1)/2, revlabel, halign=:center, valign=:center)    
    end
end

function plot_on_off_blocks!()
    for i in 1:2:8, j in 1:2:8
        color = i == j ? 3 : 4
        block!(i, i+1, j, j+1, "", color; do_transpose=false, transparent=false)
        ii = (i+1) ÷ 2
        jj = (j+1) ÷ 2
        i == 3 && j == 3 && annotate!(i + 1, j + 1, text(raw"$\mathbf{H}_{II}$", 24), halign=:center, valign=:center)
        i == 5 && j == 3 && annotate!(i + 1, j + 1, text(raw"$\mathbf{H}_{IJ}$", 24), halign=:center, valign=:center)
    end
end

function plot_blocks!(; transparent=true, labels=true)
    for i in 1:3, j in 1:3
        i < j && continue
        if i == j == 2
            block!(i, i, j, j, "ss", 1; transparent=transparent, labels=labels)
        else
            block!(i, i, j, j, "", 1; transparent=transparent, labels=labels)
        end
    end
    for i = 1:3
        block!(4, 6, i, i, "sp", 2; transparent=transparent, labels=labels)
        block!(7, 9, i, i, "sp", 2; transparent=transparent, labels=labels)
        block!(10, 14, i, i, "sd", 3; transparent=transparent, labels=labels)
    end
    block!(4, 6, 4, 6, "pp", 4; transparent=transparent, labels=labels)
    block!(4, 6, 7, 9, "pp", 4; transparent=transparent, labels=labels)
    block!(7, 9, 7, 9, "pp", 4; transparent=transparent, labels=labels)
    block!(10, 14, 4, 6, "pd", 5; transparent=transparent, labels=labels)
    block!(10, 14, 7, 9, "pd", 5; transparent=transparent, labels=labels)
    block!(10, 14, 10, 14, "dd", 6; transparent=transparent, labels=labels)
end

make_plt(N) = plot(; aspect_ratio=:equal, yflip=true, xlims=(0.9, N+1.1), ylims=(0.9, N+1.1), 
     xticks=(collect(1:N) .+ .5, string.(1:N)), yticks=(collect(1:N) .+ .5, string.(1:N)),
     grid=false)

p1 = plot(; aspect_ratio=:equal, yflip=true, axis=nothing, border=:none)
plot_on_off_blocks!()

p2 = make_plt(14)
plot_blocks!(transparent=false)

# p2 = make_plt()
# heatmap!(collect(1:14) .+ .5, collect(1:14) .+ .5, 
#          log10.(abs.(H_NMM[365,:,:] - Hp_NMM[365,:,:])), aspect_ratio=:equal, yflip=true, clims=(-10, 1))
# plot_blocks!(transparent=true)   

# l = @layout[a{0.5w,1h}  b{0.45w,1h}]
plt = plot(p1, p2, size = (1300, 600), title=reshape(["(a) Hamiltonian and overlap structure", 
    raw"(b) Block structure of $\mathbf{H}_{II}$ and $\mathbf{H}_{IJ}$"], 1, :))
savefig("HS_schematic.pdf")
plt
##

function dos_plot(i, phase::Symbol; Hp_NMM=nothing, Sp_NMM=nothing, 
                  xlabel=L"Energy $E - \epsilon_F$ / eV", ylabel="DoS", swap_xy=false,
                  shift=0.0, elims=(-10, 30))
    onsite_label, offsite_label = labels(phase)
    base = phase == :FCC ? "hybridon" : "hybridonoff"
    
    if Hp_NMM === nothing && Sp_NMM === nothing
        Hp_NMM, Sp_NMM = h5open(joinpath(DATA_PATH, "predicted_data", uppercase(string(phase)), "out_hyboff_2_6_2_$(i)_1_$(i+2)$onsite_label.h5")) do h5file
            real.(permutedims(read(h5file, "H"), (3, 1, 2))), real.(permutedims(read(h5file, "S"), (3, 1, 2)))
        end
        Hp2_NMM, Sp2_NMM = h5open(joinpath(DATA_PATH, "predicted_data", uppercase(string(phase)), "out_$(base)_2_6_2_6_1_8$offsite_label.h5")) do h5file
            real.(permutedims(read(h5file, "H"), (3, 1, 2))), real.(permutedims(read(h5file, "S"), (3, 1, 2)))
        end
        Hp_NMM[365,:,:] = Hp2_NMM[365,:,:]
        Sp_NMM[365,:,:] = I(14) # on site overlap
    end
    ϵp_kM, w_k, ϵgrid, d_p = density_of_states(kmesh, Hp_NMM, Sp_NMM, R, 
    egrid=range(-1,1,length=1000), σ=0.02)

    ϵf_p = fermi_level(ϵp_kM, w_k, 13, 0.01)

    if phase == :FCC
        d, ϵf = d_fcc, ϵf_fcc
    else
        d, ϵf = d_bcc, ϵf_bcc
    end
    x = (collect(ϵgrid) .- ϵf) * hartree2ev
    y = d / hartree2ev
    _xlabel, _ylabel = xlabel, ylabel
    if swap_xy
        x, y = y, x
        _xlabel, _ylabel = _ylabel, _xlabel
    end
    plot(x, y, lw=1.5, label="DFT", grid=nothing,
        xlabel=_xlabel, ylabel=_ylabel, color=:red, legend=nothing, title="$(string(phase)) - Degree $i")

    if swap_xy
        hline!([0.0], linestyle=:dash, label=nothing, color=:black)
    else
        vline!([0.0], linestyle=:dash, label=nothing, color=:black)
    end

    x, y = (collect(ϵgrid) .- ϵf) * hartree2ev, d_p / hartree2ev
    y .+= shift
    if swap_xy
        x, y = y, x
    end
    plot!(x, y, lw=1.5, label="ACEtb max degree $i", 
          color=:blue)
    if swap_xy
        ylims!(elims...)
        xlims!(0, 0.8)
        xticks!(Float64[])        
    else
        xlims!(elims...)
        ylims!(0, 0.8)    
        yticks!(Float64[])
    end
    #title!("Degree $i")
end

function dos_plot_new(onsite_order, onsite_degree, offsite_order, offsite_degree, phase::Symbol; Hp_NMM=nothing, Sp_NMM=nothing)
    if Hp_NMM === nothing && Sp_NMM === nothing
        phase_label = labels(phase)[1]
        Hp_NMM, Sp_NMM = h5open(joinpath(DATA_PATH, "predicted_data", uppercase(string(phase)), "out_hyboff_2_6_$(offsite_order)_$(offsite_degree)_1_$(offsite_degree+2)$phase_label.h5")) do h5file
            real.(permutedims(read(h5file, "H"), (3, 1, 2))), real.(permutedims(read(h5file, "S"), (3, 1, 2)))
        end
        Hp2_NMM, Sp2_NMM = h5open(joinpath(DATA_PATH, "predicted_data", uppercase(string(phase)), "out_hyb_$(onsite_order)_$(onsite_degree)_1_6$phase_label.h5")) do h5file
            real.(permutedims(read(h5file, "H"), (3, 1, 2))), real.(permutedims(read(h5file, "S"), (3, 1, 2)))
        end
        Hp_NMM[365,:,:] = Hp2_NMM[365,:,:]
        Sp_NMM[365,:,:] = I(14) # on site overlap
    end
    ϵp_kM, w_k, ϵgrid, d_p = density_of_states(kmesh, Hp_NMM, Sp_NMM, R, 
    egrid=range(-1,1,length=1000), σ=0.02)

    ϵf_p = fermi_level(ϵp_kM, w_k, 13, 0.01)

    if phase == :FCC
        d, ϵf = d_fcc, ϵf_fcc
    else
        d, ϵf = d_bcc, ϵf_bcc
    end
    plot(ϵgrid * hartree2ev, d / hartree2ev, lw=1, label="DFT", grid=nothing,
        xlabel="Energy / eV", ylabel="DoS", color=:red, legend=nothing, 
        title="$(string(phase)) onsite ($onsite_order,$onsite_degree) offsite ($offsite_order,$offsite_degree)")
    vline!([ϵf * hartree2ev], linestyle=:dash, label=nothing, color=:red, ylims = (0,0.6))

    plot!(ϵgrid * hartree2ev, d_p / hartree2ev, lw=1, label="ACEtb onsite ($onsite_order, $onsite_degree) offsite ($offsite_order, $offsite_degree)", 
          color=:blue)
    vline!([ϵf_p * hartree2ev], linestyle=:dash, label=nothing, color=:blue)

    xlims!(-20, 20)
    ylims!(0, 0.6)

    #title!("Degree $i")
end

##

anim_dos_bcc = @animate for i ∈ 6:12
    dos_plot(i, :BCC)
end
gif(anim_dos_bcc, "DoS_convergence_2_12_2_maxdeg_bcc.gif", fps = 2)

##

anim_dos_fcc = @animate for i ∈ 6:12
    dos_plot(i, :FCC)
end
gif(anim_dos_fcc, "DoS_convergence_2_12_2_maxdeg_fcc.gif", fps = 2)

##

anim_dos_bcc = @animate for i ∈ 6:14
    dos_plot_new(2, 4, 1, i, :BCC)
end
gif(anim_dos_bcc, "DoS_convergence_2_4_1_maxdeg_bcc.gif", fps = 2)

##

anim_dos_fcc = @animate for i ∈ 6:14
    dos_plot_new(2, 4, 1, i, :FCC)
end
gif(anim_dos_fcc, "DoS_convergence_2_4_1_maxdeg_fcc.gif", fps = 2)

##

plts = []
for i ∈ 6:12
    push!(plts, dos_plot(i, :FCC, xlabel = i == 12 ? L"Energy $E$ / eV" : nothing))
    push!(plts, dos_plot(i, :BCC, xlabel = i == 12 ? L"Energy $E$ / eV" : nothing, ylabel=nothing))
end

plt = plot(plts..., size=(800, 1000), layout=(7, 2))
savefig("DoS_convergence_order_2.pdf")
plt

##

plts = []
for i ∈ 6:14
    push!(plts, dos_plot_new(2, 4, 1, i, :FCC))
    i == 14 && xlabel!(L"Energy $E$ / eV")
    push!(plts, dos_plot_new(2, 4, 1, i, :BCC))
    ylabel!("")
    i == 14 && xlabel!(L"Energy $E$ / eV")
end

plt = plot(plts..., size=(800, 1200), layout=(9, 2))
savefig("DoS_convergence_order_1.pdf")
plt

##

function dos_error(i, phase::Symbol; Hp_NMM=nothing, Sp_NMM=nothing)
    onsite_label, offsite_label = labels(phase)
    base = phase == :FCC ? "hybridon" : "hybridonoff"
    
    if Hp_NMM === nothing && Sp_NMM === nothing
        Hp_NMM, Sp_NMM = h5open(joinpath(DATA_PATH, "predicted_data", uppercase(string(phase)), "out_hyboff_2_6_2_$(i)_1_$(i+2)$onsite_label.h5")) do h5file
            real.(permutedims(read(h5file, "H"), (3, 1, 2))), real.(permutedims(read(h5file, "S"), (3, 1, 2)))
        end
        Hp2_NMM, Sp2_NMM = h5open(joinpath(DATA_PATH, "predicted_data", uppercase(string(phase)), "out_$(base)_2_6_2_6_1_8$offsite_label.h5")) do h5file
            real.(permutedims(read(h5file, "H"), (3, 1, 2))), real.(permutedims(read(h5file, "S"), (3, 1, 2)))
        end
        Hp_NMM[365,:,:] = Hp2_NMM[365,:,:]
        Sp_NMM[365,:,:] = I(14) # on site overlap
    end

    ϵp_kM, w_k, ϵgrid, d_p = density_of_states(kmesh, Hp_NMM, Sp_NMM, R, 
    egrid=range(-1,1,length=1000), σ=0.02)
    
    if phase == :FCC
        d = d_fcc
        ϵf = ϵf_fcc
    else
        d = d_bcc
        ϵf = ϵf_bcc
    end
    return (st.wasserstein_distance(d_p, d), 
            st.wasserstein_distance(d_p[ϵgrid .< ϵf], d[ϵgrid .< ϵf]))
end

function dos_error_new(onsite_order, onsite_degree, offsite_order, offsite_degree, phase::Symbol; Hp_NMM=nothing, Sp_NMM=nothing)

    if Hp_NMM === nothing && Sp_NMM === nothing
        phase_label = labels(phase)[1]
        Hp_NMM, Sp_NMM = h5open(joinpath(DATA_PATH, "predicted_data", uppercase(string(phase)), "out_hyboff_2_6_$(offsite_order)_$(offsite_degree)_1_$(offsite_degree+2)$phase_label.h5")) do h5file
            real.(permutedims(read(h5file, "H"), (3, 1, 2))), real.(permutedims(read(h5file, "S"), (3, 1, 2)))
        end
        Hp2_NMM, Sp2_NMM = h5open(joinpath(DATA_PATH, "predicted_data", uppercase(string(phase)), "out_hyb_$(onsite_order)_$(onsite_degree)_1_6$phase_label.h5")) do h5file
            real.(permutedims(read(h5file, "H"), (3, 1, 2))), real.(permutedims(read(h5file, "S"), (3, 1, 2)))
        end
        Hp_NMM[365,:,:] = Hp2_NMM[365,:,:]
        Sp_NMM[365,:,:] = I(14) # on site overlap
    end

    ϵp_kM, w_k, ϵgrid, d_p = density_of_states(kmesh, Hp_NMM, Sp_NMM, R, 
    egrid=range(-1,1,length=1000), σ=0.02)
    
    if phase == :FCC
        d = d_fcc
        ϵf = ϵf_fcc
    else
        d = d_bcc
        ϵf = ϵf_bcc
    end

    return (st.wasserstein_distance(d_p, d), 
            st.wasserstein_distance(d_p[ϵgrid .< ϵf], d[ϵgrid .< ϵf]))
end

##

degree = 6:12
dos_error_fcc_order2 = dos_error.(degree, :FCC)
dos_error_bcc_order2 = dos_error.(degree, :BCC)

##

degree = 6:14
dos_error_fcc_order1 = dos_error_new.(2, 4, 1, degree, :FCC)
dos_error_bcc_order1 = dos_error_new.(2, 4, 1, degree, :BCC)

##

using Glob

errors = Dict()

offsite_models = [ "SS", "SP", "SD", "PS", "PP", "PD", "DS", "DP", "DD"]

onsite_models = [ "SS", "SP", "SD", "PP", "PD", "DD"]
##

using Statistics


errors = Dict()
for errorfile in glob("*_errors.json", joinpath(DATA_PATH, "model_errors", "offsite"))
    modelID = split(splitpath(errorfile)[end], "_")[1]
    errors[modelID] = JSON.parsefile(errorfile)
end

errors["tuned"] = JSON.parsefile(joinpath(DATA_PATH, "model_errors", "optimised_models", "2606434113214136884_errors.json"))

offsite_maxdegs = 6:14
offsite_errors =  Dict("H train" => zeros(2,length(offsite_maxdegs), length(offsite_models)),
                       "H test" => zeros(2,length(offsite_maxdegs), length(offsite_models)),
                       "S train" => zeros(2,length(offsite_maxdegs), length(offsite_models)),
                       "S test" => zeros(2,length(offsite_maxdegs), length(offsite_models)))
               
tuned_errors = Dict("H train" => [],
                    "H test" => [],
                    "S train" => [],
                    "S test" => [])

for (modelID, entry) in errors
    order = entry["H_params"]["order"][1][1][1]
    maxdeg = entry["H_params"]["maxdeg"][1][1][1]
    @show modelID, order, maxdeg
    H_train_err, H_test_err = entry["H_error"]
    S_train_err, S_test_err = entry["S_error"]

    if modelID == "tuned"
        for (err, label) in zip([H_train_err, H_test_err, S_train_err, S_test_err],
            ["H train", "H test", "S train", "S test"])
            for (i, block) in enumerate(err)
                push!(tuned_errors[label], block)
            end
        end
    else
        for (err, label) in zip([H_train_err, H_test_err, S_train_err, S_test_err],
                                ["H train", "H test", "S train", "S test"])
            for (i, block) in enumerate(err)
                offsite_errors[label][order, maxdeg-5, i] = mean(mean(block))
            end
        end
    end
end

##

errors = Dict()
for errorfile in glob("*_errors.json", joinpath(DATA_PATH, "model_errors", "onsite"))
    modelID = split(splitpath(errorfile)[end], "_")[1]
    errors[modelID] = JSON.parsefile(errorfile)
end

onsite_maxdegs = 4:12
onsite_errors =  Dict("H train" => zeros(length(onsite_maxdegs), length(onsite_models)),
                      "H test" => zeros(length(onsite_maxdegs), length(onsite_models)))

onsite_error_scale = [norm(H_NMM_fcc[365,1:3,1:3], Inf),     # SS
                      norm(H_NMM_fcc[365,4:9,1:3], Inf),     # SP
                      norm(H_NMM_fcc[365,10:14,1:3], Inf),   # SD
                      norm(H_NMM_fcc[365,4:9,4:9], Inf),     # PP
                      norm(H_NMM_fcc[365,10:14,4:9], Inf),   # PD
                      norm(H_NMM_fcc[365,10:14,10:14], Inf)] # DD

onsite_error_scale .*= hartree2ev
                      
for (modelID, entry) in errors
    order = entry["H_params"]["order"][1][1][1]
    maxdeg = entry["H_params"]["maxdeg"][1][1][1]
    @show order, maxdeg
    H_train_err, H_test_err = entry["H_error"]

    for (err, label) in zip([H_train_err, H_test_err],
                            ["H train", "H test"])
        for (i, block) in enumerate(err)
            onsite_errors[label][maxdeg-3, i] = mean(mean(block))
        end
    end
end

##

offsite_colors = reshape([1,2,3,2,4,5,3,5,6], 1, :)
onsite_colors = reshape([1,2,3,4,5,6], 1, :)
offsite_markers = reshape([:none, :diamond, :diamond, :square, :none, :diamond, :square, :square, :none], 1, :)

p1 = plot(onsite_maxdegs, onsite_errors["H train"], label=reshape(lowercase.(onsite_models), 1, :), 
        xlabel="Maximum degree", ylabel="RMSE / eV", 
        color=onsite_colors, yscale=:log10, legend=:topright)
plot!(onsite_maxdegs, onsite_errors["H test"], ls=:dash, label="",
        color=onsite_colors)
title!(raw"(a) Order 2. Onsite $\mathbf{H}_{IJ}-\tilde{\mathbf{H}}_{IJ}$")

p2 = plot(offsite_maxdegs, offsite_errors["H train"][1,:,:], label=reshape(lowercase.(offsite_models), 1, :), 
        markershape=offsite_markers,
        markersize=3,
        markerstrokecolor=offsite_colors,
        xlabel="Maximum degree", ylabel="RMSE / eV", 
        color=offsite_colors, yscale=:log10, legend=:topright, ylims=(2e-4, 1e-1))
plot!(offsite_maxdegs, offsite_errors["H test"][1,:,:], ls=:dash, label="",
        color=offsite_colors)
title!(raw"(b) Order 1. Offsite $\mathbf{H}_{IJ}-\tilde{\mathbf{H}}_{IJ}$")

p3 = plot(offsite_maxdegs[1:end-2], offsite_errors["H train"][2,1:end-2,:], label=reshape(lowercase.(offsite_models), 1, :), 
        markershape=offsite_markers,
        markersize=3, legend=nothing,
        markerstrokecolor=offsite_colors,
        xlabel="Maximum degree", ylabel="RMSE / eV", 
        color=offsite_colors, yscale=:log10, ylims=(2e-4, 1e-1))
plot!(offsite_maxdegs[1:end-2], offsite_errors["H test"][2,1:end-2,:], ls=:dash, label="",
        color=offsite_colors)
title!(raw"(c) Order 2. Offsite $\mathbf{H}_{IJ}-\tilde{\mathbf{H}}_{IJ}$")

p = plot(p1, p2, p3, layout=(1, 3), size=(1200, 500))
savefig("HS_convergence.pdf")

p

##

Hp_NMM_bcc, Sp_NMM_bcc = h5open(joinpath(DATA_PATH, "predicted_data", "BCC", "optimised_model_bcc.h5")) do h5file
     real.(permutedims(read(h5file, "H"), (3, 1, 2))), real.(permutedims(read(h5file, "S"), (3, 1, 2)))
end

bcc_plt, bands, ϵf_bcc = band_plot(0, :BCC, Hp_NMM=Hp_NMM_bcc, Sp_NMM=Sp_NMM_bcc, ylims=(-15, 30))
band_error_bcc_tuned = band_error(bands, bands_bcc, ϵf_bcc)
dos_error_bcc_tuned = dos_error(0, :BCC, Hp_NMM=Hp_NMM_bcc, Sp_NMM=Sp_NMM_bcc)
bcc_dos_plt = dos_plot(0, :BCC, Hp_NMM=Hp_NMM_bcc, Sp_NMM=Sp_NMM_bcc, swap_xy=true, shift=0.3, elims=(-15, 30))
xlims!(bcc_dos_plt, -0.1, 0.8)

##

Hp_NMM_fcc, Sp_NMM_fcc = h5open(joinpath(DATA_PATH, "predicted_data", "FCC", "optimised_model_fcc.h5")) do h5file
    real.(permutedims(read(h5file, "H"), (3, 1, 2))), real.(permutedims(read(h5file, "S"), (3, 1, 2)))
end

fcc_plt, bands, ϵf_fcc = band_plot(0, :FCC, Hp_NMM=Hp_NMM_fcc, Sp_NMM=Sp_NMM_fcc, ylims=(-15, 30))
band_error_fcc_tuned = band_error(bands, bands_fcc, ϵf_fcc)
dos_error_fcc_tuned = dos_error(0, :FCC, Hp_NMM=Hp_NMM_fcc, Sp_NMM=Sp_NMM_fcc)
fcc_dos_plt = dos_plot(0, :FCC, Hp_NMM=Hp_NMM_fcc, Sp_NMM=Sp_NMM_fcc, swap_xy=true, shift=0.3, elims=(-15, 30))
xlims!(fcc_dos_plt, -0.1, 0.8)
#ylims!(fcc_dos_plt, -20, 25)

##

l = @layout[a{0.7w,0.3h}  b{0.3w,0.5h}
            c{0.7w,0.3h}  d{0.3w,0.5h}]

title!(fcc_plt, "")
title!(bcc_plt, "")
ylabel!(fcc_plt, L"Energy $E - \epsilon_F$ (eV)")
ylabel!(bcc_plt, L"Energy $E - \epsilon_F$ (eV)")
title!(fcc_dos_plt, "")
title!(bcc_dos_plt, "")
xlabel!(fcc_dos_plt, "")

xlabel!(bcc_dos_plt, "DoS")
ylabel!(fcc_dos_plt, "")
ylabel!(bcc_dos_plt, "")

annotate!(fcc_plt, 0.3, -12.5, text("FCC", 10))
annotate!(bcc_plt, 0.25, -12.5, text("BCC", 10))

annotate!(fcc_dos_plt, 0.12, -12.5, text("DFT", 10, color=:red))
annotate!(fcc_dos_plt, 0.42, -12.5, text("ACE", 10, color=:blue))

annotate!(bcc_dos_plt, 0.12, -12.5, text("DFT", 10, color=:red))
annotate!(bcc_dos_plt, 0.42, -12.5, text("ACE", 10, color=:blue))


plt = plot(fcc_plt, fcc_dos_plt,
           bcc_plt, bcc_dos_plt, size=(600, 500), layout=l)
savefig("bands_tuned.pdf")
plt

##


p1 = plot(6:14, getindex.(dos_error_fcc_order1, 1), label = "FCC order 1", lw=2, color=3, 
     ylabel="DoS error", grid=false, left_margin=8mm, legend=false)
plot!(6:12, getindex.(dos_error_fcc_order2, 1), label = "FCC order 2", lw=2, color=3, ls=:dash)
hline!([dos_error_fcc_tuned[1]], color=3, label="FCC optimal", ls=:dot, lw=2)

plot!(6:14, getindex.(dos_error_bcc_order1, 1), label = "BCC order 1", lw=2, color=4)
plot!(6:12, getindex.(dos_error_bcc_order2, 1), label = "BCC order 2", lw=2, color=4, ls=:dash)
hline!([dos_error_bcc_tuned[1]], color=4, label="BCC optimal", ls=:dot, lw=2)

title!("(a) DoS error for all states")
xlims!(6,12)
ylims!(0, 1)

p2 = plot(6:14, getindex.(dos_error_fcc_order1, 2), label = "FCC order 1", lw=2, color=3, 
        ylabel="DoS error", grid=false, legend=true)
plot!(6:12, getindex.(dos_error_fcc_order2, 2), label = "FCC order 2", lw=2, color=3, ls=:dash)
hline!([dos_error_fcc_tuned[2]], color=3, label="FCC optimal", ls=:dot, lw=2)

plot!(6:14, getindex.(dos_error_bcc_order1, 2), label = "BCC order 1", lw=2, color=4)
plot!(6:12, getindex.(dos_error_bcc_order2, 2), label = "BCC order 2", lw=2, color=4, ls=:dash)
hline!([dos_error_bcc_tuned[2]], color=4, label="BCC optimal", ls=:dot, lw=2)

title!("(b) DoS error for occupied states")
xlims!(6,12)
ylims!(0, 0.2)

p3 = plot(6:14, band_error_fcc_order1 * hartree2ev, xlabel="Maximum polynomial degree", ylabel="RMSE Band Error / eV", 
    label = "FCC order 1", lw=2, color=3, grid=nothing, legend=false)
plot!(6:12, band_error_fcc_order2 * hartree2ev, label = "FCC order 2", lw=2, color=3, ls=:dash)
hline!([band_error_fcc_tuned] .* hartree2ev, color=3, label="FCC optimal", ls=:dot, lw=2)

plot!(6:14, band_error_bcc_order1 * hartree2ev, label = "BCC order 1", lw=2, color=4)
plot!(6:12, band_error_bcc_order2 * hartree2ev, label = "BCC order 2", lw=2, color=4, ls=:dash)
hline!([band_error_bcc_tuned] .* hartree2ev, color=4, label="BCC optimal", ls=:dot, lw=2)

title!("(c) Band structure error")
xlims!(6, 12)
ylims!(0, 1.0)

plt = plot(p1, p2, p3, layout=(3, 1), size=(600, 700))
savefig("bands_DoS_convergence.pdf")
plt

##

# heatmaps showing errors in H and S with optimal model

p1 = make_plt(14)
p2 = make_plt(14)
p3 = make_plt(14)
p4 = make_plt(14)
p5 = make_plt(14)
p6 = make_plt(14)
p7 = make_plt(14)
p8 = make_plt(14)
p9 = make_plt(14)
p10 = make_plt(14)
p11 = make_plt(14)
p12 = make_plt(14)

function droptol(A; tol=1e-8)
    B = zeros(size(A))
    B[abs.(A) .> tol] = A[abs.(A) .> tol]
    B
end

plt = plot(heatmap!(p4, log10.(abs.(droptol(H_NMM_fcc[365,:,:]))),  
        colorbar=false, clims=(-8, 1), title=raw"Onsite $\log_{10}|\mathbf{H}_{IJ}|$", titlefont = font(12)),
    heatmap!(p5, log10.(abs.(droptol(H_NMM_fcc[366,:,:]))),  
        colorbar=true, clims=(-8, 1), title=raw"Offsite $\log_{10}|\mathbf{H}_{IJ}|$", titlefont = font(12)),
    heatmap!(p7, log10.(abs.(droptol(H_NMM_fcc[365,:,:] - Hp_NMM_fcc[365,:,:]))), 
              colorbar=false, clims=(-8, 1), title=raw"Onsite $\log_{10}|\mathbf{H}_{II} - \tilde{\mathbf{H}}_{II}|$", titlefont = font(12)),
    heatmap!(p8, log10.(abs.(droptol(H_NMM_fcc[366,:,:] - Hp_NMM_fcc[366,:,:]))),  
              colorbar=false, clims=(-8, 1), title=raw"Offsite $\log_{10}|\mathbf{H}_{IJ} - \tilde{\mathbf{H}}_{IJ}|$", titlefont = font(12)),
      
    aspect_ratio=:equal, size=(600, 600), layout=(2, 2))

savefig("HS.pdf")
plt
