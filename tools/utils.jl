using BenchmarkTools, Serialization
using ACEhamiltonians, HDF5, Serialization
using Statistics
using ACEhamiltonians.DatabaseIO: load_hamiltonian_gamma, load_overlap_gamma, load_density_matrix_gamma
using JuLIP: Atoms
using PeriodicTable
using Statistics
using Plots

# ------------------------------------
# |***functions slicing the matrix***| 
# ------------------------------------
function slice_matrix(matrix::AbstractArray{<:Any, 3}, id::Union{NTuple{3, Int}, NTuple{4, Int}},
                        atoms::Atoms, basis_def::BasisDef)

    blocks, block_idxs = locate_and_get_sub_blocks(matrix, id..., atoms, basis_def)

    return blocks

end


# get the species and Azimuthal number
function get_az_species(basis_definition::BasisDef, id::Union{NTuple{3, Int}, NTuple{4, Int}})
    if length(id) == 3
        return (id[begin], basis_definition[id[begin]][id[end-1]], basis_definition[id[end-2]][id[end]])
    elseif length(id) == 4
        return (id[begin], id[2], basis_definition[id[begin]][id[end-1]], basis_definition[id[end-2]][id[end]])
    end
end


# return the metrics
function metrics(errors::Union{Vector{Array{Float64,3}}, Vector{<:Any}}, type::String)
    if isempty(errors)
        return nothing
    end
    errors_tensor = cat(errors..., dims=3) 
    if type == "MAE"
        return mean(abs.(errors_tensor))
    elseif type == "RMSE"
        return sqrt(mean(errors_tensor.^2))
    else
        throw(ArgumentError("invalid metrics type, the matrics should be either MAE or RMSE"))
    end
end


# return the error data dict
ob_idx = Dict(0=>"s", 1=>"p", 2=>"d", 3=>"f")

function get_error_dict(errors::Vector{Array{Float64, 3}}, atomsv::Vector{Atoms{Float64}}, model::Model)

    basis_definition = model.basis_definition

    data_dict = Dict("MAE"=>Dict(), "RMSE"=>Dict())
    data_dict["MAE"]["all"] = mean(abs.(vcat([error[:] for error in errors]...))) 
    data_dict["RMSE"]["all"] = sqrt(mean(vcat([error[:] for error in errors]...).^2))

    for site in ["on", "off"]

        submodels = site == "on" ? model.on_site_submodels : model.off_site_submodels
        data_dict["MAE"][site] = Dict()
        data_dict["RMSE"][site] = Dict()

        # all the sumodels
        error_vector = []
        for id in keys(submodels)
            error_block = slice_matrix.(errors, Ref(id), atomsv, Ref(basis_definition))
            push!(error_vector, [error[:] for error in error_block])
        end
        error_vector = [error_vector[i][j] for i in 1:length(error_vector) for j in 1:length(error_vector[i])]
        error_vector = vcat(error_vector...)
        data_dict["MAE"][site]["all"] = mean(abs.(error_vector))
        data_dict["RMSE"][site]["all"] = sqrt(mean(error_vector.^2))

        #submodels 
        data_dict["MAE"][site]["submodels"] = Dict()
        data_dict["RMSE"][site]["submodels"] = Dict()
        for id in keys(submodels)
            error_block = slice_matrix.(errors, Ref(id), atomsv, Ref(basis_definition))
            data_dict["MAE"][site]["submodels"][id] = metrics(error_block, "MAE")
            data_dict["RMSE"][site]["submodels"][id] = metrics(error_block, "RMSE")
        end

        ids = collect(keys(submodels))

        #Azimuthal number
        data_dict["MAE"][site]["Azimuthal"] = Dict()
        data_dict["RMSE"][site]["Azimuthal"] = Dict()
        az_pairs = unique([(basis_definition[i[begin]][i[end-1]], basis_definition[i[end-2]][i[end]]) for i in ids])
        for (l₁, l₂) in az_pairs
            error_block_collection = []
            for id in keys(submodels)
                if (basis_definition[id[begin]][id[end-1]], basis_definition[id[end-2]][id[end]]) == (l₁, l₂)
                    error_block = slice_matrix.(errors, Ref(id), atomsv, Ref(basis_definition))
                    push!(error_block_collection, error_block)
                end
            end
            error_block_collection = [error_block_collection[i][j] for i in 1:length(error_block_collection)
                                        for j in 1:length(error_block_collection[i])]
            data_dict["MAE"][site]["Azimuthal"][join([ob_idx[l₁],ob_idx[l₂]])] = metrics(error_block_collection, "MAE")
            data_dict["RMSE"][site]["Azimuthal"][join([ob_idx[l₁],ob_idx[l₂]])] = metrics(error_block_collection, "RMSE")     
        end
        
        #species Azimuthal
        data_dict["MAE"][site]["Azimuthal_Species"] = Dict()
        data_dict["RMSE"][site]["Azimuthal_Species"] = Dict()
        az_species_pairs = unique(get_az_species.(Ref(basis_definition), ids))
        for pair in az_species_pairs
            species = join([elements[i].symbol for i in pair[begin:end-2]])
            azimuthal = join([ob_idx[pair[end-1]],ob_idx[pair[end]]])
            error_block_collection = []
            for id in keys(submodels)
                if get_az_species(basis_definition, id) == pair
                    error_block = slice_matrix.(errors, Ref(id), atomsv, Ref(basis_definition))
                    push!(error_block_collection, error_block)
                end
            end
            error_block_collection = [error_block_collection[i][j] for i in 1:length(error_block_collection)
                                        for j in 1:length(error_block_collection[i])]
            data_dict["MAE"][site]["Azimuthal_Species"][join([species,azimuthal])] = metrics(error_block_collection, "MAE")
            data_dict["RMSE"][site]["Azimuthal_Species"][join([species,azimuthal])] = metrics(error_block_collection, "RMSE")  
        end

    end

    return data_dict

end


# ------------------------------------
# |*************plotting*************| 
# ------------------------------------

function plot_save_single(x::T₁, y::T₂, x_label::String, y_label::String, label::Union{String, Matrix, Nothing},
                            output_path::String, basen::String) where {T₁, T₂}
    title = join([x_label, y_label, basen], "_")
    i = sortperm(x)
    x = x[i]
    if typeof(y)<:Vector{Float64}
        y = y[i]
    elseif typeof(y)<:Vector{Vector{Float64}}
        y = [y_sub[i] for y_sub in y]
    else
        error("the y is not in the right form")
    end
    plt = plot(x, y, title=title, xlabel=x_label, ylabel=y_label, label=label, legend_background_color=RGBA(1, 1, 1, 0))
    filename = joinpath(output_path, join([x_label, y_label, basen], "_")*".png")
    savefig(plt, filename)
end


function plot_hyperparams(data_dict::Dict, x_label::String, output_path_figs::String)

    mkpath(output_path_figs)

    x_label in ["d_max", "r_cut"] ? nothing : throw(AssertionError("the x_label should be either d_max or r_cut"))
    assess_type = collect(keys(data_dict))
    x = x_label == "d_max" ? [i[1] for i in assess_type] : [i[2] for i in assess_type]

    for y_label in ["MAE", "RMSE"]

        y_all = [i[y_label]["all"] for i in values(data_dict)]
        label = nothing
        basen = "all"
        plot_save_single(x, y_all, x_label, y_label, label, output_path_figs, basen)

        for site in ["on", "off"]
            y_all = [i[y_label][site]["all"] for i in values(data_dict)]
            label = nothing
            basen = join([site, "all"], "_")
            plot_save_single(x, y_all, x_label, y_label, label, output_path_figs, basen)
            
            for type in ["Azimuthal", "Azimuthal_Species"]
                # This step is merely to fix the typo in former get_error_dict function
                if type == "Azimuthal_Species"
                    type = haskey(collect(values(data_dict))[1][y_label][site], "Azimuthal_Species") ? "Azimuthal_Species" : "Azimuthal_SPecies"
                end
                y_type = [i[y_label][site][type] for i in values(data_dict)]
                label = reshape(collect(keys(y_type[1])), (1,:))
                y_type = [[y_type_sub[label_sub] for y_type_sub in y_type] for label_sub in label][:]
                basen = join([site, type], "_")
                plot_save_single(x, y_type, x_label, y_label, label, output_path_figs, basen)
            end

        end

    end

end


function plot_save_single_cross(x::T₁, y::T₂, x_label::String, y_label::String, label::Union{String, Matrix, Nothing},
                            output_path::String, basen::String) where {T₁, T₂}
    title = join([y_label, basen], "_")
    i = sortperm(x)
    x = x[i]
    if typeof(y)<:Vector{Float64}
        y = y[i]
    elseif typeof(y)<:Vector{Vector{Float64}}
        y = [y_sub[i] for y_sub in y]
    else
        error("the y is not in the right form")
    end
    plt = plot(x, y, title=title, xlabel=x_label, ylabel=y_label, label=label, legend_background_color=RGBA(1, 1, 1, 0))
    filename = joinpath(output_path, join([y_label, basen], "_")*".png")
    savefig(plt, filename)
end


function plot_cross(data_dict::Dict, output_path_figs::String)

    mkpath(output_path_figs)

    x_label = "number_of_water_molecules_for_prediction"
    assess_type = collect(keys(data_dict))
    model_sys_sizes = unique([i[1] for i in assess_type])   #datasize used for model training
    pred_sys_sizes = unique([i[2] for i in assess_type])   #system size for prediction

    for y_label in ["MAE", "RMSE"]

        y_all = [[data_dict[(model_sys_size, pred_sys_size)][y_label]["all"] for pred_sys_size in pred_sys_sizes]
                for model_sys_size in model_sys_sizes]

        label = reshape(model_sys_sizes, (1,:))
        basen = "all"
        plot_save_single_cross(pred_sys_sizes, y_all, x_label, y_label, label, output_path_figs, basen)

        for site in ["on", "off"]
            y_all = [[data_dict[(model_sys_size, pred_sys_size)][y_label][site]["all"] 
                    for pred_sys_size in pred_sys_sizes] for model_sys_size in model_sys_sizes]
            label = reshape(model_sys_sizes, (1,:))
            basen = join([site, "all"], "_")
            plot_save_single_cross(pred_sys_sizes, y_all, x_label, y_label, label, output_path_figs, basen)
            
            for type in ["Azimuthal", "Azimuthal_Species"]
                if type == "Azimuthal_Species"
                    type = haskey(collect(values(data_dict))[1][y_label][site], "Azimuthal_Species") ? "Azimuthal_Species" : "Azimuthal_SPecies"
                end
                for model_sys_size in model_sys_sizes
                    y_type = [data_dict[(model_sys_size, pred_sys_size)][y_label][site][type] 
                                for pred_sys_size in pred_sys_sizes]
                    label = reshape(collect(keys(y_type[1])), (1,:))
                    y_type = [[y_type_sub[label_sub] for y_type_sub in y_type] for label_sub in label][:]
                    basen = join([site, type, model_sys_size], "_")
                    plot_save_single_cross(pred_sys_sizes, y_type, x_label, y_label, label, output_path_figs, basen)
                end
            end

        end

    end

end




function plot_save_single_size(x::T₁, y::T₂, x_label::String, y_label::String, label::Union{String, Matrix, Nothing},
                            output_path::String, basen::String) where {T₁, T₂}
    title = join([y_label, basen], "_")
    i = sortperm(x)
    x = x[i]
    if typeof(y)<:Vector{Float64}
        y = y[i]
    elseif typeof(y)<:Vector{Vector{Float64}}
        y = [y_sub[i] for y_sub in y]
    else
        error("the y is not in the right form")
    end
    plt = plot(x, y, title=title, xlabel=x_label, ylabel=y_label, label=label, legend_background_color=RGBA(1, 1, 1, 0))
    filename = joinpath(output_path, join([y_label, basen], "_")*".png")
    savefig(plt, filename)
end


function plot_size(data_dict::Dict, output_path_figs::String)

    mkpath(output_path_figs)

    x = collect(keys(data_dict))
    x_label = "number of samples"

    for y_label in ["MAE", "RMSE"]

        y_all = [i[y_label]["all"] for i in values(data_dict)]
        label = nothing
        basen = "all"
        plot_save_single_size(x, y_all, x_label, y_label, label, output_path_figs, basen)

        for site in ["on", "off"]
            y_all = [i[y_label][site]["all"] for i in values(data_dict)]
            label = nothing
            basen = join([site, "all"], "_")
            plot_save_single_size(x, y_all, x_label, y_label, label, output_path_figs, basen)
            
            for type in ["Azimuthal", "Azimuthal_Species"]
                # This step is merely to fix the typo in former get_error_dict function
                if type == "Azimuthal_Species"
                    type = haskey(collect(values(data_dict))[1][y_label][site], "Azimuthal_Species") ? "Azimuthal_Species" : "Azimuthal_SPecies"
                end
                y_type = [i[y_label][site][type] for i in values(data_dict)]
                label = reshape(collect(keys(y_type[1])), (1,:))
                y_type = [[y_type_sub[label_sub] for y_type_sub in y_type] for label_sub in label][:]
                basen = join([site, type], "_")
                plot_save_single_size(x, y_type, x_label, y_label, label, output_path_figs, basen)
            end

        end

    end

end