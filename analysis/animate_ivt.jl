# Side-by-side animation: 12 km hindcast IVT vs ERA5, one frame per model output time
# (the model side carries the run's full cadence; ERA5, 3-hourly from the wide-region
# cache, holds between its own frames).
#
# Usage: julia --project=.. analysis/animate_ivt.jl   (AR_NAME selects the run; default pnw72)

using Oceananigans
using NumericalEarth
using NumericalEarth.DataWrangling.ERA5
using CopernicusClimateDataStore
using CairoMakie
using NaturalEarth, GeoInterface
using Printf
using Dates

root = joinpath(@__DIR__, "..")
run_name = get(ENV, "AR_NAME", "pnw72")
include(joinpath(root, "case.jl"))

ivt_east_series  = FieldTimeSeries(joinpath(root, run_name * "_ivt.jld2"), "ivt_east")
ivt_north_series = FieldTimeSeries(joinpath(root, run_name * "_ivt.jld2"), "ivt_north")
model_times = ivt_east_series.times
λ, φ, _ = nodes(ivt_east_series[1])
Nx, Ny = length(λ), length(φ)
margin = 25

single_levels = ERA5HourlySingleLevel()
era5_dates = filter(d -> start_date <= d <= stop_date, collect(viz_dates))
frame_dates = [start_date + Dates.Millisecond(round(Int, 1000t)) for t in model_times]
Nt = length(frame_dates)
era5_index(date) = argmin(abs.(Dates.value.(era5_dates .- date)))

metadata(name) = Metadata(name; dataset = single_levels, dates = era5_dates,
                          region = wide_region, dir = era5_datadir)
era5_east_series  = FieldTimeSeries(metadata(:vertically_integrated_eastward_water_vapor_flux);  time_indices_in_memory = length(era5_dates))
era5_north_series = FieldTimeSeries(metadata(:vertically_integrated_northward_water_vapor_flux); time_indices_in_memory = length(era5_dates))
λᵉ, φᵉ, _ = nodes(era5_east_series[1])
λᵉ = @. ifelse(λᵉ > 180, λᵉ - 360, λᵉ)

function model_ivt(n)
    east  = interior(ivt_east_series[n],  :, :, 1)
    north = interior(ivt_north_series[n], :, :, 1)
    ivt = @. sqrt(east^2 + north^2)
    ivt[1:margin, :] .= NaN; ivt[end-margin+1:end, :] .= NaN
    ivt[:, 1:margin] .= NaN; ivt[:, end-margin+1:end] .= NaN
    return ivt
end

function era5_ivt(n)
    east  = interior(era5_east_series[n],  :, :, 1)
    north = interior(era5_north_series[n], :, :, 1)
    return @. sqrt(east^2 + north^2)
end

coastλ, coastφ = natural_earth_lines("coastline")

frame = Observable(1)
modelₙ = Observable(model_ivt(1))
era5ₙ = Observable(era5_ivt(era5_index(frame_dates[1])))
on(frame) do n
    modelₙ[] = model_ivt(n)
    era5ₙ[] = era5_ivt(era5_index(frame_dates[n]))
end

fig = Figure(size = (1450, 520))
title = @lift "IVT — " * Dates.format(frame_dates[$frame], "u d HH:MM") * " UTC"
Label(fig[0, 1:3], title, fontsize = 18, font = :bold, tellwidth = false)

axᵐ = Axis(fig[1, 1]; title = "12 km hindcast", xlabel = "longitude (°)", ylabel = "latitude (°)")
axᵉ = Axis(fig[1, 2]; title = "ERA5", xlabel = "longitude (°)")
hm = heatmap!(axᵐ, λ, φ, modelₙ; colormap = :dense, colorrange = (0, 1200))
heatmap!(axᵉ, λᵉ, φᵉ, era5ₙ; colormap = :dense, colorrange = (0, 1200))
for ax in (axᵐ, axᵉ)
    lines!(ax, coastλ, coastφ; color = :black, linewidth = 0.5)
    xlims!(ax, longitude...); ylims!(ax, latitude...)
end
Colorbar(fig[1, 3], hm; label = "IVT (kg m⁻¹ s⁻¹)")

CairoMakie.record(fig, joinpath(root, "ivt_comparison.mp4"), 1:Nt; framerate = 12) do n
    frame[] = n
end
@info "IVT_ANIMATION_OK"
