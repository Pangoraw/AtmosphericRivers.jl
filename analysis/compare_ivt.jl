# Compare the hindcast's column-integrated vapor transport against ERA5's own
# vertically-integrated vapor-flux product at matched times, inside the analysis
# region (the outer `margin` cells — relaxation + terrain blend — are masked).
#
# Usage: julia --project=.. analysis/compare_ivt.jl   (AR_NAME selects the run; default pnw72)

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
times = ivt_east_series.times
λ, φ, _ = nodes(ivt_east_series[1])
Nx, Ny = length(λ), length(φ)
margin = 25

single_levels = ERA5HourlySingleLevel()
region = BoundingBox(; longitude, latitude)

function model_ivt(n)
    east  = interior(ivt_east_series[n],  :, :, 1)
    north = interior(ivt_north_series[n], :, :, 1)
    return @. sqrt(east^2 + north^2)
end

function era5_ivt(date)
    east  = Field(Metadatum(:vertically_integrated_eastward_water_vapor_flux;
                            dataset = single_levels, date, region, dir = era5_datadir))
    north = Field(Metadatum(:vertically_integrated_northward_water_vapor_flux;
                            dataset = single_levels, date, region, dir = era5_datadir))
    e = interior(east, :, :, 1)
    v = interior(north, :, :, 1)
    magnitude = @. sqrt(e^2 + v^2)
    return magnitude, nodes(east)
end

coastλ, coastφ = natural_earth_lines("coastline")

# Landfall (Dec 8 06Z, t = 18 h) and the second pulse (Dec 9 18Z, t = 54 h).
comparison_hours = (18, 54)

fig = Figure(size = (1400, 950))
Label(fig[0, 1:3], "IVT (kg m⁻¹ s⁻¹): 12 km hindcast vs ERA5 — December 2025 AR",
      fontsize = 18, font = :bold, tellwidth = false)

for (row, hour) in enumerate(comparison_hours)
    n = argmin(abs.(times .- hour * 3600))
    date = start_date + Dates.Second(round(Int, times[n]))
    ivtᵐ = model_ivt(n)
    ivtᵐ[1:margin, :] .= NaN; ivtᵐ[end-margin+1:end, :] .= NaN
    ivtᵐ[:, 1:margin] .= NaN; ivtᵐ[:, end-margin+1:end] .= NaN
    ivtᵉ, era5_nodes = era5_ivt(date)
    λᵉ, φᵉ, _ = era5_nodes
    λᵉ = @. ifelse(λᵉ > 180, λᵉ - 360, λᵉ)

    axᵐ = Axis(fig[row, 1]; title = "$(Dates.format(date, "u d HH:MM")) — 12 km model",
               xlabel = "longitude (°)", ylabel = "latitude (°)")
    axᵉ = Axis(fig[row, 2]; title = "ERA5", xlabel = "longitude (°)")
    hm = heatmap!(axᵐ, λ, φ, ivtᵐ; colormap = :dense, colorrange = (0, 1200))
    heatmap!(axᵉ, λᵉ, φᵉ, ivtᵉ; colormap = :dense, colorrange = (0, 1200))
    for ax in (axᵐ, axᵉ)
        lines!(ax, coastλ, coastφ; color = :black, linewidth = 0.5)
        xlims!(ax, longitude...); ylims!(ax, latitude...)
    end
    row == 1 && Colorbar(fig[1:2, 3], hm; label = "IVT (kg m⁻¹ s⁻¹)")

    inner = filter(!isnan, ivtᵐ)
    @printf("t=%3d h (%s): model max IVT (interior) = %6.1f, ERA5 max = %6.1f kg/m/s\n",
            hour, date, maximum(inner), maximum(ivtᵉ))
end

save(joinpath(root, "ivt_model_vs_era5.png"), fig)
@info "IVT_COMPARISON_OK"
