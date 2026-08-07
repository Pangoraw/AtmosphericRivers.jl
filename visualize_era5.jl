# # The December 2025 Pacific Northwest atmospheric-river family in ERA5
#
# Between December 3 and 20, 2025, a family of five to six atmospheric rivers made
# landfall on the Pacific Northwest coast. The destructive core, December 8–12, was an
# AR 4 on the Ralph et al. (2019) scale (forecasts gave >50% odds of AR 5 on the coast):
# ~96 hours of continuous AR conditions over Washington, peak integrated vapor transport
# (IVT) near 1000 kg m⁻¹ s⁻¹ offshore, more than 250 mm over the Olympics and Cascades
# with snow levels at 6,000–9,000 ft, and record crests on the Skagit, Snohomish, and
# Cedar rivers. A Gulf of Alaska low paired with a subtropical high southwest of
# California anchored the corridor; the moisture traced ~11,000 km back to the western
# Pacific, routed past Hawaii.
#
# References:
# - CW3E event summary: https://cw3e.ucsd.edu/cw3e-event-summary-8-12-december-2025/
# - UW state climatologist: https://climate.uw.edu/2026/01/13/december-8-11-2025-heavy-rainfall-and-flooding-historical-context-and-a-note-on-snow-drought/
#
# This script visualizes the event directly from ERA5 single-level fields — no model:
# the AR plume as IVT magnitude (from the vertically integrated vapor-flux components)
# with sea-level pressure contours, an animation of the full family, the IVT time series
# at the Washington coast against the AR-scale thresholds, and the accumulated
# landfall precipitation. Run `slurm/predownload.batch` first (or let the data download
# on the fly on a node with internet access).

using NumericalEarth
using NumericalEarth.DataWrangling.ERA5
using CopernicusClimateDataStore
using Oceananigans
using CairoMakie
using NaturalEarth, GeoInterface   # together → NumericalEarthNaturalEarthExt (`natural_earth_lines`)
using Printf
using Dates

include("case.jl")

single_levels = ERA5HourlySingleLevel()
Nt = length(viz_dates)

# ## Load the wide-view fields
#
# ERA5 computes the vapor-flux integrals on its native model levels, so the two
# single-level components give IVT without downloading any pressure-level data.

metadata(name) = Metadata(name; dataset = single_levels, dates = viz_dates,
                          region = wide_region, dir = era5_datadir)

ivt_east_series  = FieldTimeSeries(metadata(:vertically_integrated_eastward_water_vapor_flux);  time_indices_in_memory = Nt)
ivt_north_series = FieldTimeSeries(metadata(:vertically_integrated_northward_water_vapor_flux); time_indices_in_memory = Nt)
pressure_series  = FieldTimeSeries(metadata(:mean_sea_level_pressure); time_indices_in_memory = Nt)

# The native ERA5 grid is labeled 0–360; wrap the longitudes back to ±180 for plotting.
λ, φ, _ = nodes(ivt_east_series[1])
λ_plot = @. ifelse(λ > 180, λ - 360, λ)

function ivt_magnitude(n)
    east  = interior(ivt_east_series[n],  :, :, 1)
    north = interior(ivt_north_series[n], :, :, 1)
    return @. sqrt(east^2 + north^2)
end

sea_level_pressure_hPa(n) = interior(pressure_series[n], :, :, 1) ./ 100

# Coastlines and the downscaling child domain, overlaid on every map.
coastλ, coastφ = natural_earth_lines("coastline")
boxλ = [longitude[1], longitude[2], longitude[2], longitude[1], longitude[1]]
boxφ = [latitude[1], latitude[1], latitude[2], latitude[2], latitude[1]]

function ivt_panel!(ax, n)
    hm = heatmap!(ax, λ_plot, φ, ivt_magnitude(n); colormap = :dense, colorrange = (0, 1200))
    contour!(ax, λ_plot, φ, sea_level_pressure_hPa(n);
             levels = 940:8:1044, color = (:black, 0.4), linewidth = 0.7)
    lines!(ax, coastλ, coastφ; color = :black, linewidth = 0.5)
    lines!(ax, boxλ, boxφ; color = :crimson, linestyle = :dash, linewidth = 1.5)
    xlims!(ax, wide_region.longitude...)
    ylims!(ax, wide_region.latitude...)
    return hm
end

# ## Key moments of the AR family

key_moments = [DateTime(2025, 12, 8, 6)   => "Dec 8 06Z — first landfall (AR 4)",
               DateTime(2025, 12, 9, 18)  => "Dec 9 18Z — second pulse",
               DateTime(2025, 12, 14, 18) => "Dec 14 18Z — Pineapple Express",
               DateTime(2025, 12, 18, 6)  => "Dec 18 06Z — third major AR"]

fig = Figure(size = (1300, 750))
Label(fig[0, 1:2], "ERA5 IVT (kg m⁻¹ s⁻¹) and sea-level pressure — December 2025 AR family",
      fontsize = 18, font = :bold, tellwidth = false)

panels = map(enumerate(key_moments)) do (i, (moment, title))
    row, col = fldmod1(i, 2)
    ax = Axis(fig[row, col]; title, xlabel = "longitude (°)", ylabel = "latitude (°)")
    n = findfirst(==(moment), viz_dates)
    ivt_panel!(ax, n)
end
Colorbar(fig[1:2, 3], last(panels); label = "IVT (kg m⁻¹ s⁻¹)")

save("ivt_key_moments.png", fig)

# ## Animation of the full family

frame = Observable(1)
ivtₙ = Observable(ivt_magnitude(1))
pressureₙ = Observable(sea_level_pressure_hPa(1))
on(frame) do n
    ivtₙ[] = ivt_magnitude(n)
    pressureₙ[] = sea_level_pressure_hPa(n)
end

fig_movie = Figure(size = (1000, 560))
title = @lift "ERA5 IVT — " * Dates.format(viz_dates[$frame], "u d HH:MM") * " UTC"
ax = Axis(fig_movie[1, 1]; title, xlabel = "longitude (°)", ylabel = "latitude (°)")

hm_movie = heatmap!(ax, λ_plot, φ, ivtₙ; colormap = :dense, colorrange = (0, 1200))
contour!(ax, λ_plot, φ, pressureₙ; levels = 940:8:1044, color = (:black, 0.4), linewidth = 0.7)
lines!(ax, coastλ, coastφ; color = :black, linewidth = 0.5)
lines!(ax, boxλ, boxφ; color = :crimson, linestyle = :dash, linewidth = 1.5)
xlims!(ax, wide_region.longitude...)
ylims!(ax, wide_region.latitude...)
Colorbar(fig_movie[1, 2], hm_movie; label = "IVT (kg m⁻¹ s⁻¹)")

CairoMakie.record(fig_movie, "north_pacific_ivt.mp4", 1:Nt; framerate = 10) do n
    frame[] = n
end

# ## IVT at the Washington coast against the AR-scale thresholds
#
# The Ralph et al. (2019) AR scale combines peak IVT with duration; the IVT thresholds
# alone (250/500/750/1000/1250 kg m⁻¹ s⁻¹) already tell the story of the family.

coastal_point = (-124.6, 47.9)   # Quillayute, WA coast
i = argmin(abs.(λ_plot .- coastal_point[1]))
j = argmin(abs.(φ .- coastal_point[2]))
coastal_ivt = [ivt_magnitude(n)[i, j] for n in 1:Nt]

t_seconds = Dates.value.(viz_dates .- first(viz_dates)) ./ 1000
day_dts = first(viz_dates):Day(1):last(viz_dates)
day_ticks = Dates.value.(day_dts .- first(viz_dates)) ./ 1000
day_labels = Dates.format.(day_dts, dateformat"u d")

fig_series = Figure(size = (1000, 420))
ax_series = Axis(fig_series[1, 1];
                 title = "ERA5 IVT at $(coastal_point[2])°N, $(-coastal_point[1])°W (Washington coast)",
                 ylabel = "IVT (kg m⁻¹ s⁻¹)",
                 xticks = (day_ticks[1:2:end], day_labels[1:2:end]))

for (threshold, label) in zip(250:250:1250, ["AR 1", "AR 2", "AR 3", "AR 4", "AR 5"])
    hlines!(ax_series, threshold; color = (:gray, 0.5), linestyle = :dash)
    text!(ax_series, t_seconds[end], threshold; text = label, align = (:right, :bottom),
          color = :gray, fontsize = 11)
end
lines!(ax_series, t_seconds, coastal_ivt; color = :dodgerblue, linewidth = 2)

save("coastal_ivt_series.png", fig_series)

# ## Accumulated landfall precipitation, December 7–12
#
# `total_precipitation` is accumulated over each hour and converted to a mass flux
# (kg m⁻² s⁻¹) at load time, so multiplying each hourly sample by 3600 s and summing
# gives the event total in kg m⁻² ≡ mm.

precip_meta = Metadata(:total_precipitation; dataset = single_levels, dates = precip_dates,
                       region = precip_region, dir = era5_datadir)
precip_series = FieldTimeSeries(precip_meta; time_indices_in_memory = length(precip_dates))

total_precipitation = zeros(size(interior(precip_series[1], :, :, 1)))
for n in 1:length(precip_dates)
    total_precipitation .+= interior(precip_series[n], :, :, 1) .* 3600
end

λᵖ, φᵖ, _ = nodes(precip_series[1])
λᵖ_plot = @. ifelse(λᵖ > 180, λᵖ - 360, λᵖ)

fig_precip = Figure(size = (800, 640))
ax_precip = Axis(fig_precip[1, 1];
                 title = "ERA5 precipitation, Dec 7–12 2025 (mm)",
                 xlabel = "longitude (°)", ylabel = "latitude (°)")
hm_precip = heatmap!(ax_precip, λᵖ_plot, φᵖ, total_precipitation;
                     colormap = :dense, colorrange = (0, 350))
lines!(ax_precip, coastλ, coastφ; color = :black, linewidth = 0.5)
xlims!(ax_precip, precip_region.longitude...)
ylims!(ax_precip, precip_region.latitude...)
Colorbar(fig_precip[1, 2], hm_precip; label = "mm")

save("event_precipitation.png", fig_precip)

@info "Wrote ivt_key_moments.png, north_pacific_ivt.mp4, coastal_ivt_series.png, event_precipitation.png"
