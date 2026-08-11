# Render a completed run's animations from its saved slices: the maps panel (IVT,
# surface wind, w and rain aloft) and the vertical section along 47.5°N. Kept out of
# downscale.jl so simulation jobs never pay the Makie load.
#
# Usage: julia --project=.. analysis/render_run.jl   (AR_NAME selects the run; default pnw72)

using Oceananigans
using NumericalEarth
using CairoMakie
using NaturalEarth, GeoInterface
using Printf

root = joinpath(@__DIR__, "..")
run_name = get(ENV, "AR_NAME", "pnw72")
include(joinpath(root, "case.jl"))

surface_filename = joinpath(root, run_name * "_surface.jld2")
aloft_filename   = joinpath(root, run_name * "_aloft.jld2")
section_filename = joinpath(root, run_name * "_section.jld2")
ivt_filename     = joinpath(root, run_name * "_ivt.jld2")

# ## Maps animation

ivt_east_series  = FieldTimeSeries(ivt_filename, "ivt_east")
ivt_north_series = FieldTimeSeries(ivt_filename, "ivt_north")
U_series  = FieldTimeSeries(surface_filename, "U")
w_series  = FieldTimeSeries(aloft_filename, "w")
qʳ_series = FieldTimeSeries(aloft_filename, "qʳ")
times = U_series.times

coastλ, coastφ = natural_earth_lines("coastline")
g_per_kg(field) = 1f3 * field

panels = [(title = "IVT (kg m⁻¹ s⁻¹)", colormap = :dense,   colorrange = (0, 1200),
           field = n -> sqrt(ivt_east_series[n]^2 + ivt_north_series[n]^2)),
          (title = "|U|ₛ (m s⁻¹)",     colormap = :speed,   colorrange = (0, 40),
           field = n -> U_series[n]),
          (title = "w₂ₖₘ (m s⁻¹)",     colormap = :balance, colorrange = (-3, 3),
           field = n -> w_series[n]),
          (title = "qʳ₂ₖₘ (g kg⁻¹)",   colormap = :dense,   colorrange = (0, 1.5),
           field = n -> g_per_kg(qʳ_series[n]))]

fig_maps = Figure(size = (1250, 700))
maps_n = Observable(1)

Label(fig_maps[0, 1:2],
      @lift(@sprintf("%s — December 2025 PNW atmospheric river — t = %.1f h",
                     run_name, times[$maps_n] / 3600)),
      fontsize = 18, tellwidth = false)

for (i, panel) in enumerate(panels)
    row, col = fldmod1(i, 2)
    sub = fig_maps[row, col] = GridLayout()
    ax = Axis(sub[1, 1]; title = panel.title, aspect = DataAspect())

    ## Type-erased observable: per-frame lazy operations drift in concrete type,
    ## which a @lift-typed Observable rejects at the second frame.
    frame = Observable{Any}(panel.field(1))
    on(n -> frame[] = panel.field(n), maps_n)

    hm = heatmap!(ax, frame; colormap = panel.colormap, colorrange = panel.colorrange)
    lines!(ax, coastλ, coastφ; color = :black, linewidth = 0.5)
    xlims!(ax, longitude...)
    ylims!(ax, latitude...)
    hidedecorations!(ax)
    Colorbar(sub[1, 2], hm)
end

CairoMakie.record(fig_maps, joinpath(root, run_name * "_maps.mp4"), eachindex(times); framerate = 10) do nn
    maps_n[] = nn
end

# ## Section animation

U_section  = FieldTimeSeries(section_filename, "U")
w_section  = FieldTimeSeries(section_filename, "w")
qᵛ_section = FieldTimeSeries(section_filename, "qᵛ")

section_n = Observable(1)
Uₙ  = Observable{Any}(U_section[1])
wₙ  = Observable{Any}(w_section[1])
qᵛₙ = Observable{Any}(g_per_kg(qᵛ_section[1]))
on(section_n) do n
    Uₙ[]  = U_section[n]
    wₙ[]  = w_section[n]
    qᵛₙ[] = g_per_kg(qᵛ_section[n])
end

fig_section = Figure(size = (1000, 900))

Label(fig_section[0, 1:2],
      @lift(@sprintf("Vertical structure at 47.5°N — t = %.1f h", times[$section_n] / 3600)),
      fontsize = 18, tellwidth = false)

ax_U = Axis(fig_section[1, 1]; ylabel = "z (m)", title = "|U| (m s⁻¹)")
ax_w = Axis(fig_section[2, 1]; ylabel = "z (m)", title = "w (m s⁻¹)")
ax_q = Axis(fig_section[3, 1]; ylabel = "z (m)", xlabel = "longitude (°)", title = "qᵛ (g kg⁻¹)")

hm_U = heatmap!(ax_U, Uₙ;  colormap = :speed,   colorrange = (0, 50))
hm_w = heatmap!(ax_w, wₙ;  colormap = :balance, colorrange = (-3, 3))
hm_q = heatmap!(ax_q, qᵛₙ; colormap = :dense,   colorrange = (0, 12))

Colorbar(fig_section[1, 2], hm_U)
Colorbar(fig_section[2, 2], hm_w)
Colorbar(fig_section[3, 2], hm_q)

CairoMakie.record(fig_section, joinpath(root, run_name * "_section.mp4"), eachindex(times); framerate = 10) do nn
    section_n[] = nn
end
@info "RENDER_OK"
