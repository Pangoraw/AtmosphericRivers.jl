# IVT animation for a 2-rank run: stitches the per-rank JLD2 slices (x-partitioned,
# rank 0 = west half) into the full domain per frame. Rind-masked like animate_ivt.jl.
#
# Usage: AR_NAME=pnw6km AR_DOMAIN=corridor AR_LAT_NORTH=57.5 julia --project=.. analysis/animate_ivt_stitched.jl

using NumericalEarth
using Oceananigans
using CairoMakie
using NaturalEarth, GeoInterface
using Printf

root = joinpath(@__DIR__, "..")
run_name = get(ENV, "AR_NAME", "pnw6km")
include(joinpath(root, "case.jl"))

east_series  = [FieldTimeSeries(joinpath(root, "$(run_name)_ivt_rank$(r).jld2"), "ivt_east")  for r in 0:1]
north_series = [FieldTimeSeries(joinpath(root, "$(run_name)_ivt_rank$(r).jld2"), "ivt_north") for r in 0:1]

# The ranks share a time axis but the crash can leave one file a frame ahead
Nt = minimum(length(fts.times) for fts in east_series)
times = east_series[1].times[1:Nt]

margin = parse(Int, get(ENV, "AR_MASK_MARGIN", "24"))

function stitched_ivt(n)
    east  = vcat((interior(fts[n], :, :, 1) for fts in east_series)...)
    north = vcat((interior(fts[n], :, :, 1) for fts in north_series)...)
    ivt = @. sqrt(east^2 + north^2)
    ivt[1:margin, :] .= NaN; ivt[end-margin+1:end, :] .= NaN
    ivt[:, 1:margin] .= NaN; ivt[:, end-margin+1:end] .= NaN
    return ivt
end

first_frame = stitched_ivt(1)
Nx_total, Ny_total = size(first_frame)
λ = range(longitude[1], longitude[2], length = Nx_total)
φ = range(latitude[1], latitude[2], length = Ny_total)
@info "Stitched domain $(Nx_total)x$(Ny_total), $(Nt) frames"

coastλ, coastφ = natural_earth_lines("coastline")

fig = Figure(size = (1100, 700))
n = Observable(1)
frame = Observable(first_frame)
on(nn -> frame[] = stitched_ivt(nn), n)

ax = Axis(fig[1, 1];
          title = @lift(@sprintf("%s IVT (kg m⁻¹ s⁻¹) — t = %.1f h after %s", run_name,
                                 times[$n] / 3600, start_date)),
          xlabel = "longitude (°)", ylabel = "latitude (°)", aspect = DataAspect())
hm = heatmap!(ax, λ, φ, frame; colormap = :dense, colorrange = (0, 1200))
lines!(ax, coastλ, coastφ; color = :black, linewidth = 0.5)
xlims!(ax, longitude...); ylims!(ax, latitude...)
Colorbar(fig[1, 2], hm)

CairoMakie.record(fig, joinpath(root, run_name * "_ivt.mp4"), 1:Nt; framerate = 10) do nn
    n[] = nn
end
@info "STITCHED_RENDER_OK"
