# # ERA5 → 12 km hindcast of the December 2025 Pacific Northwest atmospheric river
#
# A limited-area Breeze compressible atmosphere downscales ERA5 to ~12 km over the
# northeast Pacific and the Pacific Northwest — the last ~1700 km of the approach of the
# AR 4 that made landfall early December 8, 2025, plus the terrain (Olympics, Cascades,
# coastal ranges) that turned its vapor transport into record flooding.
#
# `nested_atmosphere_model(grid, dataset; dates, …)` builds the nest: an ERA5 parent
# `PrescribedAtmosphere` on its native 0.25° pressure-level grid drives the child through
# open lateral boundaries + Davies relaxation, with ETOPO2022 terrain blended to the
# parent's orography at the boundary and a dynamical-initialization pass balancing the
# initial state. The lower boundary is a `PrescribedOcean` carrying an ERA5 SST snapshot,
# coupled through Monin–Obukhov similarity fluxes, with all-sky RRTMGP radiation.
#
# ## v0 caveats (in scale-up order)
# - The SST is a static snapshot: `PrescribedOcean` does not yet interpolate multi-time
#   data (fine for ~3 days of ocean memory; wrong for weeks).
# - Coupling a `NestedModel` atmosphere to an ocean is an untested code path (the nested
#   `AtmosphereLandModel` is proven; `AtmosphereOceanModel` uses the same forwarding).
# - Land cells get the ERA5 skin temperature under *ocean-type* fluxes: Charnock
#   roughness over the Cascades until mixed land+ocean tile assembly exists upstream.
# - Near-surface θᵛ drifts cold over ocean in multi-day nests (NumericalEarth issue
#   #457, boundary-sourced) — watch the surface θᵛ output.

using NumericalEarth
using Oceananigans
using Oceananigans.Units
using Breeze
using CopernicusClimateDataStore # ERA5 + CGLS-albedo downloads
using CloudMicrophysics          # nested_atmosphere_model's default 1-moment mixed-phase microphysics
using RRTMGP                     # Breeze's radiative-transfer extension
using CairoMakie
using NaturalEarth, GeoInterface # `natural_earth_lines` coastlines
using CUDA
using Printf
using Dates: DateTime
using Oceananigans.Advection: AdaptiveVerticallyImplicitDiscretization

include("case.jl")

## AR_RANKS=2 partitions the domain in longitude across MPI ranks, one GPU each —
## a throughput experiment: output writers and rendering stay single-rank only.
ranks = parse(Int, get(ENV, "AR_RANKS", "1"))
arch = ranks > 1 ? Distributed(GPU(); partition = Partition(ranks)) : GPU()
Oceananigans.defaults.FloatType = Float32

name = get(ENV, "AR_NAME", "pnw")
smoke = get(ENV, "AR_SMOKE", "0") == "1"

# ## LAM grid
#
# Vertical grid as in the MC3E example: 50 cells, constant 60 m surface spacing,
# 490 m maximum, top at ~20 km. Horizontal: 324×162 at Δ = 1/9° ≈ 12 km — about
# 2.6M cells, a few GB on an A100-40GB (the first scale-up is Δ = 1/18° ≈ 6 km,
# which still fits without code changes).

z = ReferenceToStretchedDiscretization(extent = 19525.0,
                                       bias = :left,
                                       bias_edge = 0,
                                       constant_spacing = 60,
                                       constant_spacing_extent = 60,
                                       maximum_spacing = 490,
                                       stretching = LinearStretching(0.15))

Nz = length(z)

grid = LatitudeLongitudeGrid(arch;
                             longitude, latitude,
                             z = TerrainFollowingVerticalDiscretization(z),
                             size = (Nx, Ny, Nz),
                             halo = (5, 5, 5),
                             topology = (Bounded, Bounded, Bounded))

# ## Build the nest
#
# The Davies relaxation is an explicit nudge, so its rate is a fixed 1/300 s⁻¹ rather
# than tied to the adaptive Δt; WENO(5) momentum interacts less violently with the
# open-boundary halo than the default WENO(9). No `bottom_drag_coefficient`: surface
# stress, heat, and moisture come from the ocean coupling below.

dataset = ERA5HourlyPressureLevels()
relax_width = 5

# AR_AIVA=1 gives every prognostic adaptively implicit vertical advection, dropping the
# vertical advective CFL from the wizard's timescale so Δt can climb to the horizontal
# limit. Rain/snow sedimentation stays explicit — watch qʳ as Δt grows.
aiva = get(ENV, "AR_AIVA", "0") == "1"

breeze_extension = Base.get_extension(NumericalEarth, :NumericalEarthBreezeExt)
explicit_scalar_advection =
    breeze_extension.default_nested_scalar_advection(breeze_extension.default_nested_microphysics())

if aiva
    implicit_vertical = AdaptiveVerticallyImplicitDiscretization(cfl = 0.5)
    momentum_advection = WENO(order = 5, time_discretization = implicit_vertical)
    scalar_names = keys(explicit_scalar_advection)
    scalar_advection = NamedTuple{scalar_names}(map(scalar_names) do name
        name === :ρθ ? WENO(order = 5, time_discretization = implicit_vertical) :
                       WENO(order = 5, bounds = (0, 1), time_discretization = implicit_vertical)
    end)
else
    momentum_advection = WENO(order = 5)
    scalar_advection = explicit_scalar_advection
end

nest = nested_atmosphere_model(grid, dataset;
                               dates,
                               dir = era5_datadir,
                               terrain = ETOPO2022(),
                               terrain_blend_width = relax_width,
                               relaxation_rate = 1/300,
                               relaxation_width = relax_width,
                               momentum_advection,
                               scalar_advection)

parent_atmosphere = nest.parent
era5_region = BoundingBox(parent_atmosphere.grid)

if ranks == 1
    fig = visualize_nested_domain(grid;
                                  parent = era5_region,
                                  padding = 2.5,
                                  title = "ERA5 → 12 km LAM nest (December 2025 PNW atmospheric river)",
                                  label = "12 km LAM (child)",
                                  parent_label = "ERA5 parent",
                                  landmarks = ("Seattle" => (-122.3, 47.6),
                                               "Quillayute" => (-124.6, 47.9)))
    save("pnw_domains.png", fig)
end

# ## Prescribed ocean surface
#
# A `PrescribedOcean` on the child's horizontal grid carries the ERA5 sea surface
# temperature at the start date (in Kelvin). ERA5's SST is NaN over land, so land cells
# fall back to the ERA5 skin temperature — a sane surface temperature, though it feels
# ocean-type (Charnock) roughness until mixed land+ocean tiles exist upstream.

ocean_grid = LatitudeLongitudeGrid(arch;
                                   longitude, latitude,
                                   size = (Nx, Ny),
                                   halo = (5, 5),
                                   topology = (Bounded, Bounded, Flat))

single_levels = ERA5HourlySingleLevel()
sst_metadatum  = Metadatum(:sea_surface_temperature; dataset = single_levels,
                           date = start_date, region = era5_region, dir = era5_datadir)
skin_metadatum = Metadatum(:skin_temperature; dataset = single_levels,
                           date = start_date, region = era5_region, dir = era5_datadir)

sea_surface_temperature = Field{Center, Center, Nothing}(ocean_grid)
skin_temperature = Field{Center, Center, Nothing}(ocean_grid)
set!(sea_surface_temperature, sst_metadatum)
set!(skin_temperature, skin_metadatum)

Tˢ = interior(sea_surface_temperature)
Tᵏ = interior(skin_temperature)
@. Tˢ = ifelse(isnan(Tˢ), Tᵏ, Tˢ)

ocean = PrescribedOcean(ocean_grid)
interior(ocean.sea_surface_temperature) .= interior(sea_surface_temperature)

# ## RRTMGP radiation
#
# All-sky RRTMGP: interior heating on the evolving clouds, per-column sun angles, and
# the CGLS 1 km blue-sky albedo (water pixels default to open water — most of this
# domain). Hourly solves match the ERA5 cadence.

radiation = RadiativeTransferModel(grid, AllSkyOptics(), nest.child.thermodynamic_constants;
                                   solar_position = ApparentSolarPosition(epoch = start_date),
                                   surface_albedo = CopernicusAlbedo(),
                                   schedule = TimeInterval(1hour))

# ## Coupled model
#
# `AtmosphereOceanModel` wires nest, ocean, and radiation together: each step computes
# Monin–Obukhov fluxes from the child's lowest-cell state and the prescribed SST, writing
# them into the child's bottom flux boundary conditions. The adaptive outer Δt is bounded
# by the advective CFL (acoustic modes are substepped).

Δt = 10
maximum_Δt = parse(Float64, get(ENV, "AR_MAX_DT", "10"))   ## wizard cap; probe with AR_MAX_DT=30
cfl = parse(Float64, get(ENV, "AR_CFL", "0.5"))            ## wizard target; probe throughput with 0.7
atmosphere = Simulation(nest; Δt)   ## the coupled model manages Δt; this sets only the initial value
model = AtmosphereOceanModel(atmosphere, ocean; radiation)

simulation = Simulation(model; Δt, stop_time = event_hours * hours)
conjure_time_step_wizard!(simulation, IterationInterval(3); cfl, max_Δt = maximum_Δt)

smoke && (simulation.stop_iteration = 30)

# ## Output
#
# 2-D slices of shared online diagnostics — horizontal slices at the surface and ~2 km,
# and a zonal x-z section at 47.5°N (through the Quillayute coast and the Olympics) —
# plus the AR fingerprint: column-integrated vapor transport as `Integral` operations,
# already 2-D so they get their own writer.

child = nest.child
k_aloft = searchsortedfirst(Array(znodes(grid, Center())), 2000)
j_section = searchsortedfirst(Array(φnodes(grid, Center(), Center(), Center())), 47.5)

u, v, w = child.velocities
ρ = child.dynamics.total_density
qᵛ = specific_humidity(child)

fields = (θᵛ = VirtualPotentialTemperature(child),
          U  = sqrt(u^2 + v^2),
          w  = w,
          qᵛ = qᵛ,
          qʳ = child.microphysical_fields.qʳ)

ivt_fields = (ivt_east  = Integral(ρ * qᵛ * u, dims = 3),
              ivt_north = Integral(ρ * qᵛ * v, dims = 3),
              precipitable_water = Integral(ρ * qᵛ, dims = 3))

surface_filename = name * "_surface.jld2"
aloft_filename   = name * "_aloft.jld2"
section_filename = name * "_section.jld2"
ivt_filename     = name * "_ivt.jld2"

schedule = TimeInterval(30minutes)
slice_writer(indices, filename) = JLD2Writer(model, fields; schedule, filename, indices,
                                             overwrite_existing = true)

output = ranks == 1 && get(ENV, "AR_OUTPUT", "1") == "1"   ## AR_OUTPUT=0: benchmark mode, no writers

if output   ## distributed JLD2 output is untested; multi-rank runs measure stepping only
    simulation.output_writers[:surface] = slice_writer((:, :, 1),         surface_filename)
    simulation.output_writers[:aloft]   = slice_writer((:, :, k_aloft),   aloft_filename)
    simulation.output_writers[:section] = slice_writer((:, j_section, :), section_filename)
    simulation.output_writers[:ivt] = JLD2Writer(model, ivt_fields; schedule,
                                                 filename = ivt_filename,
                                                 overwrite_existing = true)
end

previous_wall_time = Ref(time_ns())

function progress(sim)
    wall_seconds = 1e-9 * (time_ns() - previous_wall_time[])
    previous_wall_time[] = time_ns()
    child = sim.model.atmosphere.model.child
    u, v, w = child.velocities
    ρ  = child.dynamics.total_density
    qᵛ = specific_humidity(child)
    qʳ = child.microphysical_fields.qʳ
    @info @sprintf("iter=%4d, t=%s, Δt=%s, wall=%6.1fs, max|u|=(%7.2f, %7.2f, %6.2f), ρ ∈ [%.4f, %.4f], qᵛ ∈ [%.4g, %.4g], qʳ ∈ [%.2g, %.2g]",
                   sim.model.clock.iteration, prettytime(sim), prettytime(sim.Δt), wall_seconds,
                   maximum(abs, u), maximum(abs, v), maximum(abs, w), minimum(ρ), maximum(ρ),
                   minimum(qᵛ), maximum(qᵛ), minimum(qʳ), maximum(qʳ))
    return nothing
end

add_callback!(simulation, progress, IterationInterval(100))

# ## Run

run!(simulation)

(smoke || !output) && exit(0)

# ## Maps animation
#
# IVT magnitude (the AR itself), surface wind speed, and vertical velocity + rain at
# ~2 km. The saved slices load back as `FieldTimeSeries` and plot directly — lazy field
# operations (the IVT magnitude, unit conversions) recompute per frame.

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
      @lift(@sprintf("ERA5 → 12 km Breeze — December 2025 PNW atmospheric river — t = %.1f h",
                     times[$maps_n] / 3600)),
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

CairoMakie.record(fig_maps, name * "_maps.mp4", eachindex(times); framerate = 10) do nn
    maps_n[] = nn   ## `record` is also exported by CUDA, so qualify it
end
nothing #hide

# ## Section animation
#
# The vertical structure along 47.5°N: the low-level jet and moist tongue feeding the
# orographic precipitation, and the mountain-wave response over the Olympics and Cascades.

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

CairoMakie.record(fig_section, name * "_section.mp4", eachindex(times); framerate = 10) do nn
    section_n[] = nn
end
nothing #hide
