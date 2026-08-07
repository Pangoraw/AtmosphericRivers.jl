# AtmosphericRivers.jl

Downscaling the **December 2025 Pacific Northwest atmospheric-river family** with
[NumericalEarth.jl](https://github.com/NumericalEarth/NumericalEarth.jl):
ERA5 reanalysis drives a [Breeze.jl](https://github.com/NumericalEarth/Breeze.jl)
limited-area compressible atmosphere through open lateral boundaries, over a prescribed
ERA5 sea surface and ETOPO2022 terrain, with all-sky RRTMGP radiation.

## The event

Between December 3 and 20, 2025, five to six atmospheric rivers made landfall on the
Pacific Northwest coast. The core event (Dec 8–12) was an AR 4 on the Ralph et al. scale:
~96 hours of continuous AR conditions over Washington, peak IVT near 1000 kg m⁻¹ s⁻¹
offshore, >250 mm over the Olympics and Cascades with snow levels at 6,000–9,000 ft, and
record crests on the Skagit, Snohomish, and Cedar rivers; ~100,000 people were placed
under evacuation orders. A Gulf of Alaska low paired with a subtropical high anchored the
corridor; the moisture traced back ~11,000 km to the western Pacific.

- [CW3E event summary, 8–12 December 2025](https://cw3e.ucsd.edu/cw3e-event-summary-8-12-december-2025/)
- [UW state climatologist retrospective](https://climate.uw.edu/2026/01/13/december-8-11-2025-heavy-rainfall-and-flooding-historical-context-and-a-note-on-snow-drought/)
- [2025 Pacific Northwest floods (Wikipedia)](https://en.wikipedia.org/wiki/2025_Pacific_Northwest_floods)

## Contents

| File | Purpose |
|---|---|
| `case.jl` | Shared configuration: dates, domain, resolution, data directory |
| `predownload.jl` | Fetch all ERA5 files up front (CDS queue is the bottleneck, not bytes) |
| `visualize_era5.jl` | Model-free ERA5 view of the event: IVT maps + animation, coastal AR-scale time series, landfall precipitation |
| `downscale.jl` | ERA5 → 12 km Breeze nest with prescribed-SST ocean coupling + RRTMGP |
| `slurm/*.batch` | Batch scripts (instantiate, predownload, visualize, downscale) |

## Workflow

```bash
cd ~/AtmosphericRivers.jl
sbatch slurm/instantiate.batch                       # resolve + precompile on an A100 node
sbatch slurm/predownload.batch                       # ~2 GB of ERA5, hours of CDS queue
sbatch slurm/visualize.batch                         # script 1: ERA5 event visualization
sbatch --export=ALL,AR_SMOKE=1 slurm/downscale.batch # 30-iteration GPU smoke test
sbatch slurm/downscale.batch                         # script 2: the 72 h hindcast
```

Requires CDS credentials (`~/.cdsapirc`, mirrored to `~/.config/era5cli/cds_key.txt`)
with the ERA5 single-level and pressure-level licences accepted.

NumericalEarth is sourced by path from `~/NumericalEarth.jl` (branch `glw/cleanup`), which
must include the ERA5 single-level table updates (Dec 2025 date range + the
`tcwv`/`viwve`/`viwvn` IVT variables). Breeze is pinned to `glw/rrtmgp-nested-integration`.

## Configuration (v0: one A100-40GB)

- **Domain**: 148°W–112°W, 38°N–56°N — the AR's last ~1700 km of approach plus the
  Olympics, Cascades, coastal ranges, and Vancouver Island.
- **Resolution**: Δ = 1/9° ≈ 12 km, 324×162×50 ≈ 2.6M cells (Float32, ~4 GB total on GPU).
- **Window**: 2025-12-07T12 + 72 h (override with `AR_HOURS`), covering the first
  landfall and the second pulse of the Dec 8–12 AR.

## Scale-up ladder

1. **6 km** (Δ = 1/18°) on the same domain — no code changes, ~10M cells.
2. **Time-varying SST**: `PrescribedOcean` currently reads only the first time index —
   needs upstream temporal interpolation.
3. **Mixed land+ocean surface**: flux assembly currently assumes one surface type per
   cell (land would overwrite ocean); needs upstream tile-fraction assembly, then a
   `SlabLand` coast under the same child. Snow does not yet reach the land bucket.
4. **Wider domain** toward the full North Pacific (genesis-to-landfall crosses the
   dateline): needs the `time_indices_in_memory` passthrough in the high-level
   `nested_atmosphere_model` (the parent currently holds every date resident on GPU),
   and dateline-aware ERA5 requests.

## Known risks (v0)

- `AtmosphereOceanModel` around a `NestedModel` atmosphere is an untested path (the
  nested `AtmosphereLandModel` is proven; the component forwarding is shared).
- Land cells carry ERA5 skin temperature under ocean-type (Charnock) roughness.
- Near-surface θᵛ drifts cold over ocean in multi-day nests
  ([NumericalEarth#457](https://github.com/NumericalEarth/NumericalEarth.jl/issues/457)) —
  watch the surface θᵛ output.
