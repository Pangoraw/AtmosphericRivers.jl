# Case configuration shared by the predownload, visualization, and downscaling scripts.
#
# The December 2025 Pacific Northwest atmospheric-river family: an AR 4 (locally
# forecast AR 5) made landfall over Washington and Oregon early on December 8, a second
# pulse arrived late December 9, and AR conditions persisted ~96 hours — record crests
# on the Skagit, Snohomish, and Cedar rivers and >250 mm over the Olympics and Cascades.

using Dates: DateTime, Hour

# Simulation window: the first landfall plus the second pulse of the Dec 8–12 AR.
# Override the duration with AR_HOURS (e.g. AR_HOURS=96 to reach the event's full tail).
start_date = DateTime(2025, 12, 7, 12)
event_hours = parse(Int, get(ENV, "AR_HOURS", "72"))
stop_date = start_date + Hour(event_hours)
dates = (start_date, stop_date)

# Child domain: the moisture corridor from east of Hawaii to the coast, the Gulf of
# Alaska parent low, and every landfall from BC to Northern California — plus the
# Pacific Northwest terrain (Olympics, Cascades, coastal ranges). Kept east of the
# dateline so one ERA5 subregion request covers the parent.
longitude = (-170, -110)
latitude = (25, 60)

# ~12 km grid spacing: 2.25× refinement over ERA5's native 0.25°
Δ = 1/9
Nx = round(Int, (longitude[2] - longitude[1]) / Δ)
Ny = round(Int, (latitude[2] - latitude[1]) / Δ)

era5_datadir = joinpath(@__DIR__, "era5")

# Wide North Pacific view for the ERA5 visualization (single-level IVT + MSLP),
# kept east of the dateline so one ERA5 subregion request covers it.
wide_region = BoundingBox(longitude = (-180, -110), latitude = (20, 62))
viz_dates = DateTime(2025, 12, 6):Hour(3):DateTime(2025, 12, 20)

# Landfall precipitation box over the Dec 8–12 core; hourly, since ERA5's
# total_precipitation is accumulated over each hour.
precip_region = BoundingBox(longitude = (-130, -115), latitude = (40, 52))
precip_dates = DateTime(2025, 12, 7):Hour(1):DateTime(2025, 12, 12)
