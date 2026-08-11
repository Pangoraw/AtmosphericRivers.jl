# Staged 2-rank probe for the scale2 segfault: the full downscale.jl import set,
# then MPI init, NCCL arch construction, and a halo exchange — with a marker after
# each stage so the crash point localizes from the log alone.

using NumericalEarth, Breeze, CopernicusClimateDataStore, CloudMicrophysics, RRTMGP
using Oceananigans, CUDA, NCCL, MPI
using Oceananigans.DistributedComputations: NCCLDistributed
using Oceananigans.BoundaryConditions: fill_halo_regions!

println("STAGE1 imports done"); flush(stdout)

MPI.Init()
println("STAGE2 MPI.Init OK rank=", MPI.Comm_rank(MPI.COMM_WORLD)); flush(stdout)

arch = NCCLDistributed(GPU(); partition = Partition(2))
println("STAGE3 arch OK rank=", arch.local_rank, " dev=", CUDA.device()); flush(stdout)

grid = RectilinearGrid(arch,
                       size = (64, 32, 8),
                       x = (0, 1), y = (0, 1), z = (0, 1),
                       halo = (3, 3, 3),
                       topology = (Periodic, Periodic, Bounded))
c = CenterField(grid)
set!(c, arch.local_rank + 1)
fill_halo_regions!(c)
println("STAGE4 halo OK rank=", arch.local_rank); flush(stdout)

MPI.Barrier(arch.communicator)
arch.local_rank == 0 && println("NODE_PROBE_OK")
