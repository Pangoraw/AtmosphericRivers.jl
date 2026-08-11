# NCCLDistributed halo-exchange test: 2 MPI ranks, 1 A100 each, x-partitioned grid.
# Each rank fills its interior with a rank-identifying constant; after
# fill_halo_regions! the x halos must hold the neighbor's constant (Periodic x, so
# both of a rank's x halos come from its peer), while y/z halos stay local.
#
# Launch: mpiexec -n 2 julia --project=. test_nccl_halo.jl   (precompile first!)

using MPI
using CUDA
using NCCL
using Oceananigans
using Oceananigans.DistributedComputations: NCCLDistributed
using Oceananigans.BoundaryConditions: fill_halo_regions!

arch = NCCLDistributed(GPU(); partition = Partition(2))
rank = arch.local_rank
peer = 1 - rank
@info "Rank $rank up on $(CUDA.device())"

grid = RectilinearGrid(arch,
                       size = (64, 32, 8),
                       x = (0, 1), y = (0, 1), z = (0, 1),
                       halo = (3, 3, 3),
                       topology = (Periodic, Periodic, Bounded))

c = CenterField(grid)
set!(c, rank + 1)
fill_halo_regions!(c)

data = Array(parent(c))
Hx, Nx = 3, size(grid, 1)

west_halo = data[1:Hx, Hx+1:end-Hx, Hx+1:end-Hx]
east_halo = data[Hx+Nx+1:Nx+2Hx, Hx+1:end-Hx, Hx+1:end-Hx]
interior_values = data[Hx+1:Hx+Nx, Hx+1:end-Hx, Hx+1:end-Hx]

@assert all(interior_values .== rank + 1) "Rank $rank: interior corrupted"
@assert all(west_halo .== peer + 1) "Rank $rank: west halo = $(west_halo[1]), expected $(peer + 1)"
@assert all(east_halo .== peer + 1) "Rank $rank: east halo = $(east_halo[1]), expected $(peer + 1)"
@info "Rank $rank: x halos hold neighbor constant $(peer + 1) — exchange correct"

MPI.Barrier(arch.communicator)
rank == 0 && println("NCCL_HALO_OK")
