# Slurm policy

Run every job that loads the Julia stack on `gpua100largex4` (or the `...spot`
partition — same instance type). The cluster's node types have different CPU
microarchitectures (login = AMD Rome, A100 = Intel Cascade Lake, gpudev = its
own), Julia precompile caches are microarch-specific, and `~/.julia` is
NFS-shared: one partition means one warm cache. A stack job on another
partition silently re-precompiles everything (~40 min) before doing any work.
