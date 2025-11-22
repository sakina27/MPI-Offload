# mpi-offload-demo

Mini benchmark to understand three MPI communication strategies:

1. **baseline** – single thread does both computation and MPI.
2. **comm_self** – extra thread stays inside MPI using a blocking recv on
   `MPI_COMM_SELF` (classic "comm-self" progress approach).
3. **offload** – dedicated offload thread owns all MPI calls; compute thread
   sends commands via a queue (similar to the paper's offload approach).

The code uses a simple 1D stencil with halo exchange between neighbors.
test
## Build

```bash
make
