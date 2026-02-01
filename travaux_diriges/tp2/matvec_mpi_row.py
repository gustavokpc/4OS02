import numpy as np
from time import time
from mpi4py import MPI

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nbp = comm.Get_size()

    dim = 8000
    assert dim % nbp == 0, "dim doit être divisible par nbp"

    Nloc = dim // nbp
    i0 = rank * Nloc
    i1 = i0 + Nloc

    if rank == 0:
        u = np.array([i + 1.0 for i in range(dim)], dtype=np.float64)
    else:
        u = np.empty(dim, dtype=np.float64)
    comm.Bcast(u, root=0)

    comm.Barrier()
    t0 = time()

    # Construire seulement A[i0:i1, :]
    i = np.arange(i0, i1, dtype=np.int64)[:, None]        # (Nloc,1)
    j = np.arange(dim, dtype=np.int64)[None, :]           # (1,dim)
    A_loc = ((i + j) % dim + 1).astype(np.float64)        # (Nloc,dim)

    v_loc = A_loc @ u                                      # (Nloc,)

    comm.Barrier()
    t1 = time()

    v = np.empty(dim, dtype=np.float64)
    comm.Allgather(v_loc, v)

    t_local = t1 - t0
    t_max = comm.reduce(t_local, op=MPI.MAX, root=0)
    if rank == 0:
        print(f"[np={nbp}] dim={dim} Nloc={Nloc}  Temps(row)={t_max:.6f} s")
        # print("v (first 10) =", v[:10])

if __name__ == "__main__":
    main()
