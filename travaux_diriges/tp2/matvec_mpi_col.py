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
    j0 = rank * Nloc
    j1 = j0 + Nloc

    if rank == 0:
        u = np.array([i + 1.0 for i in range(dim)], dtype=np.float64)
    else:
        u = np.empty(dim, dtype=np.float64)
    comm.Bcast(u, root=0)

    u_loc = u[j0:j1]

    comm.Barrier()
    t0 = time()

    # A[i,j] = (i+j) % dim + 1
    i = np.arange(dim, dtype=np.int64)[:, None]          # (dim,1)
    j = np.arange(j0, j1, dtype=np.int64)[None, :]       # (1,Nloc)
    A_loc = ((i + j) % dim + 1).astype(np.float64)       # (dim,Nloc)

    # Contribution partielle : v_part = A_loc * u_loc
    v_part = A_loc @ u_loc                                # (dim,)

    comm.Barrier()
    t1 = time()

    # Somme des contributions de tous les processus
    v = np.empty(dim, dtype=np.float64)
    comm.Allreduce(v_part, v, op=MPI.SUM)

    t_local = t1 - t0
    t_max = comm.reduce(t_local, op=MPI.MAX, root=0)
    if rank == 0:
        print(f"[np={nbp}] dim={dim} Nloc={Nloc}  Temps(col)={t_max:.6f} s")
        # print("v (first 10) =", v[:10])

if __name__ == "__main__":
    main()
