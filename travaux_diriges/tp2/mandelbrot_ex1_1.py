import numpy as np
from dataclasses import dataclass
from PIL import Image
from math import log
from time import time
import matplotlib.cm
from mpi4py import MPI


@dataclass
class MandelbrotSet:
    max_iterations: int
    escape_radius: float = 2.0

    def convergence(self, c: complex, smooth=False, clamp=True) -> float:
        value = self.count_iterations(c, smooth) / self.max_iterations
        return max(0.0, min(value, 1.0)) if clamp else value

    def count_iterations(self, c: complex, smooth=False) -> int | float:
        # zones de convergence connues (comme dans mandelbrot.py)
        if c.real * c.real + c.imag * c.imag < 0.0625:
            return self.max_iterations
        if (c.real + 1) * (c.real + 1) + c.imag * c.imag < 0.0625:
            return self.max_iterations
        if (-0.75 < c.real < 0.5):
            ct = c.real - 0.25 + 1.j * c.imag
            ctnrm2 = abs(ct)
            if ctnrm2 < 0.5 * (1 - ct.real / max(ctnrm2, 1e-14)):
                return self.max_iterations

        z = 0
        for it in range(self.max_iterations):
            z = z * z + c
            if abs(z) > self.escape_radius:
                if smooth:
                    return it + 1 - log(log(abs(z))) / log(2)
                return it
        return self.max_iterations


def block_rows(rank: int, size: int, H: int) -> tuple[int, int]:
    """Répartition statique équilibrée en blocs contigus."""
    base = H // size
    r = H % size
    y0 = rank * base + min(rank, r)
    nloc = base + (1 if rank < r else 0)
    return y0, nloc


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # paramètres (comme dans mandelbrot.py)
    mandelbrot_set = MandelbrotSet(max_iterations=50, escape_radius=10)
    width, height = 1024, 1024
    scaleX = 3.0 / width
    scaleY = 2.25 / height

    # chaque processus calcule un bloc de lignes
    y0, nloc = block_rows(rank, size, height)

    # buffer local: (width, nloc)
    local = np.empty((width, nloc), dtype=np.double)

    comm.Barrier()
    t0 = time()

    for j in range(nloc):
        y = y0 + j
        for x in range(width):
            c = complex(-2.0 + scaleX * x, -1.125 + scaleY * y)
            local[x, j] = mandelbrot_set.convergence(c, smooth=True)

    comm.Barrier()
    t1 = time()

    # rassembler sur le processus 0
    sendcount = width * nloc
    counts = comm.gather(sendcount, root=0)

    if rank == 0:
        recvbuf = np.empty(width * height, dtype=np.double)
        displs = np.zeros(size, dtype=int)
        displs[1:] = np.cumsum(counts[:-1])
    else:
        recvbuf = None
        displs = None

    # on envoie en ordre Fortran (colonnes) pour garder un reshape simple
    comm.Gatherv(local.ravel(order="F"), (recvbuf, counts, displs, MPI.DOUBLE), root=0)

    if rank == 0:
        # reconstruire convergence (width, height)
        convergence = np.empty((width, height), dtype=np.double)
        offset = 0
        for rnk in range(size):
            yy0, nn = block_rows(rnk, size, height)
            chunk = recvbuf[offset: offset + width * nn].reshape((width, nn), order="F")
            convergence[:, yy0:yy0 + nn] = chunk
            offset += width * nn

        print(f"[np={size}] Temps calcul (compute only): {t1 - t0:.6f} s")

        # image + sauvegarde (pas show)
        image = Image.fromarray(np.uint8(matplotlib.cm.plasma(convergence.T) * 255))
        image.save("mandelbrot_mpi_q1.png")
        print("Saved: mandelbrot_mpi_q1.png")


if __name__ == "__main__":
    main()


# Run ex1_1 with:
# mpirun -np 1 python mandelbrot_ex1_1.py
# mpirun -np 2 python mandelbrot_ex1_1.py
# mpirun -np 4 python mandelbrot_ex1_1.py
# mpirun -np 8 python mandelbrot_ex1_1.py
