import numpy as np
from dataclasses import dataclass
from PIL import Image
from math import log
from time import time
import matplotlib.cm
from mpi4py import MPI

# Tags MPI
WORK = 1
RESULT = 2
STOP = 3


@dataclass
class MandelbrotSet:
    max_iterations: int
    escape_radius: float = 2.0

    def convergence(self, c: complex, smooth=False, clamp=True) -> float:
        value = self.count_iterations(c, smooth) / self.max_iterations
        return max(0.0, min(value, 1.0)) if clamp else value

    def count_iterations(self, c: complex, smooth=False) -> int | float:
        if c.real*c.real + c.imag*c.imag < 0.0625:
            return self.max_iterations
        if (c.real+1)*(c.real+1) + c.imag*c.imag < 0.0625:
            return self.max_iterations
        if (-0.75 < c.real < 0.5):
            ct = c.real - 0.25 + 1j*c.imag
            ctnrm2 = abs(ct)
            if ctnrm2 < 0.5*(1 - ct.real/max(ctnrm2, 1e-14)):
                return self.max_iterations

        z = 0
        for it in range(self.max_iterations):
            z = z*z + c
            if abs(z) > self.escape_radius:
                if smooth:
                    return it + 1 - log(log(abs(z)))/log(2)
                return it
        return self.max_iterations


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    mandelbrot_set = MandelbrotSet(max_iterations=50, escape_radius=10)
    width, height = 1024, 1024
    scaleX = 3.0 / width
    scaleY = 2.25 / height

    # ================= MAÎTRE =================
    if rank == 0:
        convergence = np.empty((width, height), dtype=np.double)

        next_y = 0
        num_workers = size - 1

        comm.Barrier()
        t0 = time()

        # Envoi initial de travail
        for r in range(1, size):
            if next_y < height:
                comm.send(next_y, dest=r, tag=WORK)
                next_y += 1

        finished = 0

        while finished < num_workers:
            status = MPI.Status()
            y, line = comm.recv(source=MPI.ANY_SOURCE, tag=RESULT, status=status)
            src = status.Get_source()

            convergence[:, y] = line

            if next_y < height:
                comm.send(next_y, dest=src, tag=WORK)
                next_y += 1
            else:
                comm.send(None, dest=src, tag=STOP)
                finished += 1

        t1 = time()
        print(f"[np={size}] Temps calcul (maître–esclave) = {t1 - t0:.6f} s")

        image = Image.fromarray(np.uint8(matplotlib.cm.plasma(convergence.T) * 255))
        image.save("mandelbrot_mpi_q1_3.png")
        print("Saved: mandelbrot_mpi_q1_3.png")

    # ================= ESCLAVES =================
    else:
        comm.Barrier()

        while True:
            y = comm.recv(source=0, tag=MPI.ANY_TAG, status=MPI.Status())
            if y is None:
                break

            line = np.empty(width, dtype=np.double)
            for x in range(width):
                c = complex(-2.0 + scaleX * x, -1.125 + scaleY * y)
                line[x] = mandelbrot_set.convergence(c, smooth=True)

            comm.send((y, line), dest=0, tag=RESULT)


if __name__ == "__main__":
    main()

# Run:
# mpirun -np 2 python mandelbrot_ex1_3.py
# mpirun -np 4 python mandelbrot_ex1_3.py
# mpirun -np 8 python mandelbrot_ex1_3.py
