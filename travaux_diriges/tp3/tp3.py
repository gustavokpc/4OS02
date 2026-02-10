import numpy as np
import time
from mpi4py import MPI
import sys
from math import sqrt, log2

out  = None
DEBUG= 0

globCom = MPI.COMM_WORLD.Dup()
nbp     = globCom.size
rank    = globCom.rank
name    = MPI.Get_processor_name()

N = 256_000

if len(sys.argv) > 1:
    N = int(sys.argv[1])

filename = f"Output{rank:03d}.txt"
out      = open(filename, mode='w')

reste= N % nbp
assert(reste == 0)
NLoc = N//nbp
print(f"Nombre de valeurs locales : {NLoc}`\n")

# Génération du tableau local de valeurs
values = np.random.randint(-32768, 32768, size=NLoc,dtype=np.int64)
print(f"Valeurs initiales : {values}\n")
print(N)  
print(len(values))

for value in values:
    
