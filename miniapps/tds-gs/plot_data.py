import numpy as np
import matplotlib.pyplot as plt

ffprim = []
pprim = []

with open("fpol_pres_ffprim_pprime.data", 'r') as fid:
    for line in fid:
        out = line.split(" ")
        ffprim.append(eval(out[2]))
        pprim.append(eval(out[3]))

plt.figure()
plt.plot(ffprim)
plt.ylabel("ffprime")
plt.xlabel("psi_N")
plt.figure()
plt.plot(pprim)
plt.ylabel("pprime")
plt.xlabel("psi_N")
plt.show()

