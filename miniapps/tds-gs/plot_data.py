import numpy as np
import matplotlib.pyplot as plt

ffprim = []
pprim = []
rzbbs = []
with open("fpol_pres_ffprim_pprime.data", 'r') as fid:
    for line in fid:
        out = line.split(" ")
        ffprim.append(eval(out[2]))
        pprim.append(eval(out[3]))

with open("separated_file.data", 'r') as fid:
    for line in fid:
        if "rbbbs" in line:
            out = fid.readline()[:-2].split(" ")
            for num in out:
                rzbbs.append(eval(num))
                
rzbbs = np.array(rzbbs)
rbbbs = rzbbs[::2]
zbbbs = rzbbs[1::2]
plt.figure()
plt.plot(rbbbs, zbbbs)
plt.show()
breakpoint()
        
plt.figure()
plt.plot(ffprim)
plt.ylabel("ffprime")
plt.xlabel("psi_N")
plt.figure()
plt.plot(pprim)
plt.ylabel("pprime")
plt.xlabel("psi_N")
plt.show()

