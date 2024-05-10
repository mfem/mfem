import numpy as np
import matplotlib.pyplot as plt


ffprim = []
pprime = []
psizr = []
with open("separated_file.data", 'r') as fid:
    for line in fid:
        if "ffprim" in line:
            out = fid.readline()[:-2].split(" ")
            for num in out:
                ffprim.append(eval(num))
        if "pprime" in line:
            out = fid.readline()[:-2].split(" ")
            for num in out:
                pprime.append(eval(num))
        if "psizr" in line:
            out = fid.readline()[:-2].split(" ")
            for num in out:
                psizr.append(eval(num))

Nr = 257
Nz = 513

rmid = 6.5
rdim = 7
r0 = rmid - rdim / 2
r1 = rmid + rdim / 2
z0 = -6
z1 = 6

dr = (r1 - r0) / Nr
dz = (z1 - z0) / Nz

rv = np.linspace(r0, r1, Nr)
zv = np.linspace(z0, z1, Nz)

RV, ZV = np.meshgrid(rv, zv)

psi = np.zeros((Nz, Nr))
for j in range(Nz):
    for i in range(Nr):
        psi[j, i] = psizr[i + j * Nr]

ffprim = np.array(ffprim)
pprime = np.array(pprime)

psi_ma = -1.223975e+01
psi_x = 6.864813e-02

psi_n = (psi >= psi_ma) * (psi <= psi_x) * psi
psi_n = (psi_n - psi_ma) / (psi_x - psi_ma)

psi_n_p = np.linspace(0, 1, len(ffprim))

FFPRIM = np.interp(psi_n, psi_n_p, ffprim)
PPRIME = np.interp(psi_n, psi_n_p, pprime)

mu = 12.5663706144e-7

RHS = PPRIME * RV * mu + FFPRIM / RV
RHS = FFPRIM / RV
RHS = FFPRIM
RHS *= (psi >= psi_ma) * (psi <= psi_x) * (ZV >= -3.5) * (ZV <= 4.5)

Ip = - np.sum(RHS) * dr * dz / mu
print("plasma current: %f" % (Ip))

plt.figure()
plt.contourf(RV, ZV, psi, levels=1000)
plt.colorbar()
plt.title("psi")

plt.figure()
plt.contourf(RV, ZV, psi_n, levels=1000)
plt.colorbar()
plt.title("psi_n")

plt.figure()
plt.contourf(RV, ZV, RHS, levels=1000)
plt.colorbar()
plt.title("RHS")

plt.show()

breakpoint()
