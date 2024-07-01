import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../geometry/.")
from create_geometry import _VacuumVesselMetalWallCoordinates, _VacuumVesselFirstWallCoordinates, _VacuumVesselSecondWallCoordinates
from utils import get_psi, get_mesh, plot_structures, plot_solution, plot_mesh

plt.rcParams.update({'font.size': 16,
                     'text.usetex' : True})


Nr = 257
Nz = 513

psi_true_ = []
rz_bbs = []
ffprim = []
pprim = []
fpol = []
pres = []
with open("../data/separated_file.data", 'r') as fid:
    for line in fid:
        if "rbbbs(i),zbbbs(i)" in line:
            out = fid.readline()[:-2].split(" ")
            for num in out:
                rz_bbs.append(eval(num))
        if "psizr" in line:
            out = fid.readline()[:-2].split(" ")
            for num in out:
                psi_true_.append(eval(num))
        if "ffprim" in line:
            out = fid.readline()[:-2].split(" ")
            for num in out:
                ffprim.append(eval(num))
        if "fpol" in line:
            out = fid.readline()[:-1].split(" ")
            for num in out:
                fpol.append(eval(num))
        if "pres" in line:
            out = fid.readline()[:-1].split(" ")
            for num in out:
                try:
                    pres.append(eval(num))
                except:
                    breakpoint()

rz_bbs = np.array(rz_bbs)

rmid = 6.5
rdim = 7
r0 = rmid - rdim / 2
r1 = rmid + rdim / 2
z0 = -6
z1 = 6

rv = np.linspace(r0, r1, Nr)
zv = np.linspace(z0, z1, Nz)

psi_true = np.zeros((Nz, Nr))
for j in range(Nz):
    for i in range(Nr):
        psi_true[j, i] = psi_true_[i + j * Nr]

RV, ZV = np.meshgrid(rv, zv)


rv = rz_bbs[0::2]
zv = rz_bbs[1::2]

fig = plt.figure(figsize=(15, 5))
ax = plt.subplot(1, 3, 1)
plt.contour(RV, ZV, psi_true, 80, alpha=0.5, cmap=plt.jet())
rbbs = rz_bbs[::2]
zbbs = rz_bbs[1::2]
N = len(rbbs)
dN = N // 100
plt.plot(rbbs[::dN], zbbs[::dN], '.k')
plot_structures(lw=2)
ax.annotate("Initial $\psi$", xy=(0.5,1.05), xycoords='axes fraction',
            size=14,
            bbox=dict(boxstyle="round", fc=(0, 0, 1, .2), ec=(0, 0, 0)), ha='center')
plt.axis('scaled')
plt.colorbar()
plt.xlim((r0, r1))
plt.ylim((z0, z1))
plt.xlabel("$r$")
plt.ylabel("$z$")
plt.xticks([4, 6, 8, 10])

ax = plt.subplot(1, 3, 2)
x = np.linspace(0, 1, len(fpol))
plt.plot(x, fpol)
plt.xlabel("$\psi_{N}$")
ax.annotate("$f(\psi_{N})$", xy=(0.5,1.05), xycoords='axes fraction',
            size=14,
            bbox=dict(boxstyle="round", fc=(0, 0, 1, .2), ec=(0, 0, 0)), ha='center')
ax.annotate("Initial Solution", xy=(0.5,1.2), xycoords='axes fraction',
            size=18,
            bbox=dict(boxstyle="round", fc=(0, 0, 1, .0), ec=(0, 0, 0, 0)), ha='center')

ax = plt.subplot(1, 3, 3)
x = np.linspace(0, 1, len(pres))
plt.plot(x, pres)
plt.xlabel("$\psi_{N}$")
ax.annotate("$p(\psi_{N})$", xy=(0.5,1.05), xycoords='axes fraction',
            size=14,
            bbox=dict(boxstyle="round", fc=(0, 0, 1, .2), ec=(0, 0, 0)), ha='center')
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

fig.subplots_adjust(wspace=0.3, hspace=.2, top=.8)


plt.savefig("../figs/initial.png", dpi=200)
plt.show()
