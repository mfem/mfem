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
        if "pprim" in line:
            out = fid.readline()[:-2].split(" ")
            for num in out:
                pprim.append(eval(num))

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
plt.subplot(1, 3, 1)
plt.contour(RV, ZV, psi_true, 80, alpha=0.5, cmap=plt.jet())
rbbs = rz_bbs[::2]
zbbs = rz_bbs[1::2]
N = len(rbbs)
dN = N // 100
plt.plot(rbbs[::dN], zbbs[::dN], '.k')
plot_structures(lw=2)
plt.title("Initial $\psi$")
plt.axis('scaled')
plt.colorbar()
plt.xlim((r0, r1))
plt.ylim((z0, z1))
plt.xlabel("$r$")
plt.ylabel("$z$")
plt.xticks([4, 6, 8, 10])

plt.subplot(1, 3, 2)
x = np.linspace(0, 1, len(ffprim))
plt.plot(x, ffprim)
plt.xlabel("$\psi_{N}$")
plt.title("$S_{ff'}(\psi_{N})$")

plt.subplot(1, 3, 3)
x = np.linspace(0, 1, len(pprim))
plt.plot(x, pprim)
plt.xlabel("$\psi_{N}$")
plt.title("$S_{p'}(\psi_{N})$")
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

fig.subplots_adjust(wspace=0.3, hspace=.2)


plt.savefig("../figs/initial.png", dpi=200)
plt.show()
