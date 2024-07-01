import numpy as np
import matplotlib.pyplot as plt
from utils import get_psi, get_mesh, plot_structures, plot_solution, plot_mesh, plot_filled_solution
import sys
sys.path.append("../geometry/.")

plt.rcParams.update({'font.size': 16,
                     'text.usetex' : True})

rmid = 6.5
rdim = 7
r0 = rmid - rdim / 2
r1 = rmid + rdim / 2
z0 = -6
z1 = 6

meshnames = ["../amr_study/taylor_amr0.gf",
             "../amr_study/taylor_amr1.gf",
             "../amr_study/taylor_amr2.gf",
             "../amr_study/taylor_amr3.gf",
             "../amr_study/taylor_amr4.gf"]
meshnames_f = ["../amr_study/taylor_f_amr0.gf",
               "../amr_study/taylor_f_amr1.gf",
               "../amr_study/taylor_f_amr2.gf",
               "../amr_study/taylor_f_amr3.gf",
               "../amr_study/taylor_f_amr4.gf"]
filename = "../amr_study/f.gf"

rz_bbs = []
with open("../data/separated_file.data", 'r') as fid:
    for line in fid:
        if "rbbbs(i),zbbbs(i)" in line:
            out = fid.readline()[:-2].split(" ")
            for num in out:
                rz_bbs.append(eval(num))
rz_bbs = np.array(rz_bbs)
rbbs = rz_bbs[::2]
zbbs = rz_bbs[1::2]


plt.figure(figsize=(12, 5))
for i, meshname in enumerate(meshnames_f):
    ax = plt.subplot(1, len(meshnames), i+1)
    elements, vertices = get_mesh(meshname)
    plot_mesh(elements, vertices, lw=.4)
    plot_structures(lw=2)
    plt.plot(rbbs, zbbs, 'r', linewidth=.8)
    plt.xlim((r0, r1))
    plt.ylim((z0, z1))
    if i > 0:
        plt.gca().set_yticklabels([""] * 7)
    else:
        plt.ylabel("$z$")
    plt.xlabel("$r$")
    ax.annotate("%d" % (i), xy=(0.5,1.05), xycoords='axes fraction',
                size=12,
                bbox=dict(boxstyle="round", fc=(.1, .1, 1, .3), ec=(0,0,0)), ha='center')
    if i == 2:
        ax.annotate("Adaptive Mesh Refinement around Plasma Boundary", xy=(.5,1.15), xycoords='axes fraction',
                    size=16,
                    bbox=dict(boxstyle="round", fc=(0, 0, 1, .0), ec=(0, 0, 0, 0)), ha='center')

plt.savefig("../figs/taylor_amr_plasma.png", dpi=400)

f = get_psi(filename)
elements, vertices = get_mesh(meshname)
fig = plt.figure(figsize=(8,5))
ax = plt.subplot(1, 2, 1)
plot_filled_solution(f, elements, vertices)
plot_structures(lw=1)
plt.plot([r0, r1, r1, r0, r0],
         [z0, z0, z1, z1, z0],
         'r', linewidth=1)
plt.xlim((0, 16))
plt.ylim((-16, 16))
plt.xlabel("$r$")
plt.ylabel("$z$")
# ax.annotate("Taylor Equilibrium Solution", xy=(1.4,1.05), xycoords='axes fraction',
#             size=16,
#             bbox=dict(boxstyle="round", fc=(0, 0, 1, .0), ec=(0, 0, 0, 0)), ha='center')


####

plt.figure(figsize=(12, 5))
for i, meshname in enumerate(meshnames):
    ax = plt.subplot(1, len(meshnames), i+1)
    elements, vertices = get_mesh(meshname)
    plot_mesh(elements, vertices, lw=.4)
    plot_structures(lw=2)
    plt.xlim((r0, r1))
    plt.ylim((z0-4, z1-4))
    if i > 0:
        plt.gca().set_yticklabels([""] * 7)
    else:
        plt.ylabel("$z$")
    plt.xlabel("$r$")
    ax.annotate("%d" % (i), xy=(0.5,1.05), xycoords='axes fraction',
                size=12,
                bbox=dict(boxstyle="round", fc=(.1, .1, 1, .3), ec=(0,0,0)), ha='center')
    if i == 2:
        ax.annotate("Adaptive Mesh Refinement for Global Domain", xy=(.5,1.15), xycoords='axes fraction',
                    size=16,
                    bbox=dict(boxstyle="round", fc=(0, 0, 1, .0), ec=(0, 0, 0, 0)), ha='center')
plt.savefig("../figs/taylor_amr_global.png", dpi=400)
        

plt.show()




breakpoint()
