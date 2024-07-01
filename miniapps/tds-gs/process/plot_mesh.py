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

# meshname = "../meshes/iter_gen.msh"
meshname = "../gf/mesh_amr0.mesh"

plt.figure()
elements, vertices = get_mesh(meshname)
plot_mesh(elements, vertices, lw=.4)
plot_structures(lw=1.5)
plt.ylabel("$z$")
plt.xlabel("$r$")
plt.axis('square')
plt.xlim((0, 16))
plt.ylim((-16, 16))
plt.savefig("../figs/initial_mesh.png", dpi=200)
plt.show()

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
plt.axis('square')
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
