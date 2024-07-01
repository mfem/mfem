import numpy as np
import matplotlib.pyplot as plt
from utils import get_psi, get_mesh, plot_structures, plot_solution, plot_mesh
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

filenames = ["../gf/final_model2_pc5_cyc1_it5.gf",
             "../gf/final_model1_pc0_cyc0_it1.gf",
             "../gf/final_model4_pc5_cyc0_it5.gf"]
meshnames = ["../gf/mesh_amr4_model2_pc5_cyc1_it5.mesh",
             "../gf/mesh_amr1_model1_pc0_cyc0_it1.mesh",
             "../gf/mesh_amr3_model4_pc5_cyc0_it5.mesh"]
titles = ["Taylor Equilibrium Solution",
          "15MA ITER",
          "Luxon and Brown Solution"]
names = ['taylor_sol',
         'empirical',
         'lb']

rz_bbs = []
with open("../data/separated_file.data", 'r') as fid:
    for line in fid:
        if "rbbbs(i),zbbbs(i)" in line:
            out = fid.readline()[:-2].split(" ")
            for num in out:
                rz_bbs.append(eval(num))
rz_bbs = np.array(rz_bbs)
rv = rz_bbs[0::2]
zv = rz_bbs[1::2]
rbbs = rz_bbs[::2]
zbbs = rz_bbs[1::2]
N = len(rbbs)
dN = N // 100

for filename, meshname, title, name in zip(filenames, meshnames, titles, names):
    psi = get_psi(filename)
    elements, vertices = get_mesh(meshname)

    fig = plt.figure(figsize=(8,5))
    ax = plt.subplot(1, 2, 1)
    plot_solution(psi, elements, vertices, do_colorbar=False)
    plt.plot(rbbs[::dN], zbbs[::dN], 'k.')
    plot_structures(lw=1)
    plt.plot([r0, r1, r1, r0, r0],
             [z0, z0, z1, z1, z0],
             'r', linewidth=1)
    plt.xlim((0, 16))
    plt.ylim((-16, 16))
    plt.xlabel("$r$")
    plt.ylabel("$z$")
    ax.annotate(title, xy=(1.4,1.05), xycoords='axes fraction',
                size=16,
                bbox=dict(boxstyle="round", fc=(0, 0, 1, .0), ec=(0, 0, 0, 0)), ha='center')

    plt.subplot(1, 2, 2)
    im = plot_solution(psi, elements, vertices, do_colorbar=False)
    plt.plot(rbbs[::dN], zbbs[::dN], 'k.')
    plot_structures(lw=1.5)
    plt.xlim((r0, r1))
    plt.ylim((z0, z1))
    plt.xlabel("$r$")
    plt.ylabel("$z$")

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    clb = fig.colorbar(im, cax=cbar_ax)
    clb.ax.set_title('$\psi$')

    plt.savefig("../figs/%s.png" % (name), dpi=200)


plt.show()



breakpoint()
