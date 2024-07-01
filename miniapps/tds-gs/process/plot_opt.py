import numpy as np
import matplotlib.pyplot as plt
from utils import get_psi, get_mesh, plot_structures, plot_solution, plot_mesh
plt.rcParams.update({'font.size': 16,
                     'text.usetex' : True})

gfnames = ['../opt_study/opt_old_.gf',
           '../opt_study/opt_new.gf']
meshnames = ['../opt_study/mesh_old_.mesh',
             '../opt_study/mesh_new.gf']

gfnames = ['../gf/final_model2_pc5_cyc0_it10.gf']
meshnames = ['../meshes/mesh_refine.mesh']


psi_xs = [ 9.668901e+01]


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

rmid = 6.5
rdim = 7
r0 = rmid - rdim / 2
r1 = rmid + rdim / 2
z0 = -6
z1 = 6

for gf, mesh, psi_x in zip(gfnames, meshnames, psi_xs):
    print(gf)
    plt.figure()
    psi = get_psi(gf)
    elements, vertices = get_mesh(mesh)
    plot_solution(psi, elements, vertices, psi_x=psi_x)
    plot_structures(lw=1)
    plt.plot(rbbs[::dN], zbbs[::dN], '.k')
    # plt.colorbar()
    plt.xlim((r0, r1))
    plt.ylim((z0, z1))
    
plt.show()
    
