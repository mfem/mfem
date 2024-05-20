import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../geometry/.")
from create_geometry import _VacuumVesselMetalWallCoordinates, _VacuumVesselSecondWallCoordinates, _VacuumVesselFirstWallCoordinates


def get_psi(filename):
    fid = open(filename, "r")
    for n in range(5):
        fid.readline()
    out = fid.read().split('\n')[:-1]
    psi = np.array([eval(a) for a in out])
    return psi
    

def get_mesh(meshname):
    fid = open(meshname, "r")
    for line in fid:
        if "elements" in line:
            NE = eval(fid.readline())
            elements = []
            while True:
                lin = fid.readline()
                if lin == "\n":
                    break
                nums = lin[:-1].split(" ")
                nums = [eval(a) for a in nums]
                elements.append(nums)
            elements = np.array(elements, dtype=int)
        if "vertices" in line:
            NV = eval(fid.readline())
            fid.readline()
            vertices = []
            while True:
                lin = fid.readline()
                if lin == "\n" or lin == "":
                    break
                nums = lin[:-1].split(" ")
                try:
                    nums = [eval(a) for a in nums]
                except:
                    breakpoint()
                vertices.append(nums)
            vertices = np.array(vertices)


    return elements, vertices

def plot_structures(lw=1.5):
    
    r, z = _VacuumVesselSecondWallCoordinates(1)
    r = np.concatenate([r, [r[0]]])
    z = np.concatenate([z, [z[0]]])
    plt.plot(r, z, 'k', alpha=1, linewidth=lw)

    r, z = _VacuumVesselFirstWallCoordinates(1)
    r = np.concatenate([r, [r[0]]])
    z = np.concatenate([z, [z[0]]])
    plt.plot(r, z, 'k', alpha=1, linewidth=lw)
    
    r, z = _VacuumVesselMetalWallCoordinates(1)
    r = np.concatenate([r, [r[0]]])
    z = np.concatenate([z, [z[0]]])
    plt.plot(r, z, 'k', alpha=1, linewidth=lw)

    r0 = [1.696, 1.696, 1.696, 1.696, 1.696,
          3.9431, 8.2851, 11.9919, 11.9630, 8.3908, 4.3340]
    z0 = [-5.415, -3.6067, -1.7983, 1.8183, 3.6267,
          7.5741, 6.5398, 3.2752, -2.2336, -6.7269, -7.4665]

    r_sn = 1.5
    z_sn = 1
    r_c = 2
    z_c = 1.5

    dr = [ r_sn, r_sn, r_sn, r_sn, r_sn,
           r_c, r_c, r_c, r_c, r_c, r_c]
    dz = [ z_sn, z_sn, z_sn, z_sn, z_sn,
           z_c, z_c, z_c, z_c, z_c, z_c]
    rr = [r+d/2 for r, d in zip(r0, dr)]
    zr = [z+d/2 for z, d in zip(z0, dz)]
    rl = [r-d/2 for r, d in zip(r0, dr)]
    zl = [z-d/2 for z, d in zip(z0, dz)]

    zl[:5] = [-5.415, -3.6067, -1.7983, 1.8183, 3.6267]
    zr[:5] = [-3.6067, -1.7983, 1.8183, 3.6267, 5.435]

    dz = [a-b for a, b in zip(zr, zl)]

    for i in range(len(rl)):
        plt.plot([rl[i], rr[i], rr[i], rl[i], rl[i]],
                 [zl[i], zl[i], zr[i], zr[i], zl[i]], 'k', linewidth=lw, alpha=1)

    res = 300
    R = 16
    t = np.linspace(-np.pi/2, np.pi/2, res)
    plt.plot([0, 0], [-R, R], 'k', linewidth=lw, alpha=1)
    plt.plot(R * np.cos(t), R * np.sin(t), 'k', linewidth=lw, alpha=1)
    

def plot_solution(psi, elements, vertices):
    x = vertices[:, 0]
    y = vertices[:, 1]
    triangles = elements[:, 2:]
    jet = plt.cm.get_cmap("jet")
    p = plt.tricontour(x, y, triangles, psi, 100, alpha=.8)
    # cbar = plt.colorbar()
    # cbar.set_alpha(1)
    plt.axis('square')
    # plt.axis('equal')
    return p

def plot_mesh(elements, vertices, lw=.6):
    x = vertices[:, 0]
    y = vertices[:, 1]
    triangles = elements[:, 2:]
    plt.triplot(x, y, triangles, color='k', linewidth=lw)
    # plt.axis('equal')
    plt.axis('square')

