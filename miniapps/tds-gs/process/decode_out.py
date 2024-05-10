import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

gf = "final.gf"
mesh = "mesh_refine.mesh"

data = {}
with open(gf, "r") as fid:

    out = fid.readline()

    out = fid.readline()[:-1]
    data["FiniteElementCollection"] = out.split(": ")[-1]
    
    out = fid.readline()[:-1]
    data["VDim"] = eval(out.split(": ")[-1])

    out = fid.readline()[:-1]
    data["Ordering"] = eval(out.split(": ")[-1])

    out = fid.readline()

    psi = []
    for line in fid:
        psi.append(eval(line[:-1]))
    data["psi"] = np.array(psi)

with open(mesh, "r") as fid:

    for line in fid:
        if "dimension" in line:
            data["dimension"] = eval(fid.readline()[:-1])
        elif "elements" in line:
            num_elements = eval(fid.readline()[:-1])
            data["num_elements"] = num_elements
            elements = []
            for k in range(num_elements):
                out = fid.readline()[:-1]
                out = out.split(" ")
                for l in range(len(out)):
                    out[l] = eval(out[l])
                elements.append(out)
            data["elements"] = np.array(elements, dtype=int)
        elif "boundary" in line:
            num_boundary_elements = eval(fid.readline()[:-1])
            data["num_boundary_elements"] = num_elements
            boundary_elements = []
            for k in range(num_boundary_elements):
                out = fid.readline()[:-1]
                out = out.split(" ")
                for l in range(len(out)):
                    out[l] = eval(out[l])
                boundary_elements.append(out)
            data["boundary_elements"] = np.array(boundary_elements, dtype=int)
        elif "vertices" in line:
            num_vertices = eval(fid.readline()[:-1])
            data["num_vertices"] = num_vertices
            num_dim = eval(fid.readline()[:-1])
            vertices = []
            for k in range(num_vertices):
                out = fid.readline()[:-1]
                out = out.split(" ")
                for l in range(len(out)):
                    out[l] = eval(out[l])
                vertices.append(out)
            data["vertices"] = np.array(vertices)

# create an unstructured triangular grid instance
vert_x = data['vertices'][:, 0]
vert_y = data['vertices'][:, 1]
elements = data['elements'][:, 2:]
psi = data['psi']

triangulation = tri.Triangulation(vert_x, vert_y, elements)

# plot the mesh, slow...
# plt.figure()
# for element in elements:
#     x = [vert_x[element[i]] for i in range(len(element))]
#     y = [vert_y[element[i]] for i in range(len(element))]
#     plt.fill(x, y, edgecolor='black', fill=False)

# plot the contours
plt.figure()
plt.tricontour(triangulation, psi, levels=100)
plt.colorbar()
plt.show()
breakpoint()
