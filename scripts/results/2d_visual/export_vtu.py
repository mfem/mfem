import numpy as np
import matplotlib.pyplot as plt
import meshio
import copy

use_ho_quads = False # Use high-order (P1) VTU quads (to prevent triangular subdivison)
zscale = 0.4 # Scale z-coordinates (for easier visualization)
ref_levels = 1 # Shows up to this refinement level, 1 for base-level, 2 for base + next refinement level, etc.
M = 3
boundtxtpath = f'2DcustomboundinfoM{M}.txt'
soltxtpath = '2Dsolutioninfo_coarse.txt'
minvtupath = f'bounds_min_M{M}_r{ref_levels}{"_LO" if not use_ho_quads else ""}.vtu'
maxvtupath = f'bounds_max_M{M}_r{ref_levels}{"_LO" if not use_ho_quads else ""}.vtu'
solvtupath = 'sol.vtu'

def filter(data):
	level = data[:,5]
	keep_levels = np.zeros_like(level, dtype=bool)

	for i in range(ref_levels):
		if i == ref_levels - 1:
			keep_levels = np.logical_or(keep_levels, np.abs(level) == i+1)
		else:
			keep_levels = np.logical_or(keep_levels, level == i+1)

	data = data[keep_levels]
	return data

def expand_quad(pts):
	'''
	Quad:            	   Quad8: 
	3-----------2          3-----6-----2
	|           |          |           |
	|           |          |           |
	|           |          7           5
	|           |          |           |
	|           |          |           |
	0-----------1          0-----4-----1
	'''
	npts = len(pts)//4
	newpts = []
	for i in range(npts):
		x0 = pts[4*i, :]
		x1 = pts[4*i+1, :]
		x2 = pts[4*i+3, :]
		x3 = pts[4*i+2, :]
		newpts += [x0, x1, x2, x3, 0.5*(x0 + x1), 0.5*(x1 + x2), 0.5*(x2 + x3), 0.5*(x3 + x0)]

	return np.array(newpts)


bounddata = np.genfromtxt(boundtxtpath)
bounddata = filter(bounddata)

points_min = bounddata[:,[1,2,3]]
points_max = bounddata[:,[1,2,4]]
points_min[:,2] *= zscale
points_max[:,2] *= zscale

if use_ho_quads:
	points_max = expand_quad(points_max)
	points_min = expand_quad(points_min)
	npts = len(points_min)//8
	cellidxs = [[8*i, 8*i+1, 8*i+2, 8*i+3, 8*i+4, 8*i+5, 8*i+6, 8*i+7] for i in range(npts)]
	cells = {'quad8' : np.array(cellidxs)}
else:
	npts = len(points_min)//4
	cellidxs = [[4*i, 4*i+1, 4*i+3, 4*i+2] for i in range(npts)]
	cells = {'quad' : np.array(cellidxs)}


meshio.write_points_cells(minvtupath, points_min, cells)
meshio.write_points_cells(maxvtupath, points_max, cells)


soldata = np.genfromtxt(soltxtpath)
N = int(np.sqrt(soldata.shape[0]))
solpoints = soldata[:,[2,3,4]]
solpoints[:,2] *= zscale

cellidxs = []
for j in range(N-1):
	for i in range(N-1):
		cellidxs.append([(i) + (j)*N, (i+1) + (j)*N, (i+1) + (j+1)*N, (i) + (j+1)*N])

cells = {'quad' : np.array(cellidxs)}
meshio.write_points_cells(solvtupath, solpoints, cells, point_data={'u': solpoints[:,-1]})

