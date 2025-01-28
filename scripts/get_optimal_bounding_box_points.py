import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import argparse
from get_optimal_bounding_box_for_basis import *


def xb_from_xi(xi):
	return np.array([-1] + list(xi) + [1])

def expand_z(z):
	zz = np.zeros_like(z)
	for i in range(len(z)):
		if i == 0:
			zz[i] = -1 + np.exp(z[0])
		else:
			zz[i] = zz[i-1] + np.exp(z[i])
	if (M-2) == 1:
		return zz
	elif (M-2) % 2 == 0:
		return np.array(list(zz) + list(-zz[::-1]))
	else:
		return np.array(list(zz) + [0] + list(-zz[::-1]))


def optimize_bbox_all(xs, xb_initial, nsamp=1000, tol=1e-6):
	global N, M, p

	# Generate basis polynomials
	ups = []
	for i in range(N):
	    # Set solution to nodal interpolating basis function
	    u = np.zeros(N)
	    u[i] = 1.0
	    ups.append(np.poly1d(np.polyfit(xs, u, p)))

	# Make sure control nodes are symmetric in [-1, 1]
	z0 = np.log(xb_initial[1:] - xb_initial[:-1])[:-1]
	if M == 1:
		pass
	else:
		z0 = z0[:(M-2)//2]


	def obj(z):
		xi = expand_z(z)
		xib = xb_from_xi(xi)

		totalfun = 0
		for i in range(N):
			# Optimize upper bound
			[_, fun] = optimize_bbox_onebasis_upper(ups[i], xib, nsamp=nsamp)
			totalfun += fun

			# Optimize lower bound
			[_, fun] = optimize_bbox_onebasis_upper(-ups[i], xib, nsamp=nsamp)
			totalfun += fun

		return totalfun

	# Constrain so nodal points are increasing
	def con(z):
		return 1 - np.max(expand_z(z))
	cons = {'type': 'ineq', 'fun': con}

	result = optimize.minimize(obj, z0, method='SLSQP', constraints=cons, tol=1e-15)
	z = expand_z(result.x)

	return [z, result.fun]

def main():
	global N, M, p
	# Initialize the parser
	parser = argparse.ArgumentParser(description="A script that processes some arguments")

	# Add arguments
	parser.add_argument('--N', type=int, help='Number of rows (N)', required=True)
	parser.add_argument('--M', type=int, help='Number of columns (M)', required=True)

	args = parser.parse_args()
	N = args.N
	M = args.M

	nsamp = 1000
	p = N-1

	xs = lobatto_nodes(N)
	xb = legendre_nodes_with_endpoints(M)
	[xi, fun] = optimize_bbox_all(xs, xb, nsamp=nsamp)
	xb = xb_from_xi(xi)
	optimize_and_write(xs, xb, 'lobatto', 'opt', nsamp)

	# xs = legendre_nodes(N)
	# xb = legendre_nodes_with_endpoints(M)
	# [xi, fun] = optimize_bbox_all(xs, xb, nsamp=nsamp)
	# xb = xb_from_xi(xi)
	# optimize_and_write(xs, xb, 'legendre', 'opt', nsamp)

# Entry point of the script
if __name__ == "__main__":
    main()
