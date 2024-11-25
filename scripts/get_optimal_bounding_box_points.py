import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import argparse
from get_optimal_bounding_box_for_basis import * 


def xb_from_xi(xi):
	return np.array([-1] + list(xi) + [1])

def expand_z(z, z0len):
	if z0len == 1:
		return z
	elif z0len % 2 == 0:
		return np.array(list(z) + list(-z[::-1]))
	else:
		return np.array(list(z) + [0] + list(-z[::-1]))


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
	z0 = xb_initial[1:-1]
	z0len = len(z0)
	if z0len == 1:
		pass
	else:
		z0 = z0[:z0len//2]

	def obj(z):
		xi = expand_z(z, z0len)
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
	cons = []
	for i in range(M-2):
		def con(z, i=i):
			xi = expand_z(z, z0len)
			if i == 0:
				val = xi[i] + 1
			elif i == M-3:
				val = 1 - xi[i]
			else:
				val = xi[i] - xi[i-1]

			return val
		cons.append({'type': 'ineq', 'fun': con})

	result = optimize.minimize(obj, z0, constraints=cons, tol=1e-8)
	z = expand_z(result.x, z0len)

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

	xs = legendre_nodes(N)
	xb = legendre_nodes_with_endpoints(M)
	[xi, fun] = optimize_bbox_all(xs, xb, nsamp=nsamp)
	xb = xb_from_xi(xi)
	optimize_and_write(xs, xb, 'legendre', 'opt', nsamp)

# Entry point of the script
if __name__ == "__main__":
    main()
