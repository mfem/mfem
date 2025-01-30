import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import basinhopping
import argparse
from get_optimal_bounding_box_for_basis import *


def xb_from_xi(xi):
	return np.array([-1] + list(xi) + [1])

def expand_z(z):
	zz = np.zeros_like(z)[:-1]
	zfac = np.sum(np.exp(z))
	for i in range(len(z)-1):
		if i == 0:
			zz[i] = -1 + np.exp(z[0])/zfac
		else:
			zz[i] = zz[i-1] + np.exp(z[i])/zfac

	if (M-2) == 1:
		pass
	elif (M-2) % 2 == 0:
		zz = np.array(list(zz) + list(-zz[::-1]))
	else:
		zz = np.array(list(zz) + [0] + list(-zz[::-1]))

	return zz


def optimize_bbox_all(xs, xb_initial, nsamp=1000, tol=1e-6, return_initial_cost=False):
	global N, M, p

	# Generate basis polynomials
	ups = []
	for i in range(N):
	    # Set solution to nodal interpolating basis function
	    u = np.zeros(N)
	    u[i] = 1.0
	    ups.append(np.poly1d(np.polyfit(xs, u, p)))

	# Make sure control nodes are symmetric in [-1, 1]
	z0 = np.log(xb_initial[1:] - xb_initial[:-1])
	if M == 1:
		pass
	else:
		z0 = z0[:(M-2)//2]
	zn = np.log(-xb_initial[(M-2)//2])
	z0 = np.concatenate((z0, [zn]))

	def obj(z):
		xi = expand_z(z)
		xib = xb_from_xi(xi)

		totalfun = 0
		Neff = N//2 if N % 2 == 0 else N//2 + 1
		for i in range(Neff):
			# Optimize upper bound
			[_, fun] = optimize_bbox_onebasis_upper(ups[i], xib, nsamp=nsamp)
			if N % 2 == 1 and i == Neff:
				totalfun += fun
			else:
				totalfun += 2*fun

			# Optimize lower bound
			[_, fun] = optimize_bbox_onebasis_upper(-ups[i], xib, nsamp=nsamp)
			if N % 2 == 1 and i == Neff:
				totalfun += fun
			else:
				totalfun += 2*fun
		return totalfun
	if return_initial_cost:
		return [expand_z(z0), obj(z0)]

	minimizer_kwargs = {"method": "BFGS"}
	result = basinhopping(obj, z0, minimizer_kwargs=minimizer_kwargs, niter=40)
	z = expand_z(result.x)

	return [z, result.fun]

def plot_solution_and_bases(x_sol, l_sol, u_sol, gll_nodes):
    """
    x_sol, l_sol, u_sol: piecewise bounds from the Pyomo solution
    gll_nodes: the GLL nodes used to define Lagrange basis polynomials
    """
    # 1) Plot the piecewise-linear bounds
    #    (Just connect the breakpoints with lines)
    plt.plot(x_sol, l_sol, 'o--', color='black', label='Lower Bound')
    plt.plot(x_sol, u_sol, 'o--', color='red',   label='Upper Bound')

    # 2) Plot each Lagrange basis polynomial on a dense grid
    x_dense = np.linspace(-1, 1, 400)
    N = len(gll_nodes)
    for j in range(N):
        # Evaluate L_j at all points in x_dense
        L_vals = [lagrange_basis_polynomial(j, xd, gll_nodes) for xd in x_dense]
        plt.plot(x_dense, L_vals, label=f'Lagrange basis L_{j}')

    # 3) Make it look nice
    plt.title('Piecewise-Linear Bounds and Lagrange Basis Polynomials')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend(loc='best')
    plt.show()

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

	# Find best initial guess
	xbs = [legendre_nodes_with_endpoints(M),
			lobatto_nodes(M),
			chebyshev_nodes(M),
			np.linspace(-1, 1, M)]
	funs = np.zeros(len(xbs))
	for i in range(len(xbs)):
		[_, f] = optimize_bbox_all(xs, xbs[i], nsamp=nsamp, return_initial_cost=True)
		funs[i] = f
	xb = xbs[np.argmin(funs)]

	# Run optimizer
	[xi, fun] = optimize_bbox_all(xs, xb, nsamp=nsamp)
	xb = xb_from_xi(xi)
	optimize_and_write(xs, xb, 'lobatto', 'opt', nsamp)
	print(N,M,fun)

	# plot_solution_and_bases(x_sol, l_sol, u_sol, gll_nodes)

	# xs = legendre_nodes(N)
	# xb = legendre_nodes_with_endpoints(M)
	# [xi, fun] = optimize_bbox_all(xs, xb, nsamp=nsamp)
	# xb = xb_from_xi(xi)
	# optimize_and_write(xs, xb, 'legendre', 'opt', nsamp)

# Entry point of the script
if __name__ == "__main__":
    main()
