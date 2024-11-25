import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, optimize
from scipy.special import legendre
import argparse
from matplotlib.backends.backend_pdf import PdfPages

def lobatto_nodes(N):
	roots = legendre(N-1).deriv().roots
	x = np.concatenate(([-1], roots, [1]))
	return np.sort(x)

def chebyshev_nodes(N):
	return [-np.cos(np.pi*i/(N-1)) for i in range(N)]

def legendre_nodes(N):
	[x, _] = np.polynomial.legendre.leggauss(N)
	return np.sort(x)

def legendre_nodes_with_endpoints(N):
	[x, _] = np.polynomial.legendre.leggauss(N-2)
	x = np.concatenate(([-1], x, [1]))
	return np.sort(x)

def get_piecewise_linear_bound(xi, f, xx):
	return np.interp(xx, xi, f)

def optimize_bbox_onebasis_upper(up, xb, nsamp=1000, tol=1e-6):
	xx = np.linspace(-1, 1, nsamp)
	upx = up(xx)

	z0 = np.ones_like(xb)*np.max(upx)

	# Find discrete bounding box (on nsamp points) minimizing L2 norm
	def obj(z):
		pz = 2 # Use L2 norm
		cost = np.mean(np.maximum(0, con(z))**pz)**(1.0/pz)
		return cost

	def con(z):
		f = get_piecewise_linear_bound(xb, z, xx)
		return f - upx

	cons = []
	cons.append({'type': 'ineq', 'fun': con})

	result = optimize.minimize(obj, z0, method='SLSQP', constraints=cons, tol=1e-15)
	z = result.x

	# Take discrete bounding box and offset it by -min(f(x) - u(x)) to ensure continuous bounds preservation
	def obj2(x):
		f = get_piecewise_linear_bound(xb, z, x)
		return f - up(x)

	def con1(x):
		return x + 1

	def con2(x):
		return 1 - x

	cons = []
	cons.append({'type': 'ineq', 'fun': con1})
	cons.append({'type': 'ineq', 'fun': con2})

	newxx = np.linspace(-1, 1, int(1e6))
	x0 = newxx[np.argmin(obj2(newxx))]
	for i in range(5):
		off = optimize.minimize(obj2, x0, method='SLSQP', constraints=cons, tol=1e-12)
		if obj2(x0) < obj2(off.x):
			z -= obj2(x0)
			break
		else:
			z -= obj2(off.x)
			x0 = off.x

	off2 = optimize.minimize(obj2, off.x, method='SLSQP', constraints=cons, tol=1e-12)
	if obj2(off2.x) < -tol:
		print('Failed to find continuously bounds preserving offset.', obj2(off2.x), off.x, off2.x)

	return [z + tol, result.fun]

def optimize_and_write(xs, xb, sname, bname, nsamp=1000, plot=False):
	N = len(xs)
	M = len(xb)
	p = N-1

	# Loop over basis functions
	ups = []
	for i in range(N):
	    # Set solution to nodal interpolating basis function
	    u = np.zeros(N)
	    u[i] = 1.0
	    ups.append(np.poly1d(np.polyfit(xs, u, p)))


	xx = np.linspace(-1, 1, nsamp)
	upx = np.zeros((N, nsamp))
	for i in range(N):
		upx[i,:] = ups[i](xx)


	blow = np.zeros((N,M))
	bhigh = np.zeros((N,M))

	pdf_pages = PdfPages(f"bnddata_spts_{sname}_{N}_bpts_{bname}_{M}.pdf")

	funtotal = 0
	for i in range(N):
		[z, fun] = optimize_bbox_onebasis_upper(ups[i], xb, nsamp=nsamp)
		funtotal += fun
		bhigh[i,:] = z

		[z, fun] = optimize_bbox_onebasis_upper(-ups[i], xb, nsamp=nsamp)
		funtotal += fun
		blow[i,:] = -z

		plt.figure()
		plt.plot(xx, upx[i,:], 'k-')
		plt.plot(xb, bhigh[i,:], 'r.')
		plt.plot(xx, get_piecewise_linear_bound(xb, bhigh[i,:], xx), 'r-')
		plt.plot(xb, blow[i,:], 'b.')
		plt.plot(xx, get_piecewise_linear_bound(xb, blow[i,:], xx), 'b-')
		pdf_pages.savefig()
		plt.close()
	pdf_pages.close()


	filename = f"bnddata_spts_{sname}_{N}_bpts_{bname}_{M}.txt"
	np.savetxt(filename, [N], fmt="%d", newline="\n")
	with open(filename, "a") as f:
		np.savetxt(f, xs, fmt="%.15f", newline="\n")
		np.savetxt(f, [M], fmt="%d", newline="\n")
		np.savetxt(f, xb, fmt="%.15f", newline="\n")
		for i in range(N):
			np.savetxt(f, blow[i,:], fmt="%.15f", newline="\n")
			np.savetxt(f, bhigh[i,:], fmt="%.15f", newline="\n")

def main():
	# Initialize the parser
	parser = argparse.ArgumentParser(description="A script that processes some arguments")

	# Add arguments
	parser.add_argument('--N', type=int, help='Number of rows (N)', required=True)
	parser.add_argument('--M', type=int, help='Number of columns (M)', required=True)

	args = parser.parse_args()

	N = args.N
	M = args.M
	nsamp = 1000


	xs = lobatto_nodes(N)
	xb = lobatto_nodes(M)
	optimize_and_write(xs, xb, 'lobatto', 'lobatto', nsamp)
	xb = chebyshev_nodes(M)
	optimize_and_write(xs, xb, 'lobatto', 'chebyshev', nsamp)
	xb = legendre_nodes_with_endpoints(M)
	optimize_and_write(xs, xb, 'lobatto', 'legendre', nsamp)


	xs = legendre_nodes(N)
	xb = lobatto_nodes(M)
	optimize_and_write(xs, xb, 'legendre', 'lobatto', nsamp)
	xb = chebyshev_nodes(M)
	optimize_and_write(xs, xb, 'legendre', 'chebyshev', nsamp)
	xb = legendre_nodes_with_endpoints(M)
	optimize_and_write(xs, xb, 'legendre', 'legendre', nsamp)

# Entry point of the script
if __name__ == "__main__":
    main()

