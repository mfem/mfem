import numpy as np
import csv 
import matplotlib.pyplot as plt

def read(file):
	with open(file) as f:
		N = int(next(f))
		xs = np.array([float(next(f)) for _ in range(N)])
		M = int(next(f))
		xb = np.array([float(next(f)) for _ in range(M)])

		bhigh = np.zeros((N,M))
		blow = np.zeros((N,M))
		for i in range(N):
			blow[i,:] = np.array([float(next(f)) for _ in range(M)])
			bhigh[i,:] = np.array([float(next(f)) for _ in range(M)])

	return [xs, xb, blow, bhigh]

def get_piecewise_linear_bound(xi, f, xx):
	return np.interp(xx, xi, f)

def get_errors(xs, xb, blow, bhigh, nsamp=10000):
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
	l1err = 0
	l2err = 0
	linferr = 0

	for i in range(N):
		def con1(z):
			f = get_piecewise_linear_bound(xb, z, xx)
			return f - ups[i](xx)
		def con2(z):
			f = get_piecewise_linear_bound(xb, z, xx)
			return f - -ups[i](xx)
		l1err += np.mean(np.abs(con1(bhigh[i,:])) + np.abs(con2(-blow[i,:])))
		l2err += np.sqrt(np.mean((con1(bhigh[i,:]))**2) + np.mean((con2(-blow[i,:]))**2))
		linferr = max(linferr, np.max(con1(bhigh[i,:])))
		linferr = max(linferr, np.max(con2(-blow[i,:])))

	return [l1err, l2err, linferr]

Ns = range(3, 8)
Ms = range(4, 20)

l1errs = np.zeros((len(Ms)))
l2errs = np.zeros((len(Ms)))
linferrs = np.zeros((len(Ms)))

for i, N in enumerate(Ns):
	plt.subplot(2,3,i+1)

	l1errs = l2errs = linferrs = np.zeros((len(Ms)))
	for j, M in enumerate(Ms):
		try:
			[xs, xb, blow, bhigh] = read(f'bnddata_spts_lobatto_{N}_bpts_legendre_{M}.txt')
			[l1err, l2err, linferr] = get_errors(xs, xb, blow, bhigh)
			l1errs[j] = l1err
			l2errs[j] = l2err
			linferrs[j] = linferr
		except:
			pass
	plt.semilogy(Ms, l2errs, 'ro-')

	l1errs = l2errs = linferrs = np.zeros((len(Ms)))
	for j, M in enumerate(Ms):
		try:
			[xs, xb, blow, bhigh] = read(f'bnddata_spts_lobatto_{N}_bpts_lobatto_{M}.txt')
			[l1err, l2err, linferr] = get_errors(xs, xb, blow, bhigh)
			l1errs[j] = l1err
			l2errs[j] = l2err
			linferrs[j] = linferr
		except:
			pass
	plt.semilogy(Ms, l1errs, 'go-')

	l1errs = l2errs = linferrs = np.zeros((len(Ms)))
	for j, M in enumerate(Ms):
		try:
			[xs, xb, blow, bhigh] = read(f'bnddata_spts_lobatto_{N}_bpts_chebyshev_{M}.txt')
			[l1err, l2err, linferr] = get_errors(xs, xb, blow, bhigh)
			l1errs[j] = l1err
			l2errs[j] = l2err
			linferrs[j] = linferr
		except:
			pass
	plt.semilogy(Ms, l2errs, 'bo-')

	l1errs = l2errs = linferrs = np.zeros((len(Ms)))
	for j, M in enumerate(Ms):
		try:
			[xs, xb, blow, bhigh] = read(f'bnddata_spts_lobatto_{N}_bpts_equispaced_{M}.txt')
			[l1err, l2err, linferr] = get_errors(xs, xb, blow, bhigh)
			l1errs[j] = l1err
			l2errs[j] = l2err
			linferrs[j] = linferr
		except:
			pass
	plt.semilogy(Ms, l2errs, 'co-')

	l1errs = l2errs = linferrs = np.zeros((len(Ms)))
	for j, M in enumerate(Ms):
		try:
			[xs, xb, blow, bhigh] = read(f'bnddata_spts_lobatto_{N}_bpts_opt_{M}.txt')
			[l1err, l2err, linferr] = get_errors(xs, xb, blow, bhigh)
			l1errs[j] = l1err
			l2errs[j] = l2err
			linferrs[j] = linferr
		except:
			pass
	plt.semilogy(Ms, l2errs, 'ko-')
	plt.xlabel('M')
	plt.ylabel('L2 err')
	plt.title(f'N = {N}')
	if i == 0:
		plt.legend(['GL + endpoints', 'Chebyshev', 'GLL', 'Equispaced', 'Opt'])

plt.show()
