import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, optimize
import argparse


def getLobattoPoints(i):
	if i == 0:
		return [0.0]
	elif i == 1:
		return [-1.0,1.0]
	elif i == 2:
		return [-1.0,0.0,1.0]
	elif i == 3:
		return [-1.0,-0.4472135954999579,0.4472135954999579,1.0]
	elif i == 4:
		return [-1.0,-0.6546536707079772,0.0,0.6546536707079772,1.0]
	elif i == 5:
		return [-1.0,-0.7650553239294647,-0.2852315164806451,0.2852315164806451,0.7650553239294647,1.0]
	elif i == 6:
		return [-1.0,-0.830223896278567,-0.46884879347071423,0.0,0.46884879347071423,0.830223896278567,1.0]
	elif i == 7:
		return [-1.0,-0.8717401485096066,-0.5917001814331423,-0.20929921790247888,0.20929921790247888,0.5917001814331423,0.8717401485096066,1.0]
	elif i == 8:
		return [-1.0,-0.8997579954114602,-0.6771862795107377,-0.36311746382617816,0.0,0.36311746382617816,0.6771862795107377,0.8997579954114602,1.0]
	elif i == 9:
		return [-1.0,-0.9195339081664589,-0.738773865105505,-0.4779249498104445,-0.16527895766638703,0.16527895766638703,0.4779249498104445,0.738773865105505,0.9195339081664589,1.0]
	elif i == 10:
		return [-1.0,-0.9340014304080592,-0.7844834736631444,-0.565235326996205,-0.2957581355869394,0.0,0.2957581355869394,0.565235326996205,0.7844834736631444,0.9340014304080592,1.0]
	elif i == 11:
		return [-1.0,-0.9448992722228822,-0.8192793216440066,-0.6328761530318607,-0.3995309409653489,-0.13655293285492756,0.13655293285492756,0.3995309409653489,0.6328761530318607,0.8192793216440066,0.9448992722228822,1.0]
	elif i == 12:
		return [-1.0,-0.9533098466421639,-0.8463475646518723,-0.6861884690817575,-0.4829098210913362,-0.24928693010623998,0.0,0.24928693010623998,0.4829098210913362,0.6861884690817575,0.8463475646518723,0.9533098466421639,1.0]
	elif i == 13:
		return [-1.0,-0.9599350452672609,-0.8678010538303472,-0.7288685990913262,-0.5506394029286471,-0.34272401334271285,-0.11633186888370387,0.11633186888370387,0.34272401334271285,0.5506394029286471,0.7288685990913262,0.8678010538303472,0.9599350452672609,1.0]
	elif i == 14:
		return [-1.0,-0.9652459265038386,-0.8850820442229763,-0.7635196899518152,-0.6062532054698457,-0.4206380547136725,-0.21535395536379423,0.0,0.21535395536379423,0.4206380547136725,0.6062532054698457,0.7635196899518152,0.8850820442229763,0.9652459265038386,1.0]

def ChebyshevPoints(n):
	return [-np.cos(np.pi*i/(n-1)) for i in range(n)]

def quad_area(x1,y1,x2,y2,x3,y3,x4,y4):
	return np.abs((x1*y2 - y1*x2) + (x2*y3 - y2*x3) + (x3*y4 - y3*x4) + (x4*y1 - y4*x1))

def vars_from_z(z):
	xi = z[:M-2]
	bhigh = z[M-2:M-2 + N*M]
	blow = z[M-2 + N*M:]

	bhigh = np.reshape(bhigh, (N, M))
	blow = np.reshape(blow, (N, M))

	return [xi, bhigh, blow]

def xb_from_xi(xi):
	return np.array([-1] + list(xi) + [1])

def get_piecewise_linear_bound(xi, f, xx):
	if len(xi) != len(f):
		xib = xb_from_xi(xi)
		return np.interp(xx, xib, f)
	else:
		return np.interp(xx, xi, f)

def expand_z(z, z0len):
	if z0len == 1:
		return z
	elif z0len % 2 == 0:
		return np.array(list(z) + list(-z[::-1]))
	else:
		return np.array(list(z) + [0] + list(-z[::-1]))


def optimize_bbox_all(ups, xb_initial, nsamp=1000, tol=1e-12, fixpoints=False):
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


	result = optimize.minimize(obj, z0, constraints=cons)
	z = expand_z(result.x, z0len)

	return [z, result.fun]

def optimize_bbox_onebasis_upper(up, xb, nsamp=1000, tol=1e-12):
	xx = np.linspace(-1, 1, nsamp)
	upx = up(xx)

	z0 = np.ones(M)*np.max(upx)

	def obj(z):
		pz = 2
		cost = np.mean(np.maximum(0, con(z))**pz)**(1.0/pz)
		return cost

	def con(z):
		f = get_piecewise_linear_bound(xb, z, xx)
		return f - upx

	cons = []
	cons.append({'type': 'ineq', 'fun': con})

	result = optimize.minimize(obj, z0, method='SLSQP', constraints=cons)

	return [result.x, result.fun]


# Initialize the parser
parser = argparse.ArgumentParser(description="A script that processes some arguments")

# Add arguments
parser.add_argument('--N', type=int, help='Number of rows (N)', required=True)
parser.add_argument('--M', type=int, help='Number of columns (M)', required=True)

args = parser.parse_args()

N = args.N
M = args.M


# N = 3
# M = 4

xs = getLobattoPoints(N-1)
# xb = ChebyshevPoints(M)
# print(xb)

p = N-1

# Loop over basis functions
ups = []
for i in range(N):
    # Set solution to nodal interpolating basis function
    u = np.zeros(N)
    u[i] = 1.0
    ups.append(np.poly1d(np.polyfit(xs, u, p)))


nsamp = 1000
xx = np.linspace(-1, 1, nsamp)
upx = np.zeros((N, nsamp))
for i in range(N):
	upx[i,:] = ups[i](xx)



xb = ChebyshevPoints(M)
blow = np.zeros((N,M))
bhigh = np.zeros((N,M))

plt.figure()
funtotal = 0
for i in range(N):
	plt.subplot(1, N, i+1)
	plt.plot(xx, upx[i,:], 'k-')
	[z, fun] = optimize_bbox_onebasis_upper(ups[i], xb, nsamp=nsamp)
	funtotal += fun
	bhigh[i,:] = z

	[z, fun] = optimize_bbox_onebasis_upper(-ups[i], xb, nsamp=nsamp)
	funtotal += fun
	blow[i,:] = -z

	plt.plot(xb, bhigh[i,:], 'r.')
	plt.plot(xx, get_piecewise_linear_bound(xb, bhigh[i,:], xx), 'r-')
	plt.plot(xb, blow[i,:], 'b.')
	plt.plot(xx, get_piecewise_linear_bound(xb, blow[i,:], xx), 'b-')
print(funtotal)
print(xb)


filename = f"bnddata_{N}_{M}_chebyshev.txt"
np.savetxt(filename, [N], fmt="%d", newline="\n")
with open(filename, "a") as f:
    np.savetxt(f, xs, fmt="%f", newline="\n")
    np.savetxt(f, [M], fmt="%d", newline="\n")
    np.savetxt(f, xb, fmt="%f", newline="\n")
    for i in range(N):
        np.savetxt(f, blow[i,:], fmt="%f", newline="\n")
        np.savetxt(f, bhigh[i,:], fmt="%f", newline="\n")


funtotal = None
for it in range(1):
	[xi, fun] = optimize_bbox_all(ups, xb, nsamp=nsamp, fixpoints=False)
	if funtotal is None:
		funtotal = fun
	xb = xb_from_xi(xi)
	print(it, fun/funtotal)
blow = np.zeros((N,M))
bhigh = np.zeros((N,M))

funtotal = 0
plt.figure()
for i in range(N):
	plt.subplot(1, N, i+1)
	plt.plot(xx, upx[i,:], 'k-')
	[z, fun] = optimize_bbox_onebasis_upper(ups[i], xb, nsamp=nsamp)
	funtotal += fun
	bhigh[i,:] = z

	[z, fun] = optimize_bbox_onebasis_upper(-ups[i], xb, nsamp=nsamp)
	funtotal += fun
	blow[i,:] = -z
	plt.plot(xb, bhigh[i,:], 'r.')
	plt.plot(xx, get_piecewise_linear_bound(xb, bhigh[i,:], xx), 'r-')
	plt.plot(xb, blow[i,:], 'b.')
	plt.plot(xx, get_piecewise_linear_bound(xb, blow[i,:], xx), 'b-')
# print(funtotal)
# print(xb)

filename = f"bnddata_{N}_{M}_opt.txt"
np.savetxt(filename, [N], fmt="%d", newline="\n")
with open(filename, "a") as f:
    np.savetxt(f, xs, fmt="%f", newline="\n")
    np.savetxt(f, [M], fmt="%d", newline="\n")
    np.savetxt(f, xb, fmt="%f", newline="\n")
    for i in range(N):
		# blow[i,1:-1] -= 1e-3
		# bhigh[i,1:-1] += 1e-3
        np.savetxt(f, blow[i,:], fmt="%f", newline="\n")
        np.savetxt(f, bhigh[i,:], fmt="%f", newline="\n")

# plt.show()

# # Loop over basis functions
# for i in range(N):
#     plt.figure(i)
#     # Set solution to nodal interpolating basis function
#     u = np.zeros(N)
#     u[i] = 1.0
#     up = np.poly1d(np.polyfit(xs, u, p))

#     plt.plot(xvis, up(xvis), 'k-')

#     # Loop over intervals and store bounding coefficient values (from both sides)
#     # for this basis function
#     bvals = np.zeros((M-1, 4))
#     for j in range(M-1):
#         xl = xb[j]
#         xr = xb[j+1]
#         bvals[j,:] = optimize_bbox(up, xl, xr) # a,b,c,d coeffs

#     # Make C0 continuous and store to global array
#     b_low[i, 0] = bvals[0, 3]
#     b_high[i, 0] = bvals[0, 1]
#     for j in range(1, M-1):
#         b_low[i, j]  = min(bvals[j-1, 2] + bvals[j-1, 3], bvals[j, 3])
#         b_high[i, j] = max(bvals[j-1, 0] + bvals[j-1, 1], bvals[j, 1])
#     b_low[i, -1] = bvals[-1, 2] + bvals[-1, 3]
#     b_high[i, -1] = bvals[-1, 0] + bvals[-1, 1]


#     plt.plot(xb, b_low[i,:], 'b--')
#     plt.plot(xb, b_high[i,:], 'r--')

# plt.show()

# # Print out data
# print('Nodal interpolating points:')
# print(N)
# print(', '.join(map(str, xs)))
# print()

# print('Bounding interval points:')
# print(M)
# print(', '.join(map(str, xb)))
# print()

# for i in range(N):
#     print(f'Lower bounding point values for basis function {i}:')
#     print(', '.join(map(str, b_low[i,:])))
#     print(f'Upper bounding point values for basis function {i}:')
#     print(', '.join(map(str, b_high[i,:])))
#     print()

# # plt.show()
# filename = f"bnddata_{N}_{M}.txt"
# np.savetxt(filename, [N], fmt="%d", newline="\n")
# with open(filename, "a") as f:
#     np.savetxt(f, xs, fmt="%f", newline="\n")
#     np.savetxt(f, [M], fmt="%d", newline="\n")
#     np.savetxt(f, xb, fmt="%f", newline="\n")
#     for i in range(N):
#         np.savetxt(f, b_low[i,:], fmt="%f", newline="\n")
#         np.savetxt(f, b_high[i,:], fmt="%f", newline="\n")


# # Entry point of the script
# if __name__ == "__main__":
#     main()