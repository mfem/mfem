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

# Interpolation/interval points are defined on x \in [-1, 1]
# Nodal interpolation points (e.g., P5 Legendre points)
# xs = np.array([-0.93246951, -0.66120939, -0.23861919, 0.23861919, 0.66120939, 0.93246951])
# Interval boundary points (arbitrary)
# xb = [-1.0, -0.85, -0.5, -0.2, 0.1, 0.4, 0.7, 1.0]


def quad_area(x1,y1,x2,y2,x3,y3,x4,y4):
	return np.abs((x1*y2 - y1*x2) + (x2*y3 - y2*x3) + (x3*y4 - y3*x4) + (x4*y1 - y4*x1))


def optimize_bbox(up, xl, xr, nsamp=100, tol=1e-12):
	assert xr > xl
	# First transform to local variable xi \in [0,1]
	upx = lambda x : up((xr - xl)*x + xl)

	# Start first with optimizing on many discrete points
	nsamp = 100
	xs = np.linspace(0, 1, nsamp)
	dx = 1.0/nsamp
	us = upx(xs)

	'''
	Optimization problem (discrete convex hull):
	Find two lines:
		f = a*xi+b
		g = c*xi+d,
	minimizing the functional (bounding box area):
		quad_area(0, d, 1, c+d, 1, a+b, 0, b)
	such that:
		us - f < 0 \forall xi
		us - g > 0 \forall xi

	'''
	# Initial guess: take LSQ gradient and offset it
	m, b, _, _, _ = stats.linregress(xs, us)
	a = c = m
	b = np.max(us - m*xs)
	d = np.min(us - m*xs)
	f = a*xs + b
	g = c*xs + d

	z0 = np.array([a,b,c,d])
	def obj(z):
		a, b, c, d = z
		return quad_area(0, d, 1, c+d, 1, a+b, 0, b)

	def constraint1(z):
		a, b, c, d = z
		# us - f < 0 -> f - us > 0
		return (a*xs + b) - us

	def constraint2(z):
		a, b, c, d = z
		# us - g > 0
		return us - (c*xs + d)

	con1 = {'type': 'ineq', 'fun': constraint1}
	con2 = {'type': 'ineq', 'fun': constraint2}

	result = optimize.minimize(obj, z0, constraints=[con1, con2])
	a,b,c,d = result.x

	# Now take discrete convex hull and offset it to bound continuous polynomial.
	# This assumes that the discrete optimization was performed with sufficient resolution
	# such that the convex hull shape (i.e., bounding line slope) would not change much between
	# discrete and continuous optimization. We find continuous minima of (f - u) and (u - g) and
	# offset the intercepts (b and d) to account for negative values.

	obj_high = lambda x: (a*x + b) - upx(x)
	# Search one discrete sampling interval around discrete minimum
	x0 = xs[np.argmin(obj_high(xs))]
	bounds = [max(-tol, x0 - dx), min(x0 + dx, 1+tol)]
	result = optimize.minimize_scalar(obj_high, method='bounded', bounds=bounds)
	# Necessary around 0/1 since bounds appear to be non-inclusive
	offset = min(0, min(obj_high(result.x), obj_high(x0)))
	b = b - offset + tol

	obj_low = lambda x: upx(x) - (c*x + d)
	x0 = xs[np.argmin(obj_low(xs))]
	# Search one discrete sampling interval around discrete minimum
	bounds = [max(-tol, x0 - dx), min(x0 + dx, 1+tol)]
	result = optimize.minimize_scalar(obj_low, method='bounded', bounds=bounds)
	# Necessary around 0/1 since bounds appear to be non-inclusive
	offset = min(0, min(obj_low(result.x), obj_low(x0)))
	d = d + offset - tol

	return [a,b,c,d]

def main():
    # Initialize the parser
    parser = argparse.ArgumentParser(description="A script that processes some arguments")

    # Add arguments
    parser.add_argument('--N', type=int, help='Number of rows (N)', required=True)
    parser.add_argument('--M', type=int, help='Number of columns (M)', required=True)


    args = parser.parse_args()

    N = args.N
    M = args.M

    xs = getLobattoPoints(N-1)
    xb = ChebyshevPoints(M)

    p = N-1
    xvis = np.linspace(-1, 1, 100)

    # Store bounding points
    b_low = np.zeros((N, M))
    b_high = np.zeros((N, M))

    # Loop over basis functions
    for i in range(N):
        plt.figure(i)
        # Set solution to nodal interpolating basis function
        u = np.zeros(N)
        u[i] = 1.0
        up = np.poly1d(np.polyfit(xs, u, p))

        plt.plot(xvis, up(xvis), 'k-')

        # Loop over intervals and store bounding coefficient values (from both sides)
        # for this basis function
        bvals = np.zeros((M-1, 4))
        for j in range(M-1):
            xl = xb[j]
            xr = xb[j+1]
            bvals[j,:] = optimize_bbox(up, xl, xr) # a,b,c,d coeffs

        # Make C0 continuous and store to global array
        b_low[i, 0] = bvals[0, 3]
        b_high[i, 0] = bvals[0, 1]
        for j in range(1, M-1):
            b_low[i, j]  = min(bvals[j-1, 2] + bvals[j-1, 3], bvals[j, 3])
            b_high[i, j] = max(bvals[j-1, 0] + bvals[j-1, 1], bvals[j, 1])
        b_low[i, -1] = bvals[-1, 2] + bvals[-1, 3]
        b_high[i, -1] = bvals[-1, 0] + bvals[-1, 1]


        plt.plot(xb, b_low[i,:], 'b--')
        plt.plot(xb, b_high[i,:], 'r--')

    plt.show()

    # Print out data
    print('Nodal interpolating points:')
    print(N)
    print(', '.join(map(str, xs)))
    print()

    print('Bounding interval points:')
    print(M)
    print(', '.join(map(str, xb)))
    print()

    for i in range(N):
        print(f'Lower bounding point values for basis function {i}:')
        print(', '.join(map(str, b_low[i,:])))
        print(f'Upper bounding point values for basis function {i}:')
        print(', '.join(map(str, b_high[i,:])))
        print()

    # plt.show()
    filename = f"bnddata_{N}_{M}.txt"
    np.savetxt(filename, [N], fmt="%d", newline="\n")
    with open(filename, "a") as f:
        np.savetxt(f, xs, fmt="%f", newline="\n")
        np.savetxt(f, [M], fmt="%d", newline="\n")
        np.savetxt(f, xb, fmt="%f", newline="\n")
        for i in range(N):
            np.savetxt(f, b_low[i,:], fmt="%f", newline="\n")
            np.savetxt(f, b_high[i,:], fmt="%f", newline="\n")


# Entry point of the script
if __name__ == "__main__":
    main()

