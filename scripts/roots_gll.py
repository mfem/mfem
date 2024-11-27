import numpy as np
from scipy.special import legendre
from sympy import symbols, lambdify, diff, solve
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import argparse
from scipy.optimize import root

#python3 roots_gll.py  --N 5

def ChebyshevPoints(n):
	return [-np.cos(np.pi*i/(n-1)) for i in range(n)]

def gll_points(N):
    # Compute the GLL points
    roots = legendre(N-1).deriv().roots
    gll_points = np.concatenate(([-1], roots, [1]))
    return gll_points

def lagrange_interpolants(x_nodes):
    # Construct the Lagrange interpolants using symbolic computation
    x = symbols('x')
    N = len(x_nodes)
    L = []
    for i in range(N):
        Li = 1
        for j in range(N):
            if i != j:
                Li *= (x - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        L.append(Li)
    return L, x

def second_derivative(L, x):
    # Compute the second derivative of the Lagrange interpolants
    d2L = [diff(Li, x, 2) for Li in L]
    return d2L

def first_derivative(L, x):
    # Compute the first derivative of the Lagrange interpolants
    dL = [diff(Li, x) for Li in L]
    return dL

# def find_zeros(d2L, x):
#     # Find the zeros of the second derivative
#     zeros = []
#     for d2Li in d2L:
#         roots = solve(d2Li, x)
#         zeros.append([root.evalf() for root in roots if root.is_real])
#     return zeros

def find_zeros(d2L, x, x_nodes):
    # Find the zeros of the second derivative using scipy.optimize.root
    zeros = []
    for d2Li in d2L:
        d2Li_func = lambdify(x, d2Li, 'numpy')
        initial_guesses = np.linspace(-1, 1, len(x_nodes) - 2)  # Initial guesses for the roots
        roots = []
        for guess in initial_guesses:
            result = root(d2Li_func, guess)
            if result.success:
                roots.append(result.x[0])
        zeros.append(remove_duplicates(roots))
    return zeros

def remove_duplicates(roots, threshold=1e-6):
    # Remove duplicate roots that are within a small threshold
    if not roots:
        return roots
    roots.sort()
    unique_roots = [roots[0]]
    for i in range(1, len(roots)):
        if abs(roots[i] - roots[i-1]) > threshold:
            unique_roots.append(roots[i])
    return unique_roots

def remove_duplicates2(roots, threshold=1e-6):
    roots = np.sort(roots)
    unique_roots = [roots[0]]
    for i in range(1, len(roots)):
        if abs(roots[i] - roots[i-1]) > threshold:
            unique_roots.append(roots[i])
    return unique_roots

def plot_functions(L, dL, d2L, x_nodes, x_symbolic, x_values, N, lower, upper, int_points, piecewise=False, suffix='', lower2=None, upper2=None, int_points2=None, int_points3=None):
    # Plot the Lagrange basis functions, their first derivatives, and their second derivatives
    pdf_pages = PdfPages('ProvableBounds'+suffix+'_'+str(N)+'_'+str(len(int_points[0,:]))+'.pdf')
    plt.figure(figsize=(18, 5))

    # Plot the Lagrange basis functions
    plt.subplot(1, 3, 1)
    for i, Li in enumerate(L):
        Li_func = lambdify(x_symbolic, Li, 'numpy')
        plt.plot(x_values, Li_func(x_values), label=f'L{i}')
    plt.title('Lagrange Basis Functions')
    plt.xlabel('x')
    plt.ylabel('L(x)')
    plt.legend()
    plt.plot(x_nodes, 0*x_nodes, 'ko-')
    plt.grid(True)

    # Plot the first derivatives of the Lagrange basis functions
    plt.subplot(1, 3, 2)
    for i, dLi in enumerate(dL):
        dLi_func = lambdify(x_symbolic, dLi, 'numpy')
        plt.plot(x_values, dLi_func(x_values), label=f"L'{i}")
    plt.title('First Derivatives of Lagrange Basis Functions')
    plt.plot(0,'k-')
    plt.xlabel('x')
    plt.ylabel("L'(x)")
    plt.legend()
    plt.plot(x_nodes, 0*x_nodes, 'ko-')
    plt.grid(True)

    # Plot the second derivatives of the Lagrange basis functions
    plt.subplot(1, 3, 3)
    for i, d2Li in enumerate(d2L):
        d2Li_func = lambdify(x_symbolic, d2Li, 'numpy')
        plt.plot(x_values, d2Li_func(x_values), label=f"L''{i}")
    plt.title('Second Derivatives of Lagrange Basis Functions')
    plt.xlabel('x')
    plt.ylabel("L''(x)")
    plt.legend()
    plt.plot(x_nodes, 0*x_nodes, 'ko-')
    plt.grid(True)

    plt.tight_layout()
    # plt.show()
    pdf_pages.savefig()
    plt.close()

    for i, (Li, dLi, d2Li) in enumerate(zip(L, dL, d2L)):
        plt.figure()
        # Plot the Lagrange basis function
        Li_func = lambdify(x_symbolic, Li, 'numpy')
        plt.plot(x_values, Li_func(x_values), label=f'L{i}')
        dLi_func = lambdify(x_symbolic, dLi, 'numpy')
        dLi_vals = dLi_func(x_values)
        dLi_vals = dLi_vals/np.max(np.abs(dLi_vals))
        plt.plot(x_values, dLi_vals, label=f"L'{i}")
        d2Li_func = lambdify(x_symbolic, d2Li, 'numpy')
        d2Li_vals = d2Li_func(x_values)
        d2Li_vals = d2Li_vals/np.max(np.abs(d2Li_vals))
        plt.plot(x_values, d2Li_vals, label=f"L''{i}")
        plt.xlabel('x')
        # plt.ylabel('L(x)')
        plt.plot(x_nodes, 0*x_nodes, 'ko-')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        pdf_pages.savefig()
        plt.close()
        # plt.show()

    # plot piecewise bases
    ms = 1.0
    if piecewise:
        for i, (Li, dLi, d2Li) in enumerate(zip(L, dL, d2L)):
            plt.figure()
            # Plot the Lagrange basis function
            Li_func = lambdify(x_symbolic, Li, 'numpy')
            plt.plot(x_values, Li_func(x_values), label=f'L{i}')
            plt.plot(int_points[i,:], lower[i, :], 'rx-', linewidth=0.5, markersize=ms)
            plt.plot(int_points[i,:], upper[i, :], 'bx-', linewidth=0.5, markersize=ms)
            if lower2 is not None:
                plt.plot(int_points2[i,:], lower2[i, :], 'ro--', linewidth=0.5, markersize=ms)
                plt.plot(int_points2[i,:], upper2[i, :], 'bo--', linewidth=0.5, markersize=ms)
            plt.xlabel('x')
            # plt.ylabel('L(x)')
            plt.plot(x_nodes, 0*x_nodes, 'ko-',markerfacecolor='none')
            plt.plot(int_points[i,:], 0*int_points[i,:], 'kx-')
            if int_points3 is not None:
                plt.plot(int_points3[i,:], 0*int_points3[i,:], 'ks-',markerfacecolor='none')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            pdf_pages.savefig()
            plt.close()

    pdf_pages.close()

def evaluate_at_point(L, dL, x_symbolic, a, i):
    # Evaluate the i-th Lagrange interpolant and its first derivative at x = a
    Li_func = lambdify(x_symbolic, L[i], 'numpy')
    dLi_func = lambdify(x_symbolic, dL[i], 'numpy')
    Li_value = Li_func(a)
    dLi_value = dLi_func(a)
    return Li_value, dLi_value

def find_pl_bounds(x_nodes, L, dL, x, bi, int_points):
    nr = len(x_nodes)
    mr = len(int_points)
    upper = np.zeros(mr)
    lower = np.zeros(mr)
    for i in range(mr):
        if i == 0:
            if bi == 0:
                upper[i] = 1
                lower[i] = 1
            else:
                upper[i] = 0
                lower[i] = 0
            continue
        if i == mr-1:
            if bi == nr-1:
                upper[i] = 1
                lower[i] = 1
            else:
                upper[i] = 0
                lower[i] = 0
            continue

        xv = int_points[i]
        xmv = 0.5*(xv+int_points[i-1])
        xpv = 0.5*(xv+int_points[i+1])

        bv, dv = evaluate_at_point(L, dL, x, xv, bi)
        bmv, dmv = evaluate_at_point(L, dL, x, xmv, bi)
        bpv, dpv = evaluate_at_point(L, dL, x, xpv, bi)

        dm = xv-xmv
        dp = xv-xpv

        lower[i] = min(bv, bmv + dm*dmv, bpv+dp*dpv) - 1e-5
        upper[i] = max(bv, bmv + dm*dmv, bpv+dp*dpv) + 1e-5

    return lower, upper

def find_intersection(x1,y1,m1,x2,y2,m2):

    # Check for parallel lines
    if m1 == m2:
        return None  # Parallel lines do not intersect

    # Solve for x
    x = ((m1 * x1 - m2 * x2) + (y2 - y1)) / (m1 - m2)

    # Solve for y using the equation of the first line
    y = m1 * (x - x1) + y1

    return (x, y)

# get bounds of a function assuming f" changes sign once in the interval.
# xpp gives location of f"=0
def get_fun_ub(x_nodes, Li, dLi, d2Li, x, xstart, xend, xpp):
    xints = np.zeros(5)
    xints[0] = xstart
    xints[4] = xend
    yuints = np.zeros(5)
    ylints = np.zeros(5)


    Li_func = lambdify(x, Li, 'numpy')
    dLi_func = lambdify(x, dLi, 'numpy')
    d2Li_func = lambdify(x, d2Li, 'numpy')
    yuints[0] = Li_func(xstart)
    yuints[4] = Li_func(xend)
    ylints[0] = Li_func(xstart)
    ylints[4] = Li_func(xend)
    ddf0 = d2Li_func(xstart)

    #find intersection of tangent line at xstart with tangent line at f''=0
    int1 = find_intersection(xstart, Li_func(xstart), dLi_func(xstart), xpp, Li_func(xpp), dLi_func(xpp))
    int2 = find_intersection(xend, Li_func(xend), dLi_func(xend), xpp, Li_func(xpp), dLi_func(xpp))


    if ddf0 > 0:
        xints[1] = int1[0]
        xints[2] = xpp
        xints[3] = int2[0]

        yuints[1] = Li_func(int1[0])
        yuints[2] = Li_func(xpp)
        yuints[3] = int2[1]

        ylints[1] = int1[1]
        ylints[2] = Li_func(xpp)
        ylints[3] = Li_func(int2[0])
    else:
        xints[1] = int1[0]
        xints[2] = xpp
        xints[3] = int2[0]

        yuints[1] = int1[1]
        yuints[2] = Li_func(xpp)
        yuints[3] = Li_func(int2[0])

        ylints[1] = Li_func(int1[0])
        ylints[2] = Li_func(xpp)
        ylints[3] = int2[1]

    return (xints, yuints, ylints)

def get_combined_pl_bounds(xint, yint, xnew, upper=True):
    ynew = np.zeros(len(xnew))
    ynew[0] = yint[0]
    ynew[-1] = yint[-1]
    n = len(xnew)
    jprev = 0
    xstart = xint[jprev]
    ystart = yint[jprev]

    for i in range(1,n):
        print("================ upper->",upper)
        xv = xnew[i]
        yv = np.interp(xv, xint, yint)
        # find the previous point in xint
        jnext = -1
        for j in range(len(xint)):
            if abs(xint[j]-xv) < 1e-10:
                jnext = j
                break

        print(i,xnew[i],jprev,jnext,'k10getbounds1')
        nintervals = jnext-jprev
        if (nintervals == 1):
            ynew[i] = yv
            jprev = jnext
        else:
            xmid = xint[jprev+1]
            ymid = yint[jprev+1]
            print(xv,xstart,xmid,'k10getbounds2')

            for j in range(1,nintervals-1):
                x2 = xint[jprev+j+2]
                y2 = yint[jprev+j+2]

                m1 = (ymid-ystart)/(xmid-xstart)
                m2 = (y2-ymid)/(x2-xmid)

                # extrapolate from x0,y0 and x1,y1 to x2
                y2option = ymid + m1*(x2-xmid)
                y2save = y2
                if upper:
                    y2 = max(y2, y2option)
                else:
                    y2 = min(y2, y2option)

                xmid = x2
                ymid = y2
                print(xv,xstart,xmid,ystart,y2,y2option,'k10getbounds3')
            xstart = xmid
            ystart = ymid
            jprev = jnext
            ynew[i] = ymid
            print(xv,y2,ymid,jprev,jnext,'k10getbounds4')
    return ynew


def main():
    # Number of GLL points
    parser = argparse.ArgumentParser(description="A script that processes some arguments")

    # Add arguments
    parser.add_argument('--N', type=int, help='Number of rows (N)', required=True)
    parser.add_argument('--M', type=int, help='Number of rows (N)', required=True)
    args = parser.parse_args()
    N = args.N
    M = args.M
    # print(N)

    # Step 1: Define the GLL points
    x_nodes = gll_points(N)
    # print(x_nodes)

    # Step 2: Construct the Lagrange interpolants
    L, x = lagrange_interpolants(x_nodes)

    # Step 3: Compute the first and second derivatives
    dL = first_derivative(L, x)
    d2L = second_derivative(L, x)

    # Step 4: Find the zeros of the second derivative
    # zeros = find_zeros(d2L, x)
    zeros = find_zeros(d2L, x, x_nodes)

    x_values = np.linspace(-1, 1, 100000)

    # Print the zeros
    print("Zeros of the second derivative of each Lagrange interpolant:")
    for i, z in enumerate(zeros):
        print(f"Lagrange interpolant {i}: {z}")

    # Step 5: Plot the functions
    x_values = np.linspace(-1, 1, 1000)

    nr = len(x_nodes)
    mr = nr-1
    # mr = 2*nr

    lower = np.zeros((nr,mr))
    upper = np.zeros((nr,mr))
    int_points = np.zeros((nr,mr))

    for i in range(nr):
        # print(i)
        int_points[i,:] = np.concatenate(([-1], zeros[i], [1]))
        # int_points[i,:] = ChebyshevPoints(2*nr)
        lower[i,:],upper[i,:] = find_pl_bounds(x_nodes, L, dL, x, i, int_points[i,:])

    plot_functions(L, dL, d2L, x_nodes, x, x_values, N,
                   lower, upper, int_points, False)

    # there are nr-3 intervals where f" changes sign.
    # we will use 5 points per interval. 2 end points, 1 point for f"=0,
    # and 1 point to the left or right based on where the tangents intersect.
    # 1 of the tangent intersection points will come from the other bound.
    # total points = 4*(nr-3) + 1 = 4*nr-11

    nintp = 4*nr-11
    nzeros = nr-3
    nintervals = nr-2

    xnew = np.zeros((nr, nintp))
    nupper = np.zeros((nr, nintp))
    nlower = np.zeros((nr, nintp))
    # compute M_{min}
    xnew[:,:] = -100
    for i in range(nr):
        xnew[i,0] = -1.0
        xnew[i,nintp-1] = 1.0

        zero_intp = np.zeros(nr-2)
        zero_intp[0] = -1.0
        zero_intp[nr-3] = 1.0
        for j in range(1,nr-3):
            zero_intp[j] = 0.5*(zeros[i][j-1]+zeros[i][j])

        Li = L[i]
        dLi = dL[i]
        d2Li = d2L[i]
        # print(zero_intp)

        for j in range(nr-3):
            zstart = zero_intp[j]
            zend = zero_intp[j+1]
            zzero = zeros[i][j]
            xint, yuint, ylint = get_fun_ub(x_nodes, Li, dLi, d2Li, x, zstart, zend, zzero)
            # print(xint)
            # print(yint)
            xnew[i,4*j:4*j+5] = xint
            nupper[i,4*j:4*j+5] = yuint
            nlower[i,4*j:4*j+5] = ylint

    plot_functions(L, dL, d2L, x_nodes, x, x_values, N,
                   nlower, nupper, xnew, True, 'bounds')

    xtar = ChebyshevPoints(M)
    # now we can combine all the bounds
    xtemp = np.sort(np.unique(xnew[:,:]))
    # insert the target set of points
    xtemp = np.sort(np.unique(np.concatenate((xtemp, xtar))))
    x_temp = remove_duplicates2(xtemp, 1e-12)
    xnew2 = np.zeros((nr, len(xtemp)))
    nlower2 = np.zeros((nr, len(xtemp)))
    nupper2 = np.zeros((nr, len(xtemp)))
    for i in range(nr):
        nlower2[i,:] = np.interp(xtemp, xnew[i,:], nlower[i,:])
        nupper2[i,:] = np.interp(xtemp, xnew[i,:], nupper[i,:])
        xnew2[i,:] = xtemp

    plot_functions(L, dL, d2L, x_nodes, x, x_values, N,
                   nlower2, nupper2, xnew2, True, 'boundssample')

    # now we decimate unneeded points
    mr = len(xtar)
    xnew3 = np.zeros((nr, mr))
    nlower3 = np.zeros((nr, mr))
    nupper3 = np.zeros((nr, mr))

    print(xtemp)
    print(xtar)


    for i in range(nr):
        # print(i,"++++++++++++++++++++++")
        nlower3[i,:] = get_combined_pl_bounds(xnew2[i,:], nlower2[i,:], xtar, False)
        nupper3[i,:] = get_combined_pl_bounds(xnew2[i,:], nupper2[i,:], xtar, True)
        xnew3[i,:] = xtar
        # if (i == 1):
            # input("Press Enter to continue...")

    plot_functions(L, dL, d2L, x_nodes, x, x_values, N,
                   nlower3, nupper3, xnew3, True, 'combined', nlower2, nupper2, xnew2, xnew)






if __name__=="__main__":
    main()