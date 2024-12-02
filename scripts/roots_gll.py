import numpy as np
from scipy.special import legendre
from sympy import symbols, lambdify, diff, solve
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import argparse
from scipy.optimize import root
import sys
from scipy.optimize import minimize_scalar, minimize, fsolve,  least_squares

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

def plot_functions(L, dL, d2L, x_nodes, x_symbolic, x_values, N, lower, upper, int_points, piecewise=False, suffix='', lower2=None, upper2=None, int_points2=None, int_points3=None, zeros=None):
    # Plot the Lagrange basis functions, their first derivatives, and their second derivatives
    pdf_pages = PdfPages('ProvableBounds'+suffix+'_N='+str(N)+'_M='+str(len(int_points[0,:]))+'.pdf')
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

    # for i, (Li, dLi, d2Li) in enumerate(zip(L, dL, d2L)):
    #     plt.figure()
    #     # Plot the Lagrange basis function
    #     Li_func = lambdify(x_symbolic, Li, 'numpy')
    #     plt.plot(x_values, Li_func(x_values), label=f'L{i}')
    #     dLi_func = lambdify(x_symbolic, dLi, 'numpy')
    #     dLi_vals = dLi_func(x_values)
    #     dLi_vals = dLi_vals/np.max(np.abs(dLi_vals))
    #     plt.plot(x_values, dLi_vals, label=f"L'{i}")
    #     d2Li_func = lambdify(x_symbolic, d2Li, 'numpy')
    #     d2Li_vals = d2Li_func(x_values)
    #     d2Li_vals = d2Li_vals/np.max(np.abs(d2Li_vals))
    #     plt.plot(x_values, d2Li_vals, label=f"L''{i}")
    #     plt.xlabel('x')
    #     # plt.ylabel('L(x)')
    #     plt.plot(x_nodes, 0*x_nodes, 'ko-')
    #     plt.legend()
    #     plt.grid(True)

    #     plt.tight_layout()
    #     pdf_pages.savefig()
    #     plt.close()
    #     # plt.show()

    # plot piecewise bases
    ms = 1.0
    if piecewise:
        for i, (Li, dLi, d2Li) in enumerate(zip(L, dL, d2L)):
            plt.figure()
            # Plot the Lagrange basis function
            Li_func = lambdify(x_symbolic, Li, 'numpy')
            plt.plot(x_values, Li_func(x_values), label=f'L{i}')
            plt.plot(int_points[i,:], lower[i, :], 'rx-', linewidth=1.5, markersize=ms,label='Lower bound')
            plt.plot(int_points[i,:], upper[i, :], 'bx-', linewidth=1.5, markersize=ms,label='Upper bound')
            if lower2 is not None:
                plt.plot(int_points2[i,:], lower2[i, :], 'ro--', linewidth=1.5, markersize=ms,label='Lower bound')
                plt.plot(int_points2[i,:], upper2[i, :], 'bo--', linewidth=1.5, markersize=ms,label='Upper bound')
            plt.xlabel('x')
            # plt.ylabel('L(x)')
            plt.plot(x_nodes, 0*x_nodes, 'ko-',markerfacecolor='none',label='GLL')
            plt.plot(int_points[i,:], 0*int_points[i,:], 'kx',label='Interval points')
            if int_points3 is not None:
                plt.plot(int_points3[i,:], 0*int_points3[i,:], 'ks-',markerfacecolor='none')
            if zeros is not None:
                plt.plot(zeros[i,:], 0*zeros[i,:], 'ms',markerfacecolor='none',label='zeros')
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

def plot_roots_grouping(x_nodes, zeros, suffix=''):
    N = len(x_nodes)
    pdf_pages = PdfPages('RootGroups'+suffix+'_'+str(N)+'.pdf')
    plt.figure()
    plt.plot(x_nodes, 0*x_nodes,'ko-')
    plt.title(f'N={N}')
    for i in range(N-3):
        zero = np.sort(zeros[:,i])
        zerov = np.ones(np.shape(zero))
        # zerov += i
        plt.plot(zero,zerov,marker='o',markersize=3)


    plt.xlabel('x')
    plt.xticks(x_nodes, rotation=90)
    plt.yticks([])
    plt.grid(True)
    plt.tight_layout()
    pdf_pages.savefig()
    plt.close()
    pdf_pages.close()

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

# line from x_l that is tangent to the curve between x_c and x_r is:
# x_l + dLi(x_t)*(x-x_l), where x_c <= x_t <= x_r
# thus, dline_r = y_line_r - y_r = x_l + dLi(x_t)*(x_r-x_l) - y_r
# thus, dline_c = y_line_x_c - y_r = x_l + dLi(x_t)*(x_c-x_l) - y_r
# objective = dline_r^2
# constraint #1 dline_r > 0. so penalty = min(0, 100*dline_r)
# constraint #2 dline_c > 0. so penalty = min(0, 100*dline_c)
def convex_concave_objective1(x_t, Li, dLi, x_l, x_r, x_c):
    # Compute slope at x_t
    m = dLi(x_t)
    # Compute y1
    dline = m * (x_r - x_l) + Li(x_l) - Li(x_r)
    return dline*dline
    # Check inequality at xc
    # dline_c = m * (x_c - x_l) + Li(x_l)
    # dline_r = m * (x_r - x_l) + Li(x_l)
    # penalty = y1*y1 + min(0, 100*dline_r) + min(0, 100*dline_c)
    # return penalty

# def constraint_r(x_t, Li, dLi, x_l, x_r, x_c):
#     m = dLi(x_t)
#     dline_r = m * (x_r - x_l) + Li(x_l) - Li(x_r)
#     return dline_r  # Must be > 0

# def constraint_c(x_t, Li, dLi, x_l, x_r, x_c):
#     m = dLi(x_t)
#     dline_c = m * (x_c - x_l) + Li(x_l) - Li(x_c)
#     return dline_c  # Must be > 0

# def constraint_t(x_t, Li, dLi, x_l, x_r, x_c):
#     m = dLi(x_t)
#     dline_t = m * (x_t - x_l) + Li(x_l) - Li(x_t)
#     return dline_t  # Must be > 0

# Find the line that goes through x_l, Li(x_l) and is tangent to curve at x_t.
def equation_to_solve(x_t, x_l, Li, dLi):
    return Li(x_l) - (Li(x_t) + dLi(x_t) * (x_l - x_t))

# bound Li polynomial between x0 and x1. Assumes that d2Li=0 exactly once in
# this interval
def get_poly_convex_concave_bounds(x_nodes, Li, dLi, d2Li, x_l, x_r, x_c,ith,jth):
    fpp = d2Li(x_l)
    lower_l = 0.0
    upper_l = 0.0
    lower_r = 0.0
    upper_r = 0.0
    dL_l = dLi(x_l)
    dL_r = dLi(x_r)
    y_l = Li(x_l) #function value at x_l
    y_r = Li(x_r) #function value at x_r
    y_cp = Li(x_c) #function value at d2Li = 0
    # if (ith == 1):
    #     print(x_l, x_c, x_r, fpp, "k100")
    if fpp > 0: #first convex then concave
        #best case scenario is that the line connecting the two endpoints
        #is sufficient
        y_c_u = y_l + (y_r-y_l)*(x_c-x_l)/(x_r-x_l)
        if (y_r-y_l)/(x_r-x_l) <  dL_r and y_c_u > y_cp:
            upper_l = y_l
            upper_r = y_r
        else:
            upper_c1 = y_r + dL_r*(x_l-x_r) #line tangent from right end point
            length1 = upper_c1 - y_l

            #find the line that would go from left end point to the right
            #and be tangent to the curve somewhere between x_c and x_r
            x_guess = (x_c+x_r) / 2.0
            bounds = (x_c, x_r)
            result = least_squares(equation_to_solve, x_guess, args=(x_l, Li, dLi), bounds=bounds, xtol=1e-10)
            x_t = result.x[0]

            #check constraint
            dline_r =  Li(x_l) + dLi(x_t) * (x_r - x_l) - Li(x_r)
            dline_c =  Li(x_l) + dLi(x_t) * (x_c - x_l) - Li(x_c)

            if (x_t >= x_c and x_t <= x_r and dline_r >= 0 and dline_c >= 0):
                #candidate
                upper_c2 = y_l + dLi(x_t)*(x_r-x_l)
                length2 = upper_c2 - y_r
                if (length2 < length1 and length2 >= 0):
                    upper_l = y_l
                    upper_r = upper_c2
                else:
                    upper_l = upper_c1
                    upper_r = y_r
            else:
                upper_l = upper_c1
                upper_r = y_r

        # do lower bounds now
        # easy case is that the end points can be connected
        if (y_r-y_l)/(x_r-x_l) < dL_l and y_c_u < y_cp:
            lower_l = y_l
            lower_r = y_r
        else:
            #candidate 1 - start at x_l and tangent at x_l
            lower_c1 = y_l + dL_l*(x_r-x_l)
            length1 = y_r - lower_c1

            # candidate 2
            # starts at right and tangent somewhere between x_C and x_l
            x_guess = (x_l+x_c) / 2.0
            bounds = (x_l, x_c)
            result = least_squares(equation_to_solve, x_guess, args=(x_r, Li, dLi), bounds=bounds, xtol=1e-10)
            x_t = result.x[0]
            dline_r =  Li(x_r) + dLi(x_t) * (x_l - x_r) - Li(x_l)
            dline_c =  Li(x_r) + dLi(x_t) * (x_c - x_r) - Li(x_c)

            if (x_t >= x_l and x_t <= x_c and dline_r <= 0 and dline_c <= 0):
                #candidate
                lower_c2 = y_r + dLi(x_t)*(x_l-x_r)
                length2 = y_l - lower_c2
                if (length2 < length1 and length2 >= 0):
                    lower_l = lower_c2
                    lower_r = y_r
                elif (length1 >= 0):
                    lower_l = y_l
                    lower_r = lower_c1
                else:
                    sys.exit("Negative lengths")
            else:
                lower_l = y_l
                lower_r = lower_c1

    else: #first concave then convex case
        y_c_u = y_l + (y_r-y_l)*(x_c-x_l)/(x_r-x_l)
        if (y_r-y_l)/(x_r-x_l) > dL_l and y_c_u > y_cp:
            upper_l = y_l
            upper_r = y_r
        else:
            #we compare two lines
            #first is starts at x_l and is tangent at x_l
            upper_c1 = y_l + dL_l*(x_r-x_l)
            length1 = upper_c1 - y_r

            #second starts at x_r and is tangent at x_t somewhere in between x_l and x_c
            x_guess = (x_l+x_c) / 2.0
            bounds = (x_l, x_c)
            result = least_squares(equation_to_solve, x_guess, args=(x_r, Li, dLi), bounds=bounds, xtol=1e-10)
            x_t = result.x[0]
            # if (ith == 4 and jth == 1):
            #     print(x_t, "k102")
            #check constraint
            dline_r =  Li(x_r) + dLi(x_t) * (x_l - x_r) - Li(x_l)
            dline_c =  Li(x_r) + dLi(x_t) * (x_c - x_r) - Li(x_c)
            if (x_t >= x_l and x_t <= x_c and dline_r >= 0 and dline_c >= 0):
                    #candidate
                upper_c2 = y_r + dLi(x_t)*(x_l-x_r)
                length2 = upper_c2 - y_l
                if (length2 < length1):
                    upper_l = upper_c2
                    upper_r = y_r
                else:
                    upper_l = y_l
                    upper_r = upper_c1
            else:
                upper_l = y_l
                upper_r = upper_c1

        #best case scenario is that the line connecting the two endpoints
        #is sufficient
        y_c_u = y_l + (y_r-y_l)*(x_c-x_l)/(x_r-x_l)
        if (y_r-y_l)/(x_r-x_l) >  dL_r and y_c_u < y_cp:
            lower_l = y_l
            lower_r = y_r
        else:
            lower_c1 = y_r + dL_r*(x_l-x_r) #line tangent from right end point
            length1 = y_l - lower_c1

            #find the line that would go from left end point to the right
            #and be tangent to the curve somewhere between x_c and x_r
            x_guess = (x_c+x_r) / 2.0
            bounds = (x_c, x_r)
            result = least_squares(equation_to_solve, x_guess, args=(x_l, Li, dLi), bounds=bounds, xtol=1e-10)
            x_t = result.x[0]

            #check constraint
            dline_r =  Li(x_l) + dLi(x_t) * (x_r - x_l) - Li(x_r)
            dline_c =  Li(x_l) + dLi(x_t) * (x_c - x_l) - Li(x_c)

            if (x_t >= x_c and x_t <= x_r and dline_r <= 0 and dline_c <= 0):
                #candidate
                lower_c2 = y_l + dLi(x_t)*(x_r-x_l)
                length2 = y_r-lower_c2
                if (length2 < length1 and length2 >= 0):
                    lower_l = y_l
                    lower_r = lower_c2
                elif (length1 >= 0):
                    lower_l = lower_c1
                    upper_r = y_r
                else:
                    sys.exit("Negative lengths")
            else:
                lower_l = lower_c1
                upper_r = y_r

    return (lower_l, upper_l, lower_r, upper_r)



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

    # plot_functions(L, dL, d2L, x_nodes, x, x_values, N,
    #                lower, upper, int_points, False)


    # plot how roots are grouped
    zeros = np.array(zeros)
    plot_roots_grouping(x_nodes,zeros)

    # validate zeros
    roots_dist_threshold = 0.04
    for i in range(nr-4):
        zmax = np.max(zeros[:,i])
        zmin_next = np.min(zeros[:,i+1])
        if not zmin_next > zmax+roots_dist_threshold:
            print(i,zmax,zmin_next)
            sys.exit("Roots overlap")
    print("Roots validated")

    # generate interval points
    nzeros = nr-3
    nintervals = nr-2
    nintp = nr-2
    x_intp = np.zeros(nintp)
    x_intp[0] = -1.0
    x_intp[-1] = 1.0
    for i in range(1,nr-3):
        z1 = np.max(zeros[:,i-1])
        z2 = np.min(zeros[:,i])
        x_intp[i] = 0.5*(z1+z2)

    # compute bounds on the interval points
    nupper = array = np.full((nr,nintp), -10000.0)
    nlower = array = np.full((nr,nintp), 10000.0)
    xnewall = np.zeros((nr, nintp))
    for i in range(nr):
        xnewall[i,:] = x_intp
        Li = L[i]
        dLi = dL[i]
        d2Li = d2L[i]
        Li_func = lambdify(x, Li, 'numpy')
        dLi_func = lambdify(x, dLi, 'numpy')
        d2Li_func = lambdify(x, d2Li, 'numpy')
        for j in range(nr-3):
            x0 = x_intp[j]
            x1 = x_intp[j+1]
            xz = zeros[i,j]
            lower_l, upper_l, lower_r, upper_r = get_poly_convex_concave_bounds(x_nodes, Li_func, dLi_func, d2Li_func, x0, x1, xz,i,j)
            # if (i == 4 and j == 1):
                # print(lower_l, upper_l, lower_r, upper_r, "k10check")
            if j == 0:
                nlower[i, j] = lower_l
                nlower[i, j+1] = lower_r
                nupper[i, j] = upper_l
                nupper[i, j+1] = upper_r
            else:
                nlower[i, j] = min(lower_l, nlower[i, j])
                nlower[i, j+1] = min(lower_r, nlower[i, j+1])
                nupper[i, j] = max(upper_l, nupper[i, j])
                nupper[i, j+1] = max(upper_r, nupper[i, j+1])

    plot_functions(L, dL, d2L, x_nodes, x, x_values, N,
                   nlower, nupper, xnewall, True, '', None, None, None, None, zeros)

if __name__=="__main__":
    main()