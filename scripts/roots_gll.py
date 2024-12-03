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
	return np.sort([-np.cos(np.pi*i/(n-1)) for i in range(n)])

def gll_points(N):
    # Compute the GLL points
    roots = legendre(N-1).deriv().roots
    gll_points = np.concatenate(([-1], roots, [1]))
    return np.sort(gll_points)

def gl_points(N):
    points, weights = np.polynomial.legendre.leggauss(N)
    return np.sort(points)

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

def plot_functions(L, dL, d2L, x_nodes, x_symbolic, x_values, N, lower, upper, int_points, piecewise=False, suffix='', zeros=None):
    # Plot the Lagrange basis functions, their first derivatives, and their second derivatives
    pdf_pages = PdfPages('ProvableBounds'+suffix+'N='+str(N)+'_M='+str(len(int_points[0,:]))+'.pdf')
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

    # plot zero groupings
    if zeros is not None:
        plt.figure()
        plt.plot(x_nodes, 0*x_nodes,'ko-')
        plt.title(f'N={N}')
        for i in range(N-3):
            zero = np.sort(zeros[:,i])
            zerov = np.ones(np.shape(zero))
            # zerov += i
            plt.plot(zero,zerov,marker='o',markersize=3)
        plt.tight_layout()
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
            plt.xlabel('x')
            # plt.ylabel('L(x)')
            plt.plot(x_nodes, 0*x_nodes, 'ko-',markerfacecolor='none',label='GLL')
            plt.plot(int_points[i,:], 0*int_points[i,:], 'kx',label='Interval points')
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

#gslib approach for bounds
# x_nodes are nodes, L - basis function evaluation, dL - derivative evaluation
# x is the symbolic x,
#bi = basis index
# int_points = interval points location [-1,1]
def find_gslib_pl_bounds(x_nodes, L, dL, x, bi, int_points):
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

        lower[i] = min(bv, bmv + dm*dmv, bpv+dp*dpv)
        upper[i] = max(bv, bmv + dm*dmv, bpv+dp*dpv)

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
            if abs(xint[j]-xv) < 1e-14:
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
    # print(x_l,x_r,x_c,"k100")
    if fpp > 0: #first convex then concave
        #best case scenario is that the line connecting the two endpoints
        #is sufficient
        y_c_u = y_l + (y_r-y_l)*(x_c-x_l)/(x_r-x_l)
        # print(y_c_u, y_cp,(y_r-y_l)/(x_r-x_l),dL_r,"k101")
        if (y_r-y_l)/(x_r-x_l) <  dL_r and y_c_u > y_cp:
            # print( "k102")
            upper_l = y_l
            upper_r = y_r
        else:
            # print("k103")
            upper_c1 = y_r + dL_r*(x_l-x_r) #line tangent from right end point
            length1 = upper_c1 - y_l

            #find the line that would go from left end point to the right
            #and be tangent to the curve somewhere between x_c and x_r
            x_guess = (x_c+x_r) / 2.0
            bounds = (x_c, x_r)
            result = least_squares(equation_to_solve, x_guess, args=(x_l, Li, dLi), bounds=bounds, xtol=1e-14, ftol=1e-14, gtol=1e-14)
            x_t = result.x[0]

            #check constraint
            dline_r =  Li(x_l) + dLi(x_t) * (x_r - x_l) - Li(x_r)
            dline_c =  Li(x_l) + dLi(x_t) * (x_c - x_l) - Li(x_c)

            if (x_t >= x_c and x_t <= x_r and dline_r >= 0 and dline_c >= 0):
                # print("k104")
                #candidate
                upper_c2 = y_l + dLi(x_t)*(x_r-x_l)
                length2 = upper_c2 - y_r
                if (length2 < length1 and length2 >= 0):
                    # print("k105")
                    upper_l = y_l
                    upper_r = upper_c2
                else:
                    # print("k106")
                    print(ith,jth,"k101-upper-1")
                    upper_l = upper_c1
                    upper_r = y_r
            else:
                # yline_x_c = Li(x_l) + dLi(x_t) * (x_c - x_l)
                # print(result)
                # print(ith,jth,x_l,x_c,x_r,x_t,dline_r,dline_c,Li(x_c),yline_x_c,"k102")
                upper_l = upper_c1
                upper_r = y_r
                print(ith,jth,"k102-upper-1")

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
            result = least_squares(equation_to_solve, x_guess, args=(x_r, Li, dLi), bounds=bounds, xtol=1e-14, ftol=1e-14, gtol=1e-14)
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
                    print(ith,jth,"k101-lower-1")
                    lower_l = y_l
                    lower_r = lower_c1
                else:
                    sys.exit("Negative lengths")
            else:
                print(ith,jth,"k101-lower-2")
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
            result = least_squares(equation_to_solve, x_guess, args=(x_r, Li, dLi), bounds=bounds, xtol=1e-14, ftol=1e-14, gtol=1e-14)
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
                    print(ith,jth,"k101-upper-3")
            else:
                upper_l = y_l
                upper_r = upper_c1
                print(ith,jth,x_l,x_c,x_r,x_t,dline_r,dline_c,"k101-upper-4")

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
            result = least_squares(equation_to_solve, x_guess, args=(x_l, Li, dLi), bounds=bounds, xtol=1e-14, ftol=1e-14, gtol=1e-14)
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
                    print(ith,jth,"k101-lower-3")
                else:
                    sys.exit("Negative lengths")
            else:
                lower_l = lower_c1
                upper_r = y_r
                print(ith,jth,"k101-lower-4")

    return (lower_l, upper_l, lower_r, upper_r)

def get_convex_bounds(x_nodes, Li, dLi, d2Li, x_l, x_r, ith,jth):
    lower_l = 0.0
    upper_l = 0.0
    lower_r = 0.0
    upper_r = 0.0
    dL_l = dLi(x_l)
    dL_r = dLi(x_r)
    y_l = Li(x_l) #function value at x_l
    y_r = Li(x_r) #function value at x_r
    upper_l = y_l
    upper_r = y_r

    x_t = 0.5*(x_l+x_r)
    lower_l = Li(x_t) + dLi(x_t)*(x_l-x_t)
    lower_r = Li(x_t) + dLi(x_t)*(x_r-x_t)

    return (lower_l, upper_l, lower_r, upper_r)

def get_concave_bounds(x_nodes, Li, dLi, d2Li, x_l, x_r, ith,jth):
    lower_l = 0.0
    upper_l = 0.0
    lower_r = 0.0
    upper_r = 0.0
    dL_l = dLi(x_l)
    dL_r = dLi(x_r)
    y_l = Li(x_l) #function value at x_l
    y_r = Li(x_r) #function value at x_r
    lower_l = y_l
    lower_r = y_r

    x_t = 0.5*(x_l+x_r)
    upper_l = Li(x_t) + dLi(x_t)*(x_l-x_t)
    upper_r = Li(x_t) + dLi(x_t)*(x_r-x_t)

    return (lower_l, upper_l, lower_r, upper_r)



#finds roots of polynomial
#also gets PL-bounds
#M < 0, we use automated bounds with M = N-2 by grouping the zeros of second
#       derivative of Lagrange interpolants
# otherwise we take M points. The distribution of interval points depends on
# the argument inttype = 0 (GL), 1 (GLL), 2 (Chebyshev), 3 (for zeros of f"" based distribution)
# similarly node type = 0 (GL), 1 (GLL [default])
def main():
    # Number of GLL points
    parser = argparse.ArgumentParser(description="A script that processes some arguments")

    # Add arguments
    parser.add_argument('--N', type=int, help='Number of mesh nodes', required=True)
    parser.add_argument('--M', type=int, help='Number of interval points', default=-1)
    parser.add_argument('--itype', type=int, help='Interval point type', default=2)
    parser.add_argument('--ntype', type=int, help='Node type', default=1)
    args = parser.parse_args()
    N = args.N
    M = args.M
    IType = args.itype
    NType = args.ntype
    # print(N)

    # Step 1: Define the GLL points
    x_nodes = None
    if NType == 0:
        x_nodes = gl_points(N)
        print("GL Nodes")
    else:
        x_nodes = gll_points(N)
        print("GLL Nodes")
    # print(x_nodes)

    # Step 2: Construct the Lagrange interpolants
    L, x = lagrange_interpolants(x_nodes)

    # Step 3: Compute the first and second derivatives
    dL = first_derivative(L, x)
    d2L = second_derivative(L, x)

    # Step 4: Find the zeros of the second derivative
    # zeros = find_zeros(d2L, x)

    # Step 5: Plot the functions
    x_values = np.linspace(-1, 1, 1000)

    zeros = find_zeros(d2L, x, x_nodes)
    zeros = np.array(zeros)
    x_intp = None
    if M < 0:
        print("Computing interval points based on the zeros.")
        M = N-2
        IType = 3

        # Print the zeros
        print("Zeros of the second derivative of each Lagrange interpolant:")
        for i in range(N-3):
            print(f"Lagrange interpolant {i}: {zeros[i,:]}")

        # validate zeros
        roots_dist_threshold = 0.04
        for i in range(N-4):
            zmax = np.max(zeros[:,i])
            zmin_next = np.min(zeros[:,i+1])
            if not zmin_next > zmax+roots_dist_threshold:
                print(i,zmax,zmin_next)
                sys.exit("Roots overlap")
        print("Roots validated")

        # generate interval points
        x_intp = np.zeros(M)
        x_intp[0] = -1.0
        x_intp[-1] = 1.0
        for i in range(1,N-3):
            z1 = np.max(zeros[:,i-1])
            z2 = np.min(zeros[:,i])
            x_intp[i] = 0.5*(z1+z2)
    else:
        if (IType == 0):
            x_intp = np.sort(np.concatenate(([-1], gl_points(M-2), [1])))
            print("GL+end points as interval points.")
        elif (IType == 1):
            x_intp = gll_points(M)
            print("GLL points as interval points.")
        elif (IType == 2):
            x_intp = np.array(ChebyshevPoints(M))
            print("Chebyshev points as interval points.")
        print(x_intp)
        #validate that there is at-most 1 zero in each interval for each basis function
        for i in range(M-1):
            x0 = x_intp[i]
            x1 = x_intp[i+1]
            # print(i,x0,x1)
            nzeros = np.sum((zeros > x0) & (zeros < x1),axis=1)
            # print(nzeros)
            if np.max(nzeros) > 1:
                print(i,x0,x1)
                print(zeros)
                sys.exit("Not a valid point set. Increase M")

        # sys.exit("M < 0 only supported yet")


    # compute bounds on the interval points
    nupper = array = np.full((N,M), -10000.0)
    nlower = array = np.full((N,M), 10000.0)
    xnewall = np.zeros((N, M))
    for i in range(N):
        xnewall[i,:] = x_intp

    plot_functions(L, dL, d2L, x_nodes, x, x_values, N,
                   nlower*0.0, nupper*0.0, xnewall, True, '', zeros)

    for i in range(N):
        xnewall[i,:] = x_intp
        Li = L[i]
        dLi = dL[i]
        d2Li = d2L[i]
        Li_func = lambdify(x, Li, 'numpy')
        dLi_func = lambdify(x, dLi, 'numpy')
        d2Li_func = lambdify(x, d2Li, 'numpy')
        for j in range(M-1):
            x0 = x_intp[j]
            x1 = x_intp[j+1]
            nzeros = np.sum((zeros[i,:] > x0) & (zeros[i,:] < x1))
            # print(i,j,nzeros)
            if nzeros == 0:
                fpp = d2Li_func(0.5*(x0+x1))
                # if (i == 0 and j == 0):
                    # print(i,j,fpp,"k102")
                if (fpp > 0):
                    lower_l, upper_l, lower_r, upper_r = get_convex_bounds(x_nodes, Li_func, dLi_func, d2Li_func, x0, x1, i,j)
                elif (fpp < 0):
                    lower_l, upper_l, lower_r, upper_r = get_concave_bounds(x_nodes, Li_func, dLi_func, d2Li_func, x0, x1, i,j)
                else:
                    sys.exit("Accidentally hit d2f = 0")
            else:
                mask = (zeros[i,:] > x0) & (zeros[i,:] < x1)
                xz = zeros[i,mask]
                lower_l, upper_l, lower_r, upper_r = get_poly_convex_concave_bounds(x_nodes, Li_func, dLi_func, d2Li_func, x0, x1, xz,i,j)

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

    if (NType == 0):
        prefix = "GL_"
    else:
        prefix = "GLL_"

    if (IType == 0):
        prefix = prefix + "GL_"
    elif (IType == 1):
        prefix = prefix + "GLL_"
    elif (IType == 2):
        prefix = prefix + "Cheb_"
    elif (IType == 3):
        prefix = prefix + "Roots_"
    plot_functions(L, dL, d2L, x_nodes, x, x_values, N,
                   nlower, nupper, xnewall, True, prefix, zeros)


    #validate bounds
    npts = 1000000
    x_values = np.linspace(-1, 1, npts)
    delmax = 0
    for i in range(N):
        Li = L[i]
        Li_func = lambdify(x, Li, 'numpy')
        for j in range(npts):
            xv = x_values[j]
            yl = np.interp(xv, x_intp, nlower[i,:])
            yu = np.interp(xv, x_intp, nupper[i,:])
            yv = Li_func(xv)
            if (yv < yl):
                delmax = max(delmax,yl-yv)
                # print(i,j,xv,yl,yu,yv,yl-yv,yv-yu)
            elif  (yv > yu):
                delmax = max(delmax,yv-yu)
                # sys.exit("Piecewise linear bounds no good")

    print("PL Bounds violate constraints at-most by: ",delmax)
    sys.exit("Piecewise linear bounds are good")




if __name__=="__main__":
    main()