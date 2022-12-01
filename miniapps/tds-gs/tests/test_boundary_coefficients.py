import numpy as np
import scipy as sp

"""
Get exact values for integrals used in unit tests
"""

rho_Gamma = 2.5
mu = 2

def M(x, y):
    """
    x, y: vectors [xr, xz] and [yr, yz]
    """

    xr = x[0]
    xz = x[1]
    yr = y[0]
    yz = y[1]
    if xr == 0 or yr == 0:
        return 0.0

    # kxy \in [0, 1]
    # kxy = 1 when x = y
    kxy = np.sqrt((4 * xr * yr) / ((xr + yr) ** 2 + (xz - yz) ** 2))

    tol = 1e-12
    if np.abs(kxy - 1) < tol:
        # avoid singularity
        return 0.0

    # complete elliptic integral of first kind
    # singular when kxy = 1
    K = sp.special.ellipk(kxy)
    # complete elliptic integral of second kind
    E = sp.special.ellipe(kxy)

    print("kxy", kxy)
    print("K", K)
    print("E", E)

    return kxy * ((2 - kxy ** 2) * E / (2 - 2 * kxy ** 2) - K) / (4 * np.pi * (xr * yr) ** 1.5 * mu)

def N(x):
    xr = x[0]
    xz = x[1]
    if xr == 0:
        return 0.0

    deltap = np.sqrt(xr ** 2 + (rho_Gamma + xz) ** 2)
    deltam = np.sqrt(xr ** 2 + (rho_Gamma - xz) ** 2)

    return (1 / deltap + 1 / deltam - 1 / rho_Gamma) / (xr * mu)
    

x = [2, 3]
y = [1, 2]
print("x =",x)
print("y =",y)
print("N(x) =",N(x))
print("N(y) =",N(y))
print("M(x,y) =",M(x, y))
print("M(y,x) =",M(x, y))
print("M(x,x)=",M(x, x))
print("M(y,y)=",M(y, y))
x[0] = 0
print("x =",x)
print("N(x) =",N(x))
print("M(x,y) =",M(x, y))
print("M(y,x) =",M(y, x))


breakpoint()
