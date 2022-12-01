import numpy as np
import scipy as sp

"""
Get exact values for integrals used in unit tests
"""

def a(theta):
    k = 3.2
    return theta

def b(theta):
    return theta

def c(theta):
    k = 3.2
    return np.sin(k * theta) + theta

def d(theta):
    k = 2.3
    return np.cos(k * theta) + theta * theta

def M(theta, phi):
    return 1.0

def M2(theta, phi):
    return theta + phi

def integrand(theta, phi):
    R = 2.5
    return (a(theta)-a(phi)) * M(theta, phi) * (b(theta) - b(phi)) * R ** 2

def integrand2(theta, phi):
    R = 2.5
    return (c(theta)-c(phi)) * M2(theta, phi) * (d(theta) - d(phi)) * R ** 2

result = sp.integrate.dblquad(integrand, -np.pi/2, np.pi/2, lambda x: -np.pi/2, lambda x: np.pi/2)
print(result)
R = 2.5
print(R**2 * np.pi**4 / 6)
result2 = sp.integrate.dblquad(integrand2, -np.pi/2, np.pi/2, lambda x: -np.pi/2, lambda x: np.pi/2)
print(result2)

# print(np.pi**4 / 6)
breakpoint()
