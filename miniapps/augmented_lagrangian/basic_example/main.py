import numpy as np
import numpy.random as random

"""

    Basic stochastic optimization problem

    min_x  { F(x) := E[ ξ_i x_i^2 ] | x \in C, G(x) = 0 }

    s.t.   G(x) := 1/N Σ_i x_i - 1
              C := { x | x_even ≤ 0.5 }

    where each ξ_i ~ Unif(0.5,1.5)
           
"""

class basic_example():

    """
        Oracle for the basic stochastic optimization problem

        size : dimension of design space x ∈ R^size
    """
    def __init__(self, **kwargs):
        self.size = kwargs.get('size', 100)
        self._x = np.asarray(kwargs.get('x0', np.zeros(self.size)))
        self._lam = kwargs.get('lam0', 0.0)
        self._xi = np.ones_like(self.x)

    @property
    def x(self):
        return self._x
    
    @x.setter
    def x(self, new_x):
        # new_x[::2]  = np.clip(new_x[::2], 0.0, None)
        new_x[1::2] = np.clip(new_x[1::2], None, 0.5)
        self._x = np.array(new_x)
        return self._x

    @property
    def lam(self):
        return self._lam
    
    @lam.setter
    def lam(self, new_lam):
        self._lam = new_lam
        return self._lam

    @property
    def xi(self):
        # self._xi = random.uniform(0.5,1.5,size=self.size)
        return self._xi

    def f(self):
        return np.inner(self.xi, self.x**2) / self.size
    
    def gradf(self):
        return 2.0 * self.xi * self.x / self.size

    def G(self):
        return np.mean(self.x) - 1.0
    
    def gradG(self):
        return np.ones(self.size) / self.size
    
    def L(self, alpha=0.0):
        return self.f() - self.lam * self.G() + alpha/2.0 * (self.G() ** 2)

    def gradL(self, alpha=0.0):
        return self.gradf() + (alpha * self.G() - self.lam) * self.gradG()



def test():
        size = 100
        config = {
            'size' : size,
            'x0'   : np.random.uniform(size=size),
            'lam0' : -1.0
        }

        prob = basic_example(**config)
        alpha1 = 1e-1
        alpha2 = 1e-1
        dist2 = 1.0
        tol1 = 1e-4
        tol2 = 1e-4
        while dist2 > tol2:
            dist1 = 1.0
            while dist1 > tol1:
                tmp1 = prob.x
                # prob.x = prob.x * np.exp( -alpha1 * prob.gradL(alpha2) )
                prob.x = prob.x - alpha1 * prob.gradL(alpha2)
                dist1 = np.linalg.norm(tmp1 - prob.x) / alpha1
                # print('L(x_k) = ', prob.L(alpha=alpha2))
                # print('dist1  = ', dist1)
            tmp2 = prob.lam
            prob.lam = prob.lam - alpha2 * prob.G()
            dist2 = np.linalg.norm(tmp2 - prob.lam) / alpha2
            print('F(x_k) = ', prob.f(), ',   G(x_k) = ', prob.G(), ',   lambda = ', prob.lam)
            # print('dist2  = ', dist2)



if __name__ == '__main__':
	test()