import numpy as np
from numpy.random import default_rng

"""

    Optimal truss design:

    min_x  E [ g(x) ]

    s.t.   A ≤ x_k ≤ B, \sum_k x_k ≤ C

"""

class truss():

    """
        Oracle for the optimal truss design problem
    """
    def __init__(self, **kwargs):
        try:
            seed = kwargs.get('seed')
            self.rng = default_rng(seed=seed)
        except:
            self.rng = default_rng()
        self.size = 7
        self._A = 0.5
        self._B = 2
        self._C = 9
        self._c = np.array([1,1,2,2,2,2,2])/(2*np.sqrt(3))
        self._x = np.asarray(kwargs.get('x0', (self._B - self._A)/2*np.ones(self.size)))
        self._s = np.array([0.0])
        self._lam = kwargs.get('lam0', 0.0)
        self.InitStressParams()

    def InitStressParams(self):
        # Y = log(X) ~ N(μ,σ)
        # E[Y_i] = log(μ_i) - 1/2 log(1 + cv_i^2)
        # V[Y_i] = log(1 + cv_i^2)
        # Cov(Y_i,Y_j) = log(1 + Corr(X_i,X_j) cv_i cv_j), where cv_i = μ_i/σ_i
        mu    = np.array([100,100,200,200,200,200,200])
        sigma = np.array([20,20,40,40,40,40,40])
        Corr  = np.array([[1.0,0.8,0.5,0.5,0.5,0.5,0.5],
                          [0.8,1.0,0.8,0.8,0.8,0.8,0.8],
                          [0.5,0.8,1.0,0.8,0.8,0.8,0.8],
                          [0.5,0.8,0.8,1.0,0.8,0.8,0.8],
                          [0.5,0.8,0.8,0.8,1.0,0.8,0.8],
                          [0.5,0.8,0.8,0.8,0.8,1.0,0.8],
                          [0.5,0.8,0.8,0.8,0.8,0.8,1.0]])
        cv = sigma/mu
        self.mean_Gaussian = np.log(mu) - 1/2 * np.log(1 + cv**2)
        Cov_Gaussian  = np.zeros_like(Corr)
        for i in range(self.size):
            for j in range(self.size):
                Cov_Gaussian[i,j] = np.log(1 + Corr[i,j] * cv[i] * cv[j])
        self.L_Gaussian = np.linalg.cholesky(Cov_Gaussian)
    
    def SampleStresses(self):
        y = self.rng.normal(size=self.size)
        y = self.L_Gaussian @ y
        y += self.mean_Gaussian
        return np.exp(y)

    def SampleLoad(self):
        mu    = 1e2
        sigma = 4e1
        cv = sigma/mu
        mean_y = np.log(mu) - 1/2 * np.log(1 + cv**2)
        std_y  = np.sqrt(np.log(1 + cv**2))
        return np.exp(self.rng.normal(mean_y,std_y))

    @property
    def x(self):
        return np.hstack((self._x, self._s))

    @x.setter
    def x(self, new_x):
        new_x = np.array(new_x)
        self._x = np.clip(new_x[:-1], self._A, self._B)
        self._s = max(0.0,new_x[-1]) # slack variable s ≥ 0
        return np.hstack((self._x, self._s))

    @property
    def lam(self):
        return self._lam

    @lam.setter
    def lam(self, new_lam):
        self._lam = new_lam
        return self._lam

    def sample(self):
        self.index = self.rng.integers(0,self.size)
        self.load = self.SampleLoad()
        self.stress = self.SampleStresses()[self.index]

    def f(self):
        self.sample()
        return self.load / self._c[self.index] - self.stress * self._x[self.index]

    def gradf(self):
        self.sample()
        gradf_x = np.zeros(self.size)
        gradf_x[self.index] = -self.stress
        return np.hstack((gradf_x,[0]))

    def G(self):
        # return self._C - np.sum(self._x) 
        return np.sum(self._x) - self._C + self._s

    def gradG(self):
        # return np.hstack((-self._x,[0.0]))
        return np.hstack((self._x,[1.0]))

    def L(self, beta=0.0):
        return self.f() - self.lam * self.G() + beta/2.0 * (self.G() ** 2)

    def gradL(self, beta=0.0):
        return self.gradf() + (beta * self.G() - self.lam) * self.gradG()

"""

    Optimal truss design:

    min_x  CVaR_γ [ g(x) ]

    s.t.   A ≤ x_k ≤ B, \sum_k x_k ≤ C

"""

class truss_CVaR(truss):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._t = kwargs.get('t0', 0.0)
        self.gamma = kwargs.get('gamma', 0.5)
        self.x_hist = []

    def update_history(self):
        self.x_hist.append(self.x)

    @property
    def x(self):
        return np.hstack((self._x, self._t, self._s))

    @x.setter
    def x(self, new_x):
        new_x = np.array(new_x)
        self._x = np.clip(new_x[:-2], self._A, self._B)
        self._t = new_x[-2]          # auxiliary variable t
        self._s = max(0.0,new_x[-1]) # slack variable s ≥ 0
        return np.hstack((self._x, self._t, self._s))

    @property
    def lam(self):
        return self._lam

    @lam.setter
    def lam(self, new_lam):
        self._lam = new_lam
        return self._lam

    def h(self, x):
        return self.load / self._c[self.index] - self.stress * x[self.index]

    def f(self):
        tmp = self.h(self._x) - self._t
        return self._t + 1/(1-self.gamma) * max(0.0,tmp)
    
    def gradG(self):
        return np.hstack((self._x,[0,1]))

    def phi(self, beta):
        self.sample()
        mu = 0.0
        n = len(self.x_hist)
        for i in range(n):
            x = self.x_hist[i][:-2]
            t = self.x_hist[i][-2]
            mu += beta * (self.h(x) - t)
            mu = np.clip(mu,0,1)
        tmp1 = self.h(self._x) - self._t
        tmp2 = beta * tmp1 + mu
        if tmp2 < 0:
            phi = -mu**2 / 2 / beta
        elif 0 <= tmp2 < 1:
            phi = beta * tmp1**2 / 2 + mu * tmp1
        else:
            phi = (tmp2 - (mu**2 + 1)/2) / beta
        return phi

    def gradphi(self, beta):
        self.sample()
        mu = 0.0
        n = len(self.x_hist)
        for i in range(n):
            x = self.x_hist[i][:-2]
            t = self.x_hist[i][-2]
            mu += beta * (self.h(x) - t)
            mu = np.clip(mu,0,1)
        tmp1 = self.h(self._x) - self._t
        tmp2 = beta * tmp1 + mu
        if tmp2 < 0:
            dphidh = 0.0
        elif 0 <= tmp2 < 1:
            dphidh = tmp2
        else:
            dphidh = 1.0
        dphidx = np.zeros_like(self._x)
        dphidx[self.index] = dphidh * (-self.stress)
        dphidt = dphidh * (-1)
        return np.hstack((dphidx,[dphidt],[0]))

    def L(self, beta=0.0):
        return self._t + 1/(1-self.gamma) * self.phi(beta) - self.lam * self.G() + beta/2.0 * (self.G() ** 2)

    def gradL(self, beta=0.0):
        tmp = np.zeros_like(self.x)
        tmp[-2] = 1.0
        return tmp + 1/(1-self.gamma) * self.gradphi(beta) + (beta * self.G() - self.lam) * self.gradG()

def test():
    config = {
        'lam0' : 0.0
    }

    # prob = truss_BPoF(**config)
    prob = truss(**config)
    alpha1 = 1e-3
    alpha2 = 1e-3
    tol1 = 1e0
    tol2 = 1e0
    dist2 = 10*tol2
    while dist2 > tol2:
        dist1 = 10*tol1
        while dist1 > tol1:
            tmp1 = prob.x
            # prob.x = prob.x * np.exp( -alpha1 * prob.gradL(alpha2) )
            prob.x = prob.x - alpha1 * prob.gradL(alpha2)
            dist1 = np.linalg.norm(tmp1 - prob.x) / alpha1
            print('L(x_k) = ', prob.L(alpha2))
            print('dist1  = ', dist1)
        tmp2 = prob.lam
        prob.lam = prob.lam - alpha2 * prob.G()
        dist2 = np.linalg.norm(tmp2 - prob.lam) / alpha2
        print('F(x_k) = ', prob.f())
        print('G(x_k) = ', prob.G())
        print('lambda = ', prob.lam)
        print('dist2  = ', dist2)
        print('x_k    = ', prob.x[:-1])
        print('s      = ', prob.x[-1])

if __name__ == '__main__':
	test()