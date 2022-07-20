from re import M
import numpy as np
import numpy.random as random
from tensorflow.keras.datasets import mnist
from numpy.random import default_rng
from scipy.special import expit

"""

    Binary classification problem:

    min_x  E_{y,z}[ log(1 + exp(-z<x,y>)) ] + \gamma/2 ||x||^2

    s.t.   <a,x> = 0, |<b,x>| ≤ c

    where (y,z) come from the MNIST data set

"""

class classification():

    """
        Oracle for the disparate impact constrained classification problem
    """
    def __init__(self, **kwargs):
        try:
            seed = kwargs.get('seed')
            self.rng = default_rng(seed=seed)
        except:
            self.rng = default_rng()
        test = load_mushrooms()
        # test, _ = mnist.load_data()
        self.dataset_size = test[0].shape[0]
        self.test_y = np.array(test[0], dtype=np.float64).reshape((self.dataset_size,-1))
        self.test_z = np.array(test[1], dtype=np.float64)/10
        self.size = self.test_y.shape[-1]
        self._x = np.asarray(kwargs.get('x0', self.rng.standard_normal(self.size)/self.dataset_size))
        self._s = np.array([0.0])
        self._lam = kwargs.get('lam0', np.array([0.0,0.0]))
        self._gamma = kwargs.get('gamma', 1/self.dataset_size)
        self._a = np.asarray(kwargs.get('a', self.rng.standard_normal(self.size)))
        self._b = np.asarray(kwargs.get('b', self.rng.standard_normal(self.size)))
        # self._a = np.zeros_like(self._a)
        # self._b = np.zeros_like(self._b)
        self._c = kwargs.get('c', 0.2)
        assert(self._c > 0)

    @property
    def x(self):
        return np.hstack((self._x, self._s))

    @x.setter
    def x(self, new_x):
        new_x = np.array(new_x)
        self._x = new_x[:-1]
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
        index = self.rng.integers(0,self.dataset_size)
        self.y = self.test_y[index,:]
        self.z = self.test_z[index]

    def f(self):
        self.sample()
        tmp = self.z * np.dot(self._x,self.y)
        tmp = expit(tmp)
        return -np.log(tmp) + self._gamma/2 * np.dot(self._x,self._x)

    def gradf(self):
        self.sample()
        tmp = self.z * np.dot(self._x,self.y)
        gradf_x = -self.z * self.y * expit(-tmp) + self._gamma * self._x
        # gradf_x += self.rng.standard_normal(self.size)/self.size
        return np.hstack((gradf_x,[0]))

    def G(self):
        G1 = np.dot(self._a,self._x)
        G2 = (self._c - abs(np.dot(self._b,self._x))) - self._s
        return np.hstack((G1,G2))

    def gradG(self):
        gradG1 = np.hstack((self._a,[0]))
        sgn = np.sign(-np.dot(self._b,self._x))
        gradG2 = np.hstack((sgn*self._b,[-1]))
        return np.vstack((gradG1,gradG2))

    def L(self, beta=0.0):
        return self.f() - np.dot(self._lam,self.G()) + beta/2.0 * np.sum(self.G() ** 2)

    def gradL(self, beta=0.0):
        return self.gradf() + (np.transpose(beta * self.G() - self._lam) @ self.gradG())


def load_rcv1():
    filename = './augmented_lagrangian/python_examples/rcv1_train.txt'
    dataset_size = 20242
    size = 47236
    y, z = load_data(filename,dataset_size,size)
    z = (z+1)/2 # map {-1,1} to {0,1}
    return y, z

def load_mushrooms():
    filename = './augmented_lagrangian/python_examples/mushrooms.txt'
    dataset_size = 8124
    size = 112
    y, z = load_data(filename,dataset_size,size)
    z = z-1 # map {1,2} to {0,1}
    return y, z

def load_data(filename,dataset_size,size):
    with open(filename, 'r') as the_file:
        y = np.zeros((dataset_size,size))
        z = np.zeros(dataset_size)
        for i, line in enumerate(the_file):
            line = line.strip()
            tmp, line = line.split(' ',1)
            z[i] = np.int(tmp)
            j_val_list = line.split(' ')
            for j_val in j_val_list:
                j, val = j_val.split(':')
                y[i,int(j)-1] = np.float64(val)
    return y, z

def test():
    config = {
        'lam0' : [0.0, 0.0]
    }

    prob = classification(**config)
    alpha1 = 1e-2
    alpha2 = 1e-2
    dist2 = 1.0
    tol1 = 1e-4
    tol2 = 1e-5
    while dist2 > tol2:
        dist1 = 1.0
        while dist1 > tol1:
            tmp1 = prob.x
            # prob.x = prob.x * np.exp( -alpha1 * prob.gradL(alpha2) )
            prob.x = prob.x - alpha1 * prob.gradL(alpha2)
            dist1 = np.linalg.norm(tmp1 - prob.x) / alpha1
            print('L(x_k) = ', prob.L(alpha=alpha2))
            print('dist1  = ', dist1)
        tmp2 = prob.lam
        prob.lam = prob.lam - alpha2 * prob.G()
        dist2 = np.linalg.norm(tmp2 - prob.lam) / alpha2
        print('F(x_k) = ', prob.f())
        print('G(x_k) = ', prob.G())
        print('lambda = ', prob.lam)
        print('dist2  = ', dist2)
        # print('x_k[0] = ', prob.x[0])
        print('s      = ', prob.x[-1])

if __name__ == '__main__':
	test()