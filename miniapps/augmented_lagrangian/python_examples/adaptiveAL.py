from statistics import variance
import numpy as np
from classification import *
from truss import *
from pylab import *
from math import ceil

class AL():
    """

        Augmented Lagrangian

    """
    def __init__(self, **kwargs):
        self.prob = kwargs.get('prob')
        self.alpha = kwargs.get('alpha', 1e-1)
        self.beta = kwargs.get('beta', 1e-1)
        self.primal_tol = kwargs.get('primal_tol', 1e-3)
        self.dual_tol = kwargs.get('dual_tol', 1e-3)
        self.print_primal = kwargs.get('print_primal', False)
        self.history = kwargs.get('history', False)
        if self.history:
            self.F_hist = []
            self.G_hist = []

    def __call__(self):
        done = False
        primal_err = self.primal_update()
        if self.print_primal:
            print('primal_err = ',primal_err)
        if primal_err < self.primal_tol:
            dual_err = self.dual_update()
            print('dual_err = ',dual_err)
            if dual_err < self.dual_tol:
                done = True
        if self.history:
            self.F_hist.append(self.F())
            self.G_hist.append(self.G())
        return done

    def F(self):
        F = 0.0
        try:
            for _ in range(self.prob.dataset_size):
                F += self.prob.f()
            return F / self.prob.dataset_size
        except:
            return self.prob.f()
    
    def G(self):
        return self.prob.G()

    def primal_update(self):
        tmp = self.prob.x
        self.prob.x = self.prob.x - self.alpha * self.prob.gradL(self.beta)
        return np.linalg.norm(tmp - self.prob.x) / self.alpha

    def dual_update(self):
        if hasattr(self.prob, "update_history"):
            self.prob.update_history()
        tmp = self.prob.lam
        self.prob.lam = self.prob.lam - self.beta * self.prob.G()
        return np.linalg.norm(tmp - self.prob.lam) / self.beta


class adaptiveAL(AL):
    """

        Augmented Lagrangian with adaptive sampling

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.theta = kwargs.get('theta', 0.5)
        self.sample_size = kwargs.get('sample_size', 10)
        assert(self.sample_size > 1)
        if self.history:
            self.sample_size_hist = []
            self.rho_hist = []

    def __call__(self):
        done = super().__call__()
        if self.history:
            self.sample_size_hist.append(self.sample_size)
            self.rho_hist.append(self.rho)
        return done

    def F(self):
        F = 0.0
        try:
            for _ in range(self.prob.dataset_size):
                F += self.prob.f()
            return F / self.prob.dataset_size
        except:
            for _ in range(self.sample_size):
                F += self.prob.f()
            return F / self.sample_size

    def primal_update(self):
        mean_gradL, std = self.mean_gradL()
        tmp = self.prob.x
        self.prob.x = self.prob.x - self.alpha * mean_gradL
        norm_reduced_gradient = np.linalg.norm(tmp - self.prob.x) / self.alpha + 1e-12
        self.rho = min(100, (std/norm_reduced_gradient/self.theta)**2)
        if self.rho > 1:
            self.sample_size = ceil(self.rho*self.sample_size)
        elif self.rho < 0.5 and self.sample_size > 100:
            self.sample_size = ceil(max(2, self.rho*self.sample_size))
        if self.print_primal:
            # print('std         = ',std)
            print('rho         = ',self.rho)
            print('sample_size = ',self.sample_size)
        return norm_reduced_gradient

    # def dual_update(self):
    #     err = AL.dual_update(self)
    #     self.sample_size = ceil(max(2, self.sample_size/2))
    #     return err

    def mean_gradL(self):
        mean_gradL = np.zeros_like(self.prob.x)
        mean_norm2_gradL = 0.0
        for _ in range(self.sample_size):
            gradL = self.prob.gradL(self.beta)
            mean_gradL += gradL
            mean_norm2_gradL += np.dot(gradL,gradL)
        mean_gradL /= self.sample_size
        mean_norm2_gradL /= self.sample_size
        norm2_mean_gradL = np.dot(mean_gradL,mean_gradL)
        variance = (mean_norm2_gradL - norm2_mean_gradL)/(self.sample_size-1)
        return mean_gradL, np.sqrt(variance)


def run():
    prob_config = {
        'seed'   : 4000
    }
    # config = {
    #     'prob'         : truss_CVaR(**prob_config),
    #     'alpha'        : 1e-2,
    #     'beta'         : 1e0,
    #     'primal_tol'   : 2e0,
    #     'dual_tol'     : 5e-1,
    #     'print_primal' : True,
    #     'theta'        : 0.9,
    #     'sample_size'  : 100,
    #     'history'      : True
    # }
    # config = {
    #     'prob'         : truss(**prob_config),
    #     'alpha'        : 1e-2,
    #     'beta'         : 1e1,
    #     'primal_tol'   : 2e0,
    #     'dual_tol'     : 1e-1,
    #     'print_primal' : True,
    #     'theta'        : 0.9,
    #     'sample_size'  : 4,
    #     'history'      : True
    # }
    config = {
        'prob'         : classification(**prob_config),
        'alpha'        : 1e-1,
        'beta'         : 1e-2,
        'primal_tol'   : 5e-1,
        'dual_tol'     : 5e-4,
        'print_primal' : True,
        'theta'        : 0.5,
        'sample_size'  : 4,
        'history'      : True
    }
    # al_step = AL(**config)
    al_step = adaptiveAL(**config)
    while True:
        done = al_step()
        if done:
            print('F(x_opt) = ', al_step.F())
            print('G(x_opt) = ', al_step.G())
            print('x_opt    = ', al_step.prob.x)
            break
    return np.array(al_step.F_hist), np.array(al_step.G_hist), np.array(al_step.sample_size_hist), np.array(al_step.rho_hist)

def plot_figs(F,G,S,savefigs=False):
    cumulS = np.cumsum(S)
    
    F = np.abs(F - F[-1])

    try:
        G = np.linalg.norm(G,axis=1)
    except:
        G = np.abs(G)
    
    figure()
    # plot(F,label=r'$F(x_k)$')
    semilogy(F,label=r'$|F(x_k)-F(x^\ast)|$')
    xlabel(r'Iteration')
    ylabel(r'$|F(x_k)-F(x^\ast)|$')
    # legend()
    if savefigs:
        savefig('F_vs_k.png', dpi='figure')

    figure()
    semilogy(G,label=r'$\|G(x_k)\|$')
    xlabel(r'Iteration')
    ylabel(r'$\|G(x_k)\|$')
    # legend()
    if savefigs:
        savefig('G_vs_k.png', dpi='figure')
    
    figure()
    semilogy(S,label=r'|S_k|')
    xlabel(r'Iteration')
    ylabel(r'Sample size')
    # legend()
    if savefigs:
        savefig('S_vs_k.png', dpi='figure')

    figure()
    # plot(cumulS,F,label=r'$F(x_k)$')
    semilogy(cumulS,F,label=r'$|F(x_k)-F(x^\ast)|$')
    xlabel(r'Effective gradient evaluations')
    ylabel(r'$|F(x_k)-F(x^\ast)|$')
    # legend()
    if savefigs:
        savefig('F_vs_grads.png', dpi='figure')
    

    show()


if __name__ == '__main__':
    savefigs = True
    F, G, S, rho = run()
    plot_figs(F,G,S,savefigs=savefigs)