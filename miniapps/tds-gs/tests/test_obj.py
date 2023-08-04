import numpy as np
import matplotlib.pyplot as plt

psi_true_ = []
rz_bbs = []
with open("../separated_file.data", 'r') as fid:
    for line in fid:
        if "rbbbs(i),zbbbs(i)" in line:
            out = fid.readline()[:-2].split(" ")
            for num in out:
                rz_bbs.append(eval(num))
        if "psizr" in line:
            out = fid.readline()[:-2].split(" ")
            for num in out:
                psi_true_.append(eval(num))

rz_bbs = np.array(rz_bbs)

Nr = 257
Nz = 513

rmid = 6.5
rdim = 7
r0 = rmid - rdim / 2
r1 = rmid + rdim / 2
z0 = -6
z1 = 6

rv = np.linspace(r0, r1, Nr)
zv = np.linspace(z0, z1, Nz)

psi_true = np.zeros((Nz, Nr))
for j in range(Nz):
    for i in range(Nr):
        psi_true[j, i] = psi_true_[i + j * Nr]

RV, ZV = np.meshgrid(rv, zv)

# plt.contourf(RV, ZV, psi_true)
# plt.show()

psi = psi_true[0]

alpha = np.array(
    [[.2, .2, .6],
     [.3, .3, .4],
     [.4, .3, .3],
     [.9, .05, .05]])

Iv = np.array([[4, 5, 6],
               [50, 49, 51],
               [100, 101, 102],
               [200, 201, 202]])
# Jv = [[10, 11, 10],
#       [5, 3, 4],
#       [100, 101, 99],
#       [205, 206, 204]]

N_control = len(Iv)
i_x = 60
i_ma = 90
# j_x = 30
# j_ma = 200

def obj(psi):
    psi_ma = psi[i_ma]
    psi_x = psi[i_x]
    obj = 0
    for k in range(N_control):
        psi_interp = 0
        for m in range(3):
            psi_interp += alpha[k, m] * psi[Iv[k, m]]
        psi_N = (psi_interp - psi_ma) / (psi_x - psi_ma)
        obj += .5 * (psi_N - 1.0) ** 2
    return obj

def grad(psi):
    psi_ma = psi[i_ma]
    psi_x = psi[i_x]
    grad_obj = 0 * psi
    for k in range(N_control):
        psi_interp = 0
        for m in range(3):
            psi_interp += alpha[k, m] * psi[Iv[k, m]]
        psi_N = (psi_interp - psi_ma) / (psi_x - psi_ma)

        
        factor = 1 / (psi_x - psi_ma)
        for m in range(3):
            grad_obj[Iv[k, m]] += alpha[k, m] * factor * (psi_N - 1)

        # breakpoint()
        grad_obj[i_x] += (psi_ma - psi_interp) / (psi_x - psi_ma) ** 2 * (psi_N - 1)
        grad_obj[i_ma] += (- (psi_ma - psi_interp) / (psi_x - psi_ma) ** 2 - 1 / (psi_x - psi_ma)) * (psi_N - 1)

    return grad_obj

eps = 1e-4
grad_calc = grad(psi)
grad_FD = 0 * psi
obj_0 = obj(psi)
for i in range(len(psi)):
    psi[i] += eps
    obj_1 = obj(psi)
    psi[i] -= eps

    grad_FD[i] = (obj_1 - obj_0) / eps

    if grad_calc[i] != 0 or grad_FD[i] != 0:
        print("%d, calc: %e, FD: %e, diff: %e" % (i, grad_calc[i], grad_FD[i], grad_calc[i]-grad_FD[i]))





    
    


    
