import numpy as np
import matplotlib.pyplot as plt

with open("matrices.txt", "r") as fid:
    out = fid.readline()

    data = {}
    for i, name in zip([0, 1, 2], ["By", "B", "C"]):
        out = fid.readline()
        Mat = np.zeros((484, 484))
        while True:
            out = fid.readline()

            if "%" in out:
                break

            out = out[:-1].split(" ")
            out = [eval(a) for a in out]
            Mat[out[0] - 1, out[1] - 1] = out[2]

        data[name] = Mat

    for i, name in zip([0, 1], ["Cy", "Ba"]):
        Vec = []
        for out in fid:

            if "%" in out:
                break
            out = out[:-1].split(" ")
            out = [eval(a) for a in out]
            Vec += out
        data[name] = np.array(Vec)
        
B = data["B"] # -F H^{-1} F^T
C = data["C"] # G_yy
By = data["By"]
Cy = data["Cy"]
Ba = data["Ba"]

plt.rcParams.update({'font.size': 16,
                     'text.usetex' : True})

A = By - np.outer(Ba, Cy)
D = By.T - np.outer(Cy, Ba)

plt.figure()
plt.spy(A)
plt.title("$A$")
plt.tight_layout()

plt.figure()
plt.spy(B)
plt.title("$B$")
plt.tight_layout()

plt.figure()
plt.spy(C)
plt.title("$C$")
plt.tight_layout()

plt.figure()
plt.spy(D)
plt.title("$D$")
plt.tight_layout()

S = D - C @ np.linalg.inv(A) @ B
plt.figure()
plt.spy(S)
plt.title("$D - C A^{-1} B$")
plt.tight_layout()

plt.figure()
plt.spy(By)
plt.title("$B_y$")
plt.tight_layout()

plt.figure()
plt.spy(np.outer(Ba, Cy))
plt.title("$B_a C_y^T$")
plt.tight_layout()

###########

eig_D = np.linalg.eig(D)[0]
plt.figure()
plt.plot(np.real(eig_D), np.imag(eig_D), '.')
plt.title('$\\rho(D)$')
plt.xlabel("Re")
plt.ylabel("Im")
plt.tight_layout()

eig_S = np.linalg.eig(S)[0]
plt.figure()
plt.plot(np.real(eig_S), np.imag(eig_S), '.')
plt.title('$\\rho(S)$')
plt.xlabel("Re")
plt.ylabel("Im")
plt.tight_layout()


eig_S = np.linalg.eig(S)[0]
plt.figure()
plt.subplot(1, 4, 1)
plt.plot(np.real(eig_S), np.imag(eig_S), '.')
plt.title('$\\rho(S)$')
plt.xlabel("Re")
plt.ylabel("Im")


S_approx = By.T
PS = np.linalg.inv(S_approx) @ S
eig_PS = np.linalg.eig(PS)[0]
plt.subplot(1, 4, 2)
plt.plot(np.real(eig_PS), np.imag(eig_PS), '.')
plt.title('$\\rho((B_y^T)^{-1} S)$')
plt.xscale('log')
plt.xlabel("Re")
plt.ylabel("Im")

S_approx = D
PS = np.linalg.inv(S_approx) @ S
eig_PS = np.linalg.eig(PS)[0]
plt.subplot(1, 4, 3)
plt.plot(np.real(eig_PS), np.imag(eig_PS), '.')
plt.title('$\\rho((D)^{-1} S)$')
plt.xscale('log')
plt.xlabel("Re")
plt.ylabel("Im")

S_approx = D - C @ np.linalg.inv(np.diag(np.diag(A))) @ B
PS = np.linalg.inv(S_approx) @ S
eig_PS = np.linalg.eig(PS)[0]
plt.subplot(1, 4, 4)
plt.plot(np.real(eig_PS), np.imag(eig_PS), '.')
plt.title('$\\rho((D - C diag(A)^{-1} B)^{-1} S)$')
plt.xscale('log')
plt.xlabel("Re")
plt.ylabel("Im")


S_approx = D - C @ By @ B
PS = np.linalg.inv(S_approx) @ S
eig_PS = np.linalg.eig(PS)[0]
plt.figure()
plt.plot(np.real(eig_PS), np.imag(eig_PS), '.')
plt.title('$\\rho((D - C B_y B)^{-1} S)$')
plt.xscale('log')
plt.xlabel("Re")
plt.ylabel("Im")

plt.tight_layout()


lump = np.diag(np.sum(B, axis=1))
S_approx = D - C @ np.linalg.inv(By) @ lump
PS = np.linalg.inv(S_approx) @ S
eig_PS = np.linalg.eig(PS)[0]
plt.figure()
plt.plot(np.real(eig_PS), np.imag(eig_PS), '.')
plt.title('$\\rho((D - C B_y^{-1} lump(B))^{-1} S)$')
plt.xlabel("Re")
plt.ylabel("Im")
plt.tight_layout()


from scipy.sparse.linalg import spilu
lump = np.diag(np.sum(B, axis=1))
By_inv = spilu(By)
# S_approx = D - C @ By_inv.solve(lump)
S_approx = D - C @ By_inv.solve(B)
PS = np.linalg.inv(S_approx) @ S
eig_PS = np.linalg.eig(PS)[0]
plt.figure()
plt.plot(np.real(eig_PS), np.imag(eig_PS), '.')
plt.title('$\\rho((D - C ILU(B_y) B)^{-1} S)$')
plt.xlabel("Re")
plt.ylabel("Im")
plt.tight_layout()

plt.figure()
plt.spy(np.linalg.inv(By) @ lump)
plt.title("$B_y^{-1} lump(B)$")
plt.tight_layout()

plt.figure()
plt.spy(C @ np.linalg.inv(By) @ lump)
plt.title("$C B_y^{-1} lump(B)$")
plt.tight_layout()

plt.show()


