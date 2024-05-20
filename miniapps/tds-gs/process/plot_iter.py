import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 16,
                     'text.usetex' : True})

filename = "../out_iter/iters_model2_pc5_cyc1_it5.txt"

fid = open(filename, 'r')

data = []
amr = []
for line in fid:
    if "nonl_res" in line:
        key1 = "nonl_res: "
        ind1 = line.index(key1)
        key2 = ", ratio"
        ind2 = line.index(key2)

        out = eval(line[ind1+len(key1):ind2])
        data.append(out)
    if "amr" in line:
        num = eval(line.split(' ')[0].split('=')[-1])
        if len(amr) > 0 and num != amr[-1]:
            amr.append(amr[-1])
        amr.append(num)
amr.append(amr[-1])
print(data)
print(amr)

data = np.array(data)
plt.figure()
uamr = np.unique(amr)
for i in uamr:
    inds = np.where(amr == i)[0]
    plt.plot(data[inds], '.--')
plt.yscale('log')
plt.legend(uamr)
plt.xlabel("iteration")
plt.title("nonlinear equation residual")
plt.tight_layout()
plt.savefig("../figs/taylor_newton.png", dpi=200)
plt.show()
        
        
