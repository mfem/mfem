import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 16,
                     'text.usetex' : True})

filenames = [
    "../out_iter/iters_model4_pc5_cyc0_it10.txt",
    "../out_iter/iters_model2_pc5_cyc1_it10.txt",
    "../out_iter/iters_model1_pc5_cyc1_it10.txt"]

titles = ['Luxon and Brown', 'Taylor State', '15MA ITER']

plt.figure(figsize=(12, 5))
count = 1
for filename in filenames:
    print(filename)
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

    ax = plt.subplot(1, 3, count)
    data = np.array(data)
    uamr = np.unique(amr)
    for i in uamr:
        inds = np.where(amr == i)[0]
        if i == 0:
            Nx = len(inds) + 1
        try:
            plt.plot(data[inds], '.--')
        except:
            breakpoint()
    plt.yscale('log')
    plt.legend(uamr)
    plt.xlabel("iteration")
    plt.xticks(np.arange(Nx-1))
    
    ax.annotate(titles[count-1], xy=(0.5,1.05), xycoords='axes fraction',
                size=14,
                bbox=dict(boxstyle="round", fc=(0, 0, 1, .2), ec=(0, 0, 0)), ha='center')
    if count == 2:
        ax.annotate("Nonlinear Iterations", xy=(0.5,1.2), xycoords='axes fraction',
                    size=18,
                    bbox=dict(boxstyle="round", fc=(0, 0, 1, .0), ec=(0, 0, 0, 0)), ha='center')
        

    plt.tight_layout()
    count += 1
plt.savefig("../figs/taylor_newton.png", dpi=200)
plt.show()
        
        
