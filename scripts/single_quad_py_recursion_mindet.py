import matplotlib.pyplot as plt
import numpy as np
import math

plt.rcParams['lines.markersize'] = 4

# Function to load data from a file
def load_data(filename, x_col, y_col, y_col2=None):
    data = np.loadtxt(filename)
    if y_col2 is None:
        return data[:, x_col], data[:, y_col]
    else:
        return data[:, x_col], data[:, y_col], data[:, y_col2]

#for M=4, there are 16 sub-zones. Each sub-zone will have 4 points.
#so we expect 64 values per level
def get_npatch_per_level(filename,M):
    data = np.loadtxt(filename)
    data = data[:,-1]
    data = np.abs(data)
    maxlevels = (int)(np.max(data))
    levels = range(1,maxlevels+1)
    ents   = np.zeros(maxlevels)
    count = 0
    for lev in levels:
        nents = np.count_nonzero(data == lev)
        val = nents/(M*M*4)
        val = math.ceil(val)
        ents[count] = val
        count += 1
    return levels, ents

# Load data for single_quad_bernstein_bounds
x1, y1, nel1 = load_data("/Users/mittal3/LLNL/mfem-detJ/mfem/examples/single_quad_bernstein_bounds.txt", 0, 2, 1)

# Load data for single_quad_custom_bounds4
x2, y2, y2b = load_data("/Users/mittal3/LLNL/mfem-detJ/mfem/examples/single_quad_custom_bounds4.txt", 0, 1, 2)
levels2, patches2 = get_npatch_per_level("/Users/mittal3/LLNL/mfem-detJ/mfem/examples/2DcustomboundinfoM4.txt",4)

# Load data for single_quad_custom_bounds5
x3, y3, y3b = load_data("/Users/mittal3/LLNL/mfem-detJ/mfem/examples/single_quad_custom_bounds5.txt", 0, 1, 2)
levels3, patches3 = get_npatch_per_level("/Users/mittal3/LLNL/mfem-detJ/mfem/examples/2DcustomboundinfoM5.txt",5)

# Load data for single_quad_custom_bounds6
x4, y4, y4b = load_data("/Users/mittal3/LLNL/mfem-detJ/mfem/examples/single_quad_custom_bounds6.txt", 0, 1, 2)
levels4, patches4 = get_npatch_per_level("/Users/mittal3/LLNL/mfem-detJ/mfem/examples/2DcustomboundinfoM6.txt",6)


x1a, y1a, y1ab = load_data("/Users/mittal3/LLNL/mfem-detJ/mfem/examples/single_quad_custom_bounds3.txt", 0, 1, 2)
levels1a, patches1a = get_npatch_per_level("/Users/mittal3/LLNL/mfem-detJ/mfem/examples/2DcustomboundinfoM3.txt",3)

# Plot the data
plt.figure(figsize=(8, 6))
plt.plot(x1, -y1, label='Bernstein', marker='o', color='red')

# plt.plot(x1a-1, -y1a, label='Propose approach ($M=N-1$)', marker='x')

plt.plot(x2-1, -y2, label='Propose approach ($M=N$)', marker='s')
# plt.plot(x2[:-1]-1, y2b[:-1], marker='s', linestyle='none', color='blue')
# plt.plot(x2[-1]-1, -y2b[-1], marker='s', linestyle='none', color='blue')

plt.plot(x3-1, -y3, label='Propose approach ($M=N+1$)', marker='^')
# plt.plot(x3[:-1]-1, y3b[:-1], marker='^', linestyle='none', color='green')
# plt.plot(x3[-1]-1, -y3b[-1], marker='^', linestyle='none', color='green')

plt.plot(x4-1, -y4, label='Propose approach ($M=N+2$)', marker='d')
plt.plot([np.min(x1),np.max(x1)],[0.000215656,0.000215656],'k-')

plt.yscale('log')
plt.gca().invert_yaxis()
yticks = plt.gca().get_yticks()
plt.yticks(yticks, -yticks)

# Labels and title
plt.xlabel("Recursion depth")
plt.ylabel("$\min\,\,{det}(J)$")
# plt.title("Comparison of Bounds")
plt.legend()
plt.grid()

outpathpre = "/Users/mittal3/LLNL/mfem-detJ/mfem/scripts/results/single_quad/"
plt.savefig(outpathpre+"single_quad_detj_comparison.png", dpi=300, bbox_inches="tight")


for i in range(len(x1)):
    nel = (int)(nel1[i])
    plt.text(x1[i], -y1[i], f'$N_E=${nel}', fontsize=12, ha='left', va='top')
plt.savefig(outpathpre+"single_quad_detj_comparison_1.png", dpi=300, bbox_inches="tight")

for i in range(len(x2)):
    nel = (int)(patches2[i])
    plt.text(x2[i]-1, -y2[i], f'{nel}', fontsize=12, ha='left', va='top')

for i in range(len(x3)):
    nel = (int)(patches3[i])
    plt.text(x3[i]-1, -y3[i], f'{nel}', fontsize=12, ha='left', va='top')

for i in range(len(x4)):
    nel = (int)(patches4[i])
    plt.text(x4[i]-1, -y4[i], f'{nel}', fontsize=12, ha='left', va='top')

plt.savefig(outpathpre+"single_quad_detj_comparison_2.png", dpi=300, bbox_inches="tight")

