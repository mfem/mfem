import numpy as np
import matplotlib.pyplot as plt

with open("../out/03_spy.txt", 'r') as fid:
    st = fid.readline().replace('\n', '')
    N = eval(st.split('=')[-1])

    I = []
    J = []
    A = []
    for line in fid:
        row = line.replace(' \n', '').replace('i=', '').replace('j=', '').replace('a=', '').split(',')
        I.append(eval(row[0]))
        J.append(eval(row[1]))
        A.append(eval(row[2]))

    Mat = np.zeros((N, N))
    Mat[I, J] = A

    plt.spy(Mat)
    plt.tight_layout()
    plt.savefig('../out/03_spy.png', dpi=200)
    

