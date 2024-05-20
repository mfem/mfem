import numpy as np
import matplotlib.pyplot as plt


filename = "spy_model2_amr4.txt"
fid = open(filename, "r")

dataI = {}
dataJ = {}
dataA = {}
key = fid.readline()[:-1]
dataI[key] = []
dataJ[key] = []
dataA[key] = []
for line in fid:
    if line == "\n":
        key = fid.readline()[:-1]
        dataI[key] = []
        dataJ[key] = []
        dataA[key] = []
    else:
        out = line[:-1].split(" ")
        try:
            dataA[key].append(eval(out[2]))
            dataJ[key].append(eval(out[1]))
            dataI[key].append(eval(out[0]))
        except:
            continue
        
for key in dataI.keys():
    I = np.array(dataI[key], dtype=int)
    J = np.array(dataJ[key], dtype=int)
    A = np.array(dataA[key])
    N = max(np.max(I), np.max(J)) + 1
    Mat = np.zeros((N, N))
    Mat[I, J] = A
    print(key, np.sum(Mat != 0))

    plt.figure()
    plt.spy(Mat)
    plt.title(key)
plt.show()
breakpoint()
        
        
