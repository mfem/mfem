import numpy as np
from numpy import linalg as LA

cpu = np.loadtxt("cpu_u.txt");
gpu = np.loadtxt("gpu_u.txt");

err = LA.norm((cpu-gpu));

print ("difference is ");
print err
