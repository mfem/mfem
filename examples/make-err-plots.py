
# Take err-N.dat files and create error plots for each cycle.
# Negative values are interpreted as refined regions and plotted in
# red.

import math
import matplotlib.pyplot as plt

for i in range(20):
    print(f"{i}")
    with open(f"err-{i}.dat") as f:
        lines = f.readlines()
        x = [float(line.split()[0]) for line in lines]
        y = [float(line.split()[1]) for line in lines]

        # errors in refined zones are marked as negative by convention
        # from ex18. in order to detect "negative zero", we use the
        # copysign method, which distinguishes negative zero from
        # zero.
        r = [(x,-y) for x,y in zip(x,y) if math.copysign(1, y) == -1.0]
        xr,yr = zip(*r)
        print(xr)
        print(yr)
        plt.scatter(x,y)
        plt.scatter(xr,yr,c='r')
        plt.title(f"Timestep {i}")
        plt.xlabel("reference el")
        plt.ylabel("L2 error")
        plt.ylim(0,0.05)
        #plt.show()
        plt.savefig(f"opt-err-{i}.png",dpi=60)
        plt.clf()
