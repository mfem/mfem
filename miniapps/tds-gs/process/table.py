import numpy as np


# two tables: pc_option 0, pc_option 5
# rows: iterations
# columns: v, w

print("")
for pc in [0, 5]:
    print("pc=%d" % (pc))
    for it in [1, 5, 10]:
        print("%d & " % (it), end="")
        for cyc in [0, 1]:
            name = "../out_iter/iters_model2_pc%d_cyc%d_it%d.txt" % (pc, cyc, it)
            fid = open(name, 'r')
            data = []
            for line in fid:
                if "amr" in line:
                    out = line[:-1].split(" ")
                    out = [eval(a.split("=")[-1]) for a in out]
                    data.append(out)
            data = np.array(data, dtype=int)
            print("%.1f (%d, %d)" % (np.average(data[:, 2]), np.min(data[:, 2]), np.max(data[:, 2])), end="")
            if cyc == 0:
                print(" & ", end="")
            else:
                print(" \\\\")

    print("")
        

            

