import numpy as np


# two tables: pc_option 0, pc_option 5
# rows: iterations
# columns: v, w

def table(model, opt=1):
    print("model %d" % (model))
    print("")
    for pc in [0, 5, 6]:
        print("pc=%d" % (pc))
        # if pc == 7 or pc == 0:
        #     its = [1, 3, 5]
        #     cycs = [1]
        # else:
        #     its = [1, 5, 10]
        #     cycs = [0]
        cycs = [0, 1]
        its = [1, 3, 5, 10]
        for it in its:
            print("%d & " % (it), end="")
            for cyc in cycs:
                num_first_newton = 0.0
                num_subsequent_newton = 0.0
                num_amr = 0.0
                name = "../out_iter/iters_model%d_pc%d_cyc%d_it%d.txt" % (model, pc, cyc, it)
                fid = open(name, 'r')
                data = []
                for line in fid:
                    if "amr" in line:
                        out = line[:-1].split(" ")
                        out = [eval(a.split("=")[-1]) for a in out]
                        data.append(out)
                        if out[1] == 0:
                            num_amr += 1.0
                        if num_amr == 1:
                            num_first_newton += 1.0
                        else:
                            num_subsequent_newton += 1.0
                    if "nonl_res" in line:
                        key1 = "nonl_res: "
                        ind1 = line.index(key1)
                        key2 = ", ratio"
                        ind2 = line.index(key2)

                        out_res = eval(line[ind1+len(key1):ind2])


                data = np.array(data, dtype=int)
                if out_res < 1e-6:
                    if opt == 1:
                        print("%.1f (%d, %d)" % (np.average(data[:, 2]), np.min(data[:, 2]), np.max(data[:, 2])), end="")
                    else:
                        if num_amr > 1:
                            print("(%d, %.1f) %.1f" % (num_first_newton, num_subsequent_newton / (num_amr - 1), np.average(data[:, 2])), end="")
                        else:
                            print("(%d) %.1f" % (num_first_newton, np.average(data[:, 2])), end="")
                else:
                    print("%.1e" % (out_res), end="")
                if cyc != cycs[-1]:
                    print(" & ", end="")
                else:
                    print(" \\\\")

        print("")

table(1, opt=2)
table(2, opt=2)
table(4, opt=2)
# table(3, opt=2)


