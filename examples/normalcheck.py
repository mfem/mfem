import numpy as np
import subprocess


def read_vector(filename):
    v = []
    with open(filename, "r") as fd:
        for line in fd:
            v.append(float(line))
    return np.array(v)


def check_vectors(filea="elimination.vector",
                  fileb="schur.vector"):
    veca = read_vector(filea)
    vecb = read_vector(fileb)
    return np.linalg.norm(veca - vecb) / np.linalg.norm(vecb)


def main():
    with open("normalcheck.log", "w") as fd:
        for mesh in ["../data/square-disc-p3.mesh",
                     "icf.mesh",
                     "sphere_hex27.mesh"]:
            print(mesh)
            p = subprocess.Popen(["./ex-normal",
                                  "--reltol", "1.e-10",
                                  "--mesh", mesh],
                                 stdout=fd, stderr=fd)
            p.communicate()
            p = subprocess.Popen(["./ex-normal",
                                  "--reltol", "1.e-10",
                                  "--elimination",
                                  "--mesh", mesh],
                                 stdout=fd, stderr=fd)
            p.communicate()
            print("  ", check_vectors())


def penalty():
    print("Check penalty sanity, error should scale like 1/penalty:")
    with open("penalty.log", "w") as fd:
        p = subprocess.Popen(["./ex-normal",
                              "--reltol", "1.e-10"],
                             stdout=fd, stderr=fd)
        p.communicate()
        for penalty in [1.0, 1.e+2, 1.e+4, 1.e+6]:
            p = subprocess.Popen(["./ex-normal",
                                  "--reltol", "1.e-10",
                                  "--penalty", str(penalty)],
                                 stdout=fd, stderr=fd)
            p.communicate()
            print("  penalty parameter:", penalty, ", error:",
                  check_vectors(filea="penalty.vector"))


if __name__ == "__main__":
    main()
    penalty()
