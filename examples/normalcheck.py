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


def run(fd, t, mesh="../data/square-disc-p3.mesh", diffusion=False):
    commandline = ["./ex-normal",
                   "--reltol", "1.e-10",
                   "--mesh", mesh]
    if diffusion:
        commandline.append("--diffusion")
        commandline = commandline + ["--boundary-attribute", "1"]
    if t[0] == "p":
        commandline = commandline + ["--penalty", t[1:]]
    elif t == "elimination":
        commandline = commandline + ["--elimination"]
    p = subprocess.Popen(commandline, stdout=fd, stderr=fd)
    p.communicate()


def main():
    with open("normalcheck.log", "w") as fd:
        for mesh in ["../data/square-disc-p3.mesh",
                     "icf.mesh",
                     "sphere_hex27.mesh"]:
            print(mesh)
            ds = [False]
            if mesh == "../data/square-disc-p3.mesh":
                ds = [False, True]
            for d in ds:
                print("  diffusion:", d)
                run(fd, "schur", mesh=mesh, diffusion=d)
                run(fd, "elimination", mesh=mesh, diffusion=d)
                print("    ", check_vectors())


def penalty(diffusion=False):
    print("Check penalty sanity, error should scale like 1/penalty:")
    with open("penalty.log", "w") as fd:
        run(fd, "schur", diffusion=diffusion)
        for penalty in [1.0, 1.e+2, 1.e+4, 1.e+6]:
            run(fd, "p" + str(penalty), diffusion=diffusion)
            print("  penalty parameter:", penalty, ", error:",
                  check_vectors(filea="penalty.vector"))


if __name__ == "__main__":
    main()
    penalty()
    penalty(True)
