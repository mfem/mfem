# Update the LD_LIBRARY_PATH of the C++14 kernel so it can find mfem without
# extra pragma cling statements

import json

kernelspec = "/srv/conda/envs/notebook/share/jupyter/kernels/xcpp14/kernel.json"

with open(kernelspec, "r") as f:
    obj = json.load(f)

obj["env"] = {"LD_LIBRARY_PATH": "/srv/conda/envs/notebook/lib"}

with open(kernelspec, "w") as f:
    json.dump(obj, f)
