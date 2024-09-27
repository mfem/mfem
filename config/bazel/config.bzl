"""
Bazel configuration helper functions
"""

load("@rules_cc//cc:defs.bzl", "cc_binary")

### Examples ##################################################################
def mfem_examples():
    for n in range(1, 40):
        example = "ex" + str(n)
        cc_binary(
            name = example,
            srcs = ["examples/" + example + ".cpp"],
            deps = [
                "mfem",
                "mfem_hpp",
                "examples_hpp",
            ],
        )

def mfem_parallel_examples():
    for n in range(1, 40):
        example = "ex" + str(n) + "p"
        cc_binary(
            name = example,
            srcs = ["examples/" + example + ".cpp"],
            deps = [
                "mfem",
                "mfem_hpp",
                "examples_hpp",
                "@mpi",
            ],
        )
