"""
Bazel BUILD helper functions
"""

load("@rules_cc//cc:defs.bzl", "cc_binary")

def generate_examples():
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
