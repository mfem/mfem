"""
Bazel configuration helper functions
"""

load("@rules_cc//cc:defs.bzl", "cc_binary")

### Examples ##################################################################
def mfem_serial_examples():
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

### MFEM_USE_* ################################################################
def _mfem_use(ctx):
    value = "//" if not ctx.attr.use else ""
    value += "#define " + ctx.attr.define
    return [
        platform_common.TemplateVariableInfo({ctx.attr.define: value}),
    ]

mfem_use = rule(
    implementation = _mfem_use,
    attrs = {
        "define": attr.string(),
        "use": attr.bool(),
    },
)
