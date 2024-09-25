# load("@bazel//tools/build_defs/repo:http.bzl", "http_archive")
# load("@bazel//tools/build_defs/repo:local.bzl", "new_local_repository")

### Load rules ################################################################
load("@rules_cc//cc:defs.bzl", "cc_library")

### MFEM examples #############################################################
load("//:.bazel.bzl", "generate_examples", "get_bash_bin")

generate_examples()

### MPI #######################################################################
# add_mpi_local_repository()

# new_local_repository(
#     # repo_mapping, # _not_ supported in `MODULE.bazel`
#     name = "MPI",
#     # build_file = "bazel/mpi",
#     # build_file_content = "",
#     path = "/opt/homebrew/opt/openmpi",
# )

# mpicc [-showme[:<command,compile,link,incdirs,libdirs,libs,version,help>]] args
#   -showme:command    Show command used to invoke real compiler
#   -showme:compile    Show flags added when compiling
#   -showme:link       Show flags added when linking
#   -showme:incdirs    Show list of include dirs added when compiling
#   -showme:libdirs    Show list of library dirs added when linking
#   -showme:libs       Show list of libraries added when linking
#   -showme:version    Show version of Open MPI
#   -showme:help       This help message
genrule(
    name = "mpi",
    srcs = [],
    outs = ["mpi_path"],
    cmd_bash = "which mpicxx > $@ ",
    message = "Running which_mpicxx",
)
# load("path.bzl", "MPI_PATH")
# which_mpicxx()
# my_rule()
# mpi_path(name = "mpi")

get_bash_bin(name = "bash")

### MFEM library ##############################################################

cc_library(
    name = "mfem",
    deps = [
        "bash",
        "fem",
        "general",
        "linalg",
        "mesh",
    ],
)

### Sources ###################################################################

cc_library(
    name = "fem",
    srcs = glob([
        "fem/*.cpp",
        "fem/ceed/**/*.cpp",
        "fem/fe/*.cpp",
        "fem/integ/*.cpp",
        "fem/lor/*.cpp",
        # skip moonolith
        "fem/qinterp/*.cpp",
        "fem/tmop/*.cpp",
    ]),
    deps = [
        "config_hpp",
        "fem_hpp",
        "general_hpp",
        "linalg_hpp",
        "mesh_hpp",
    ],
)

cc_library(
    name = "general",
    srcs = glob(["general/*.cpp"]),
    deps = [
        "config_hpp",
        "general_hpp",
        "linalg_hpp",
    ],
)

cc_library(
    name = "linalg",
    srcs = glob(["linalg/**/*.cpp"]),
    deps = [
        "config_hpp",
        "fem_hpp",
        "general_hpp",
        "linalg_hpp",
        "mesh_hpp",
    ],
)

cc_library(
    name = "mesh",
    srcs = glob([
        "mesh/*.cpp",
        "mesh/submesh/*.cpp",
    ]),
    deps = [
        "config_hpp",
        "fem_hpp",
        "general_hpp",
        "linalg_hpp",
        "mesh_hpp",
    ],
)

### Headers ###################################################################

cc_library(
    name = "examples_hpp",
    srcs = glob(["examples/*.hpp"]),
)

cc_library(
    name = "general_hpp",
    srcs = glob(
        [
            "general/*.hpp",
            "general/*.h",
        ],
    ),
)

cc_library(
    name = "mfem_hpp",
    srcs = ["mfem.hpp"],
    deps = ["config_hpp"],
)

cc_library(
    name = "config_hpp",
    srcs = glob(["config/*.hpp"]),
)

cc_library(
    name = "fem_hpp",
    srcs = glob([
        "fem/*.hpp",
        "fem/*.h",
    ]),
    deps = [
        "fem_ceed_hpp",
        "fem_fe_hpp",
        "fem_integ_hpp",
        "fem_lor_hpp",
        # "fem_moonolith_hpp",
        "fem_qinterp_hpp",
        "fem_tmop_hpp",
    ],
)

cc_library(
    name = "fem_ceed_hpp",
    srcs = glob(["fem/ceed/**/*.hpp"]),
)

cc_library(
    name = "fem_moonolith_hpp",
    srcs = glob(["fem/moonolith/**/*.hpp"]),
)

cc_library(
    name = "fem_fe_hpp",
    srcs = glob(["fem/fe/*.hpp"]),
)

cc_library(
    name = "fem_integ_hpp",
    srcs = glob(["fem/integ/*.hpp"]),
)

cc_library(
    name = "fem_lor_hpp",
    srcs = glob(["fem/lor/*.hpp"]),
)

cc_library(
    name = "fem_qinterp_hpp",
    srcs = glob(["fem/qinterp/*.hpp"]),
)

cc_library(
    name = "fem_tmop_hpp",
    srcs = glob(["fem/tmop/*.hpp"]),
)

cc_library(
    name = "linalg_hpp",
    srcs = glob(["linalg/*.hpp"]),
    deps = [
        "linalg_batched_hpp",
        "linalg_simd_hpp",
    ],
)

cc_library(
    name = "linalg_batched_hpp",
    srcs = glob(["linalg/batched/*.hpp"]),
)

cc_library(
    name = "linalg_simd_hpp",
    srcs = glob(["linalg/simd/*.hpp"]),
)

cc_library(
    name = "mesh_hpp",
    srcs = glob(["mesh/*.hpp"]),
    deps = ["submesh_hpp"],
)

cc_library(
    name = "submesh_hpp",
    srcs = glob(["mesh/submesh/*.hpp"]),
)
