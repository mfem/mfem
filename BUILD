### String Flags ##############################################################
load("@bazel_skylib//rules:common_settings.bzl", "string_flag")

### Load rules ################################################################
load("@rules_cc//cc:defs.bzl", "cc_library")

### Load config/bazel/config ##################################################
load("//config/bazel:config.bzl", "mfem_serial_examples",
                                  "mfem_parallel_examples", "mfem_use")

### Load config/bazel/settings ################################################
### https://bazel.build/docs/configurable-attributes
load("//config/bazel:settings.bzl", "mode", "precision", "print_mode", "print_precision")

# FLAG: Serial/Parallel MODE ##################################################
string_flag(
    name = "mode",
    values = ["serial", "parallel"],
    build_setting_default = "serial",
)

mode(name = "serial")

mode(name = "parallel")

config_setting(
    name = "serial_mode",
    flag_values = {":mode": "serial"},
)

config_setting(
    name = "parallel_mode",
    flag_values = {":mode": "parallel"},
)

print_mode(
    name = "print_mode",
    mode = select({
        ":serial_mode": "serial",
        ":parallel_mode": "parallel",
    }),
)

# FLAG: Double/Single PRECISION ###############################################
string_flag(
    name = "precision",
    values = ["single", "double"],
    build_setting_default = "double",
)

precision(name = "double")

precision(name = "single")

config_setting(
    name = "single_precision",
    flag_values = {":precision": "single"},
)

config_setting(
    name = "double_precision",
    flag_values = {":precision": "double"},
)

print_precision(
    name = "print_precision",
    precision = select({
        ":single_precision": "single",
        ":double_precision": "double",
    }),
)

# MFEM_USE_* definitions ######################################################
mfem_use(
    name = "mfem_not_mpi",
    define = "MFEM_USE_MPI",
    use = False,
)

mfem_use(
    name = "mfem_use_mpi",
    define = "MFEM_USE_MPI",
    use = True,
)

mfem_use(
    name = "mfem_not_metis",
    define = "MFEM_USE_METIS",
    use = False,
)

mfem_use(
    name = "mfem_use_metis",
    define = "MFEM_USE_METIS",
    use = True,
)

mfem_use(
    name = "mfem_not_metis_5",
    define = "MFEM_USE_METIS_5",
    use = False,
)

mfem_use(
    name = "mfem_use_metis_5",
    define = "MFEM_USE_METIS_5",
    use = True,
)

mfem_use(
    name = "no_mfem_hypre_version",
    define = "MFEM_HYPRE_VERSION",
    use = False,
)

mfem_use(
    name = "mfem_hypre_version",
    define = "MFEM_HYPRE_VERSION",
    use = True,
)

mfem_use(
    name = "mfem_not_double",
    define = "MFEM_USE_DOUBLE",
    use = False,
)

mfem_use(
    name = "mfem_use_double",
    define = "MFEM_USE_DOUBLE",
    use = True,
)

mfem_use(
    name = "mfem_not_single",
    define = "MFEM_USE_SINGLE",
    use = False,
)

mfem_use(
    name = "mfem_use_single",
    define = "MFEM_USE_SINGLE",
    use = True,
)

### https://bazel.build/reference/be/general#genrule
genrule(
    name = "genrule_config_bazel",
    srcs = ["BUILD"],
    outs = ["config/bazel.hpp"],
    cmd = """cat <<EOF > $@
// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_CONFIG_HEADER
#define MFEM_CONFIG_HEADER

// MFEM version: integer of the form: (major*100 + minor)*100 + patch.
#define MFEM_VERSION 40701

// MFEM version string of the form "3.3" or "3.3.1".
#define MFEM_VERSION_STRING "4.7.1"

// MFEM version type, see the MFEM_VERSION_TYPE_* constants below.
#define MFEM_VERSION_TYPE ((MFEM_VERSION) % 2)

// MFEM version type constants.
#define MFEM_VERSION_TYPE_RELEASE 0
#define MFEM_VERSION_TYPE_DEVELOPMENT 1

// Separate MFEM version numbers for major, minor, and patch.
#define MFEM_VERSION_MAJOR ((MFEM_VERSION) / 10000)
#define MFEM_VERSION_MINOR (((MFEM_VERSION) / 100) % 100)
#define MFEM_VERSION_PATCH ((MFEM_VERSION) % 100)

// The absolute path of the MFEM source prefix.
#define MFEM_SOURCE_DIR "$$(realpath $$(realpath BUILD)/..)"

// The absolute path of the MFEM installation prefix.
#define MFEM_INSTALL_DIR "$$(realpath $(BINDIR))"

// Description of the git commit used to build MFEM.
#define MFEM_GIT_STRING "heads/bazel-git-..."

// Build the parallel MFEM library.
// Requires an MPI compiler, and the libraries HYPRE and METIS.
$(MFEM_USE_MPI)

// Enable MFEM features that use the METIS library (parallel MFEM).
$(MFEM_USE_METIS)

// Enable this option if linking with METIS version 5 (parallel MFEM).
$(MFEM_USE_METIS_5)

// Version of HYPRE used for building MFEM.
// macOS: 23200, ubuntu: 21821
$(MFEM_HYPRE_VERSION) 23200

// Use single/double-precision floating point type
$(MFEM_USE_DOUBLE)
$(MFEM_USE_SINGLE)

// Internal MFEM option: enable group/batch allocation for some small objects.
#define MFEM_USE_MEMALLOC

// Which library functions to use in class StopWatch for measuring time.
// For a list of the available options, see INSTALL.
// If not defined, an option is selected automatically.
// 0/1/2/3/4/5/6/NO
#define MFEM_TIMER_TYPE 0

#endif // MFEM_CONFIG_HEADER
EOF""",
    local = False,
    message = "Generating config bazel.hpp file",
    toolchains = select({
        ":serial_mode": [
            ":mfem_not_mpi",
            ":mfem_not_metis",
            ":mfem_not_metis_5",
            ":no_mfem_hypre_version",
        ],
        ":parallel_mode": [
            ":mfem_use_mpi",
            ":mfem_use_metis",
            ":mfem_use_metis_5",
            ":mfem_hypre_version",
        ],
        "//conditions:default": [
            ":mfem_not_mpi",
            ":mfem_not_metis",
            ":mfem_not_metis_5",
            ":mfem_hypre_version",
            ":mfem_use_double",
            ":mfem_not_single",
        ],
    }) + select({
        ":double_precision": [
            ":mfem_use_double",
            ":mfem_not_single",
        ],
        ":single_precision": [
            ":mfem_not_double",
            ":mfem_use_single",
        ],
    }),
)

cc_library(
    name = "config_bazel_hpp",
    srcs = ["config/bazel.hpp"],
    includes = ["config"],
)

### MFEM Examples #############################################################
mfem_serial_examples()
mfem_parallel_examples()

### MFEM library ##############################################################

cc_library(
    name = "mfem",
    deps = [
        "fem",
        "general",
        "linalg",
        "mesh",
        "@config",
    ] + select({
        ":parallel_mode": ["@mpi"],
        "//conditions:default": [],
    }),
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
    ] + select({
        ":parallel_mode": ["@mpi"],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "general",
    srcs = glob(["general/*.cpp"]),
    deps = [
        "config_hpp",
        "general_hpp",
        "linalg_hpp",
    ] + select({
        ":parallel_mode": ["@mpi"],
        "//conditions:default": [],
    }),
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
    ] + select({
        ":parallel_mode": ["@mpi"],
        "//conditions:default": [],
    }),
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
    ] + select({
        ":parallel_mode": ["@mpi"],
        "//conditions:default": [],
    }),
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
    deps = [
        ":config_bazel_hpp",
        "@config",
    ],
)

cc_library(
    name = "fem_hpp",
    srcs = glob([
        "fem/*.hpp",
        "fem/*.h",
    ]),
    deps = [
        "config_hpp",
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
