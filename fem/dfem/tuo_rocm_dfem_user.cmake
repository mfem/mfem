if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Build type: Debug, Release, RelWithDebInfo, or MinSizeRel." FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_FLAGS "--save-temps -Rpass-analysis=kernel-resource-usage -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false")

set(MFEM_PRECISION "double" CACHE STRING
    "Floating-point precision to use: single, or double")

option(BUILD_SHARED_LIBS "Enable shared library build of MFEM" ON)
option(MFEM_USE_MPI "Enable MPI parallel build" ON)
option(MFEM_USE_METIS "Enable METIS usage" ${MFEM_USE_MPI})
option(MFEM_USE_ENZYME "Enable Enzyme" ON)
option(MFEM_USE_HIP "Enable HIP" ON)

set(MFEM_MPI_NP 4 CACHE STRING "Number of processes used for MPI tests")

option(MFEM_ENABLE_TESTING ON)

set(HIP_ARCH "gfx942" CACHE STRING "Target HIP architecture.")

# Make sure all dirs are absolute
set(ENZYME_DIR "/usr/workspace/andrej1/dfem-tuo-magic/local/cmake/Enzyme" CACHE PATH "Path to the Enzyme library.")
set(HYPRE_DIR "/usr/workspace/andrej1/dfem-tuo-magic/local" CACHE PATH "Path to the hypre library.")
set(METIS_DIR "/usr/workspace/andrej1/dfem-tuo-magic/local" CACHE PATH "Path to the METIS library.")

set(CMAKE_SKIP_PREPROCESSED_SOURCE_RULES ON) # Skip *.i rules
set(CMAKE_SKIP_ASSEMBLY_SOURCE_RULES ON) # Skip *.s rules
