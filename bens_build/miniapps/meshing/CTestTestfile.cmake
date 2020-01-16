# CMake generated Testfile for 
# Source directory: /Users/ben/Documents/SoftwareLibraries/mfem/miniapps/meshing
# Build directory: /Users/ben/Documents/SoftwareLibraries/mfem/bens_build/miniapps/meshing
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(mesh-optimizer "/Users/ben/Documents/SoftwareLibraries/mfem/bens_build/miniapps/meshing/mesh-optimizer" "-no-vis" "-m" "/Users/ben/Documents/SoftwareLibraries/mfem/miniapps/meshing/icf.mesh")
add_test(pmesh-optimizer_np=4 "/usr/local/bin/mpiexec" "-np" "4" "/Users/ben/Documents/SoftwareLibraries/mfem/bens_build/miniapps/meshing/pmesh-optimizer" "-no-vis" "-m" "/Users/ben/Documents/SoftwareLibraries/mfem/miniapps/meshing/icf.mesh")
