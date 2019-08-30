# CMake generated Testfile for 
# Source directory: /g/g19/bs/quartz/mfem/miniapps/meshing
# Build directory: /g/g19/bs/quartz/mfem/bens_build/miniapps/meshing
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
ADD_TEST(mesh-optimizer "/g/g19/bs/quartz/mfem/bens_build/miniapps/meshing/mesh-optimizer" "-no-vis" "-m" "/g/g19/bs/quartz/mfem/miniapps/meshing/icf.mesh")
ADD_TEST(pmesh-optimizer_np=4 "/usr/bin/srun" "-np" "4" "/g/g19/bs/quartz/mfem/bens_build/miniapps/meshing/pmesh-optimizer" "-no-vis" "-m" "/g/g19/bs/quartz/mfem/miniapps/meshing/icf.mesh")
