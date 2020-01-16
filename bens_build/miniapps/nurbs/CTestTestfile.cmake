# CMake generated Testfile for 
# Source directory: /Users/ben/Documents/SoftwareLibraries/mfem/miniapps/nurbs
# Build directory: /Users/ben/Documents/SoftwareLibraries/mfem/bens_build/miniapps/nurbs
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(nurbs_ex1_ser "/Users/ben/Documents/SoftwareLibraries/mfem/bens_build/miniapps/nurbs/nurbs_ex1" "-no-vis")
add_test(nurbs_ex1_per_ser "/Users/ben/Documents/SoftwareLibraries/mfem/bens_build/miniapps/nurbs/nurbs_ex1" "-no-vis" "-m" "../../data/beam-hex-nurbs.mesh" "-pm" "1" "-ps" "2")
add_test(nurbs_ex1p_np=4 "/usr/local/bin/mpiexec" "-np" "4" "/Users/ben/Documents/SoftwareLibraries/mfem/bens_build/miniapps/nurbs/nurbs_ex1p" "-no-vis")
add_test(nurbs_ex11p_np=4 "/usr/local/bin/mpiexec" "-np" "4" "/Users/ben/Documents/SoftwareLibraries/mfem/bens_build/miniapps/nurbs/nurbs_ex11p" "-no-vis")
