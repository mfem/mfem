# CMake generated Testfile for 
# Source directory: /Users/ben/Documents/SoftwareLibraries/mfem/miniapps/performance
# Build directory: /Users/ben/Documents/SoftwareLibraries/mfem/bens_build/miniapps/performance
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(performance_ex1_ser "/Users/ben/Documents/SoftwareLibraries/mfem/bens_build/miniapps/performance/performance_ex1" "-no-vis" "-r" "2")
add_test(performance_ex1p_np=4 "/usr/local/bin/mpiexec" "-np" "4" "/Users/ben/Documents/SoftwareLibraries/mfem/bens_build/miniapps/performance/performance_ex1p" "-no-vis" "-rs" "2")
