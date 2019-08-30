# CMake generated Testfile for 
# Source directory: /g/g19/bs/quartz/mfem/miniapps/performance
# Build directory: /g/g19/bs/quartz/mfem/bens_build/miniapps/performance
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
ADD_TEST(performance_ex1_ser "/g/g19/bs/quartz/mfem/bens_build/miniapps/performance/performance_ex1" "-no-vis" "-r" "2")
ADD_TEST(performance_ex1p_np=4 "/usr/bin/srun" "-np" "4" "/g/g19/bs/quartz/mfem/bens_build/miniapps/performance/performance_ex1p" "-no-vis" "-rs" "2")
