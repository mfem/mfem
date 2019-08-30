# CMake generated Testfile for 
# Source directory: /g/g19/bs/quartz/mfem/miniapps/nurbs
# Build directory: /g/g19/bs/quartz/mfem/bens_build/miniapps/nurbs
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
ADD_TEST(nurbs_ex1_ser "/g/g19/bs/quartz/mfem/bens_build/miniapps/nurbs/nurbs_ex1" "-no-vis")
ADD_TEST(nurbs_ex1_per_ser "/g/g19/bs/quartz/mfem/bens_build/miniapps/nurbs/nurbs_ex1" "-no-vis" "-m" "../../data/beam-hex-nurbs.mesh" "-pm" "1" "-ps" "2")
ADD_TEST(nurbs_ex1p_np=4 "/usr/bin/srun" "-np" "4" "/g/g19/bs/quartz/mfem/bens_build/miniapps/nurbs/nurbs_ex1p" "-no-vis")
ADD_TEST(nurbs_ex11p_np=4 "/usr/bin/srun" "-np" "4" "/g/g19/bs/quartz/mfem/bens_build/miniapps/nurbs/nurbs_ex11p" "-no-vis")
