# CMake generated Testfile for 
# Source directory: /g/g19/bs/quartz/mfem/miniapps/electromagnetics
# Build directory: /g/g19/bs/quartz/mfem/bens_build/miniapps/electromagnetics
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
ADD_TEST(tesla_np=4 "/usr/bin/srun" "-np" "4" "/g/g19/bs/quartz/mfem/bens_build/miniapps/electromagnetics/tesla" "-no-vis" "-maxit" "2" "-cr" "0 0 -0.2 0 0 0.2 0.2 0.4 1")
ADD_TEST(volta_np=4 "/usr/bin/srun" "-np" "4" "/g/g19/bs/quartz/mfem/bens_build/miniapps/electromagnetics/volta" "-no-vis" "-maxit" "2" "-dbcs" "1" "-dbcg" "-ds" "0.0 0.0 0.0 0.2 8.0")
ADD_TEST(joule_np=4 "/usr/bin/srun" "-np" "4" "/g/g19/bs/quartz/mfem/bens_build/miniapps/electromagnetics/joule" "-no-vis" "-p" "rod" "-tf" "3" "-m" "/g/g19/bs/quartz/mfem/miniapps/electromagnetics/cylinder-hex.mesh")
ADD_TEST(maxwell_np=4 "/usr/bin/srun" "-np" "4" "/g/g19/bs/quartz/mfem/bens_build/miniapps/electromagnetics/maxwell" "-no-vis" "-abcs" "-1" "-dp" "-0.3 0.0 0.0 0.3 0.0 0.0 0.1 1 .5 .5")
