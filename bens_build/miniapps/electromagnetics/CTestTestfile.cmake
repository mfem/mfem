# CMake generated Testfile for 
# Source directory: /Users/ben/Documents/SoftwareLibraries/mfem/miniapps/electromagnetics
# Build directory: /Users/ben/Documents/SoftwareLibraries/mfem/bens_build/miniapps/electromagnetics
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(tesla_np=4 "/usr/local/bin/mpiexec" "-np" "4" "/Users/ben/Documents/SoftwareLibraries/mfem/bens_build/miniapps/electromagnetics/tesla" "-no-vis" "-maxit" "2" "-cr" "0 0 -0.2 0 0 0.2 0.2 0.4 1")
add_test(volta_np=4 "/usr/local/bin/mpiexec" "-np" "4" "/Users/ben/Documents/SoftwareLibraries/mfem/bens_build/miniapps/electromagnetics/volta" "-no-vis" "-maxit" "2" "-dbcs" "1" "-dbcg" "-ds" "0.0 0.0 0.0 0.2 8.0")
add_test(joule_np=4 "/usr/local/bin/mpiexec" "-np" "4" "/Users/ben/Documents/SoftwareLibraries/mfem/bens_build/miniapps/electromagnetics/joule" "-no-vis" "-p" "rod" "-tf" "3" "-m" "/Users/ben/Documents/SoftwareLibraries/mfem/miniapps/electromagnetics/cylinder-hex.mesh")
add_test(maxwell_np=4 "/usr/local/bin/mpiexec" "-np" "4" "/Users/ben/Documents/SoftwareLibraries/mfem/bens_build/miniapps/electromagnetics/maxwell" "-no-vis" "-abcs" "-1" "-dp" "-0.3 0.0 0.0 0.3 0.0 0.0 0.1 1 .5 .5")
