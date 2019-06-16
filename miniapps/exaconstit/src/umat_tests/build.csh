#! /bin/csh -f

# done with intel 18.0.2 compilers

# mpic++ -fPIC userumat.cxx -c -g -O0 -o userumat.o
# 
# # gfortran -fPIC umat_fcc_dislpar_size.f  -c -g -O0 -o umat.o -funderscoring -ffixed-line-length-none
# #
# # ifort -c -fPIC -auto -extend_source -w90 -w95 -WB -g -fpp -openmp
# # ifort -c -fPIC -auto -extend_source -WB -g -fpp umat_fcc_dislpar_size.f -o umat.o
# ifort -c -fPIC -auto -extend_source -fp-model strict -fp-model source -fpconstant -prec-div -prec-sqrt -no-ftz -g umat_fcc_dislpar_size.f -o umat.o 
# # -fp-model strict -fp-model source -fpconstant -prec-div -prec-sqrt -no-ftz   -g -O2   -module ../lib/fortran -fPIC  
# # ifort -c -fPIC -auto -extend_source -w90 -w95 -WB -fpp -O2 -openmp
# 
# 
# mpicc -fPIC -shared userumat.o umat.o -o userumat.so -lgfortran -lstdc++

mpic++ -fPIC userumat.cxx -c -O -o userumat.o
gfortran -fPIC umat_test.f  -c -g -O0 -o umat.o -funderscoring -ffixed-line-length-none
mpicc -fPIC -shared userumat.o umat.o -o userumat.so -lgfortran -lstdc++

#ifort -c -fPIC -auto -extend_source -fp-model strict -fp-model source -fpconstant -prec-div -prec-sqrt -no-ftz -O umat_fcc_dislpar_size.f -o umat.o 

#mpicc -fPIC -shared userumat.o umat.o -o userumat.so -lgfortran -lstdc++
