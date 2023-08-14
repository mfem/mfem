# OneProcessAMGContact



Be sure to edit the makefile so that it points to a parallel MFEM build

specifically the MFEM_BUILD_DIR


after building exQPContactBlockTL one can

1. run the bash script scalingJobArray.bat via `source scalingJobArray.bat' which will populate the CG iterations required to solve
      various linear systems into the data/ subdirectory
2. run the python script data/process.py in order to put the scaling information into the single files algorithmicScaling_Elasticity.dat and algorithmicScaling_noElasticity.dat
      in order to see the number of average AMG-CG iterations per optimization solve. 


