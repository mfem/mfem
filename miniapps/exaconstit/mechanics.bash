### LSF syntax
#BSUB -nnodes 1                   #number of nodes
#BSUB -W 120                      #walltime in minutes
#BSUB -G guests                   #account
#BSUB -e myerrors.txt             #stderr
#BSUB -o myoutput.txt             #stdout
#BSUB -J myjob                    #name of job
#BSUB -q pdebug                   #queue to use

jsrun -n8 ./mechanics_driver -hexmesh -mx 1.0 -nx 10 -tf 25.0 -dt 0.1 -vardt -nsteps 35 -cust-dt 'custom_dt_coarse.txt' -attrid '1 2 3 4' -disp '0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.001' -bcid '3 1 2 3' -rs 0 --visit -nsvars 43 -svars 'state_cp_becker.txt' -nprops 88 -props 'props_cp_becker.txt' -umat -cp -ng 500 -gmap './CA_results/grainIDs_u.in' -g './CA_results/fe.in' -gc -gcstride 9 -rel 5e-6 -abs 1e-8
