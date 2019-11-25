import subprocess
import os
import sys 


print ("Python Script Running\n")

args = []

args.append(' -o 2') 
args.append(' -k 1.59154943092') 
args.append(' -initref 2') 
args.append(' -ref 2') 
args.append(' -sol 3') 
run = 'mpirun'
nprocs = ' -np 6'
ex = ' FOSLS_maxwellp'

cmd = run + nprocs + ex

for i in args:
	cmd += i


print(cmd)
os.system(cmd) 
 
#end thats all