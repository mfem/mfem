# This is the regression testing script for ex5-nguygen
# The reference cases are stored in the regress_test folder
# Currently, this uses cases 1 and 2
# Naming convention of the saved output reference files:
# 1st digit -> Problem number: 1 or 2 (for now)
# 2nd digit -> Polynomial order
# 3rd digit -> DG: 0 == False and 1 == True
# 4th digit -> Hybridization: 0 == False and 1 == True

import os
import subprocess

class bcolors:
	HEADER = '\033[95m'
	OKGREEN = '\033[92m'
	FAIL = '\033[91m'
	RESET = "\033[0m"

tol = 1e-7

print('Running Regression Testing:')
path = 'regress_test/'
filenames = os.listdir(path)

for i in range(len(filenames)):
	# Parcing reference file
	problem = filenames[i][7]
	order = filenames[i][8]
	dg = int(filenames[i][9])
	hb = int(filenames[i][10])

	dg_com = ''
	if int(dg) == 1:
		dg_com = " -dg "
	hb_com = ''
	if int(hb) == 1:
		hb_com = "-hb "

	p = subprocess.getoutput("grep ' --ncells-x' "+path+filenames[i]+"| cut -d ' ' -f 5")
	nx = p.split()[0]
	p = subprocess.getoutput("grep ' --ncells-y' "+path+filenames[i]+"| cut -d ' ' -f 5")
	ny = p.split()[0]

	p = subprocess.getoutput("grep '|| t_h - t_ex || / || t_ex || = ' "+path+filenames[i]+"  | cut -d '=' -f 2-")
	ref_L2_t = float(p.split()[0])
	p = subprocess.getoutput("grep '|| q_h - q_ex || / || q_ex || = ' "+path+filenames[i]+"  | cut -d '=' -f 2-")
	ref_L2_q = float(p.split()[0])
	p = subprocess.getoutput("grep 'GMRES+' "+path+filenames[i]+"  | cut -d '+' -f 2-")
	precond_ref = p.split()[0]

	# Run test case
	print("----------------------------------------------------------------")
	print("Case: "+filenames[i])
	command_line = "./ex5-nguyen -no-vis -nx "+str(nx)+" -ny "+str(ny)+" -p "+problem+" -o "+order+dg_com+hb_com

	cmd_out = subprocess.getoutput(command_line)
	new_cmd_out = cmd_out.splitlines()
	indx_t = new_cmd_out[-1].find('= ')
	indx_q = new_cmd_out[-2].find('= ')
	precond_test_idx_s = new_cmd_out[-4].find('+')
	precond_test_idx_e = new_cmd_out[-4].find(' ')

	test_L2_t = float(new_cmd_out[-1][indx_t+2::])
	test_L2_q = float(new_cmd_out[-2][indx_q+2::])
	precond_test = new_cmd_out[-4][precond_test_idx_s+1:precond_test_idx_e]

	if precond_test == precond_ref:
		if abs(ref_L2_t - test_L2_t) < tol and abs(ref_L2_q - test_L2_q) < tol:
			print(bcolors.OKGREEN + "SUCCESS: " + bcolors.RESET, end="", flush=True)
			print(command_line)
		else:
			print(bcolors.FAIL + "FAIL: " + bcolors.RESET, end="", flush=True)
			print(command_line)
			print(cmd_out)
	else:
		print(bcolors.HEADER + "SKIPPING: "+ bcolors.RESET +command_line+" → incompatible preconditioner")

