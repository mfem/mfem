#!/usr/bin/env python

# This is the regression testing script for ex5-nguygen
# The reference cases are stored in the regress_test folder

import os
import sys
import subprocess

class bcolors:
	HEADER = '\033[95m'
	OKGREEN = '\033[92m'
	FAIL = '\033[91m'
	WARN = '\033[93m'
	RESET = '\033[0m'

tol = 1e-7

print('Running Regression Testing:')
path = 'regress_test/'
if len(sys.argv) > 1:
	filenames = sys.argv[1:]
else:
	filenames = os.listdir(path)
	filenames = [ path + name for name in filenames ]

failed = 0

for filename in filenames:
	# Parsing reference file

	print("----------------------------------------------------------------")
	print("Case: " + filename)

	if not os.path.isfile(filename):
		failed += 1
		print(bcolors.WARN + "NOT FOUND" + bcolors.RESET)
		continue

	def get_ref_option(file, option):
		ref_out = subprocess.getoutput("grep '^   "+option+"$' "+file)
		return len(ref_out) > 0

	dg = get_ref_option(filename, '--discontinuous')
	bcn = get_ref_option(filename, '--bc-neumann')
	rd = get_ref_option(filename, '--reduction')
	hb = get_ref_option(filename, '--hybridization')
	upwind = get_ref_option(filename, '--upwinded')
	nonlin = get_ref_option(filename, '--nonlinear')
	nonlin_conv = get_ref_option(filename, '--nonlinear-convection')
	nonlin_diff = get_ref_option(filename, '--nonlinear-diffusion')

	def get_ref_param(file, param, default=""):
		ref_out = subprocess.getoutput("grep '^   "+param+"' "+file+" | cut -d ' ' -f 5")
		if len(ref_out) > 0:
			return ref_out.split()[0]
		else:
			return default

	problem = int(get_ref_param(filename, '--problem'))
	order = int(get_ref_param(filename, '--order'))
	nx = int(get_ref_param(filename, '--ncells-x'))
	ny = int(get_ref_param(filename, '--ncells-y'))
	kappa = float(get_ref_param(filename, '--kappa', "1"))
	hdg = int(get_ref_param(filename, '--hdg_scheme', "1"))
	nls = int(get_ref_param(filename, '--nonlinear-solver', "1"))

	ref_out = subprocess.getoutput("grep '|| t_h - t_ex || / || t_ex || = ' "+filename+"  | cut -d '=' -f 2-")
	ref_L2_t = float(ref_out.split()[0])
	ref_out = subprocess.getoutput("grep '|| q_h - q_ex || / || q_ex || = ' "+filename+"  | cut -d '=' -f 2-")
	ref_L2_q = float(ref_out.split()[0])
	solver = "GMRES"
	if (nonlin or nonlin_diff) and hb:
		if nls == 1:
			solver = "LBFGS"
		elif nls == 2:
			solver = "LBB"
		elif nls == 3:
			solver = "Newton"
	ref_out = subprocess.getoutput("grep '"+solver+"+' "+filename+"  | cut -d '+' -f 2-")
	precond_ref = ref_out.split()[0]

	# Run test case
	command_line = "./ex5-nguyen -no-vis"
	command_line += " -nx " + str(nx)
	command_line += " -ny " + str(ny)
	command_line += " -p " + str(problem)
	command_line += " -o " + str(order)
	if dg:
		command_line += ' -dg'
	if bcn:
		command_line += ' -bcn'
	if rd:
		command_line += ' -rd'
	if hb:
		command_line += ' -hb'
	if upwind:
		command_line += ' -up'
	if nonlin:
		command_line += ' -nl'
	if nonlin_conv:
		command_line += ' -nlc'
	if nonlin_diff:
		command_line += ' -nld'
	if kappa != 1.:
		command_line += ' -k ' + str(kappa)
	if hdg != 1:
		command_line += ' -hdg ' + str(hdg)
	if nls != 1:
		command_line += ' -nls ' + str(nls)

	cmd_out = subprocess.getoutput(command_line)
	split_cmd_out = cmd_out.splitlines()
	indx_t = split_cmd_out[-1].find('= ')
	indx_q = split_cmd_out[-2].find('= ')
	precond_test_idx_s = split_cmd_out[-4].find('+')
	precond_test_idx_e = split_cmd_out[-4].find(' ')

	fail = False

	try:
		test_L2_t = float(split_cmd_out[-1][indx_t+2::])
		test_L2_q = float(split_cmd_out[-2][indx_q+2::])
		precond_test = split_cmd_out[-4][precond_test_idx_s+1:precond_test_idx_e]
	except:
		fail = True

	if not fail:
		if precond_test == precond_ref:
			if abs(ref_L2_t - test_L2_t) < tol and abs(ref_L2_q - test_L2_q) < tol:
				print(bcolors.OKGREEN + "SUCCESS: " + bcolors.RESET + command_line, flush=True)
			else:
				fail = True
		else:
			print(bcolors.HEADER + "SKIPPING: "+ bcolors.RESET + command_line + " → incompatible preconditioner")
	
	if fail:
		print(bcolors.FAIL + "FAIL: " + bcolors.RESET + command_line, flush=True)
		print(cmd_out)
		failed += 1

print("----------------------------------------------------------------")
if failed == 0:
	print(bcolors.OKGREEN + "SUCCESS: " + bcolors.RESET + "all tests finished succesfully!")
else:
	print(bcolors.FAIL + "FAIL: " + bcolors.RESET + str(failed) + " / " + str(len(filenames)) + " tests failed!")
