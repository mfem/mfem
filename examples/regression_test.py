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

tol = 1e-4
def equal(a, b):
	return abs(a - b) / (abs(a) + abs(b)) < tol

print('Running Regression Testing:')
parallel = False
filenames = []
if len(sys.argv) > 1:
	if sys.argv[1] == '-par':
		parallel = True
		if len(sys.argv) > 2:
			filenames = sys.argv[2:]
	else:
		filenames = sys.argv[1:]

if parallel:
	path = 'regress_test_par/'
else:
	path = 'regress_test/'

if len(filenames) == 0:
	filenames = os.listdir(path)
	filenames = [ path + name for name in filenames ]

failed = 0
skipped = 0

for i, filename in enumerate(filenames):
	# Parsing reference file

	print("----------------------------------------------------------------")
	print(f"Case {i+1}/{len(filenames)}: {filename}")

	if not os.path.isfile(filename):
		failed += 1
		print(f"{bcolors.WARN}NOT FOUND{bcolors.RESET}")
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
	nls = int(get_ref_param(filename, '--nonlinear-solver', "0"))

	file = open(filename, "r")
	ref_out = file.readlines()
	ref_L2_t_idx = ref_out[-1].find('= ')
	ref_L2_q_idx = ref_out[-2].find('= ')
	ref_L2_t = float(ref_out[-1][ref_L2_t_idx+2::])
	ref_L2_q = float(ref_out[-2][ref_L2_q_idx+2::])
	ref_solver_idx = ref_out[-4].find(' ')
	ref_solver = ref_out[-4][:ref_solver_idx]
	ref_iters_idx_a = ref_out[-4].find('converged in')
	ref_iters_idx_b = ref_out[-4].find(' iterations')
	ref_iters = int(ref_out[-4][ref_iters_idx_a+13:ref_iters_idx_b])

	# Construct the command line
	if parallel:
		command_line = "mpirun -np 2 ./ex5p-nguyen -no-vis"
	else:
		command_line = "./ex5-nguyen -no-vis"

	command_line += f" -nx {nx}"
	command_line += f" -ny {ny}"
	command_line += f" -p {problem}"
	command_line += f" -o {order}"
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
		command_line += f' -k {kappa}'
	if hdg != 1:
		command_line += f' -hdg {hdg}'
	if nls != 0:
		command_line += f' -nls {nls}'

	print(f"RUNNING: {command_line}", end='\r', flush=True)

	# Run test case
	cmd_out = subprocess.getoutput(command_line)
	split_cmd_out = cmd_out.splitlines()

	# Process the result
	fail = False
	try:
		test_L2_t_idx = split_cmd_out[-1].find('= ')
		test_L2_q_idx = split_cmd_out[-2].find('= ')
		test_L2_t = float(split_cmd_out[-1][test_L2_t_idx+2::])
		test_L2_q = float(split_cmd_out[-2][test_L2_q_idx+2::])
		test_solver_idx = split_cmd_out[-4].find(' ')
		test_solver = split_cmd_out[-4][:test_solver_idx]
		test_iters_idx_a = split_cmd_out[-4].find('converged in ')
		test_iters_idx_b = split_cmd_out[-4].find(' iterations')
		test_iters = int(split_cmd_out[-4][test_iters_idx_a+13:test_iters_idx_b])
	except:
		fail = True

	if not fail:
		if test_solver == ref_solver and test_iters == ref_iters:
			if equal(ref_L2_t, test_L2_t) and equal(ref_L2_q, test_L2_q):
				print(f"{bcolors.OKGREEN}SUCCESS:{bcolors.RESET} {command_line}", flush=True)
			else:
				fail = True
		elif test_solver != ref_solver:
			print(f"{bcolors.HEADER}SKIPPING:{bcolors.RESET} {command_line} → incompatible preconditioner")
			skipped += 1
		else:
			print(f"{bcolors.WARN}DIFFERS:{bcolors.RESET} {command_line} → different number of iterations")
			print(cmd_out)
			failed += 1
	
	if fail:
		print(f"{bcolors.FAIL}FAILING:{bcolors.RESET} {command_line}", flush=True)
		print(cmd_out)
		failed += 1

print("----------------------------------------------------------------")
if skipped > 0:
	skipped_str = f" ({skipped} / {len(filenames)} skipped)"
else:
	skipped_str = ""
if failed == 0:
	print(f"{bcolors.OKGREEN}SUCCESS:{bcolors.RESET} all tests finished succesfully!" + skipped_str)
else:
	print(f"{bcolors.FAIL}FAIL:{bcolors.RESET} {failed} / {len(filenames)} tests failed!" + skipped_str)
