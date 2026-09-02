#!/usr/bin/env python

# This is the regression testing script for (p)convdiff
# The reference cases are stored in the regress_test(_par) folder

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
# Locate the compared lines by what they say rather than by how far they are
# from the end. Indexing from the end works only while nothing else is ever
# printed after the solve, and --postprocess prints an extra error line; the
# last matching line is the same line the old [-1]/[-2]/[-4] picked whenever
# there is no such extra, so no existing reference reads differently.
def last_line_with(lines, text):
	for line in reversed(lines):
		if text in line:
			return line
	raise ValueError(f"no line containing {text!r}")

def parse_result(lines):
	t_line = last_line_with(lines, '|| t_h - t_ex ||')
	q_line = last_line_with(lines, '|| q_h - q_ex ||')
	s_line = last_line_with(lines, 'converged in ')
	L2_t = float(t_line[t_line.find('= ')+2::])
	L2_q = float(q_line[q_line.find('= ')+2::])
	solver = s_line[:s_line.find(' ')]
	a = s_line.find('converged in ')
	b = s_line.find(' iterations')
	# The postprocessed potential, when --postprocess printed one. Compared
	# like the other two rather than merely tolerated: a reference named for a
	# flag whose value nothing checks is not coverage of that flag.
	try:
		pp_line = last_line_with(lines, '|| t_pp - t_ex ||')
		L2_pp = float(pp_line[pp_line.find('= ')+2::])
	except ValueError:
		L2_pp = None
	return solver, int(s_line[a+13:b]), L2_q, L2_t, L2_pp

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

# Resolve the reference directories against this script rather than against
# the working directory, so the suite can be run from an out-of-source build
# directory, where only the binaries are present.
base = os.path.dirname(os.path.abspath(__file__))
if parallel:
	path = os.path.join(base, 'regress_test_par') + '/'
else:
	path = os.path.join(base, 'regress_test') + '/'

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
	trh1 = get_ref_option(filename, '--trace-H1')
	trbc = get_ref_option(filename, '--trace-ess-bc')
	pa = get_ref_option(filename, '--partial-assembly')
	upwind = get_ref_option(filename, '--upwinded')
	nonlin = get_ref_option(filename, '--nonlinear')
	nonlin_flux = get_ref_option(filename, '--nonlinear-flux')
	nonlin_pot = get_ref_option(filename, '--nonlinear-pot')
	nonlin_conv = get_ref_option(filename, '--nonlinear-convection')
	nonlin_diff = get_ref_option(filename, '--nonlinear-diffusion')
	nc = get_ref_option(filename, '--nc-mesh')
	pface_max = get_ref_option(filename, '--p-face-max')
	postproc = get_ref_option(filename, '--postprocess')

	def get_ref_param(file, param, default=""):
		ref_out = subprocess.getoutput("grep '^   "+param+"' "+file+" | cut -d ' ' -f 5")
		if len(ref_out) > 0:
			return ref_out.split()[0]
		else:
			return default

	# Anchored on a trailing space, because the reader above is not: a bare
	# '^   --p-refine' also matches '   --p-refine-x', and which value came
	# back would then depend on the order OptionsParser happened to print
	# them in. Used only by the options added below, to leave the reading of
	# every existing reference exactly as it was.
	def get_ref_param_exact(file, param, default=""):
		ref_out = subprocess.getoutput("grep '^   "+param+" ' "+file+" | cut -d ' ' -f 5")
		if len(ref_out) > 0:
			return ref_out.split()[0]
		else:
			return default

	pref = int(get_ref_param_exact(filename, '--p-refine', "0"))
	prefx = float(get_ref_param_exact(filename, '--p-refine-x', "0.5"))

	problem = int(get_ref_param(filename, '--problem'))
	order = int(get_ref_param(filename, '--order'))
	nx = int(get_ref_param(filename, '--ncells-x'))
	ny = int(get_ref_param(filename, '--ncells-y'))
	mesh = str(get_ref_param(filename, '--mesh'))
	if parallel:
		ref = int(get_ref_param(filename, "--serial-ref-levels", "-1"))
	else:
		ref = int(get_ref_param(filename, "--ref-levels", "-1"))
	kappa = float(get_ref_param(filename, '--kappa', "1"))
	hdg = int(get_ref_param(filename, '--hdg_scheme', "1"))
	nls = int(get_ref_param(filename, '--nonlinear-solver', "0"))

	file = open(filename, "r")
	ref_out = file.readlines()
	ref_solver, ref_iters, ref_L2_q, ref_L2_t, ref_L2_pp = parse_result(ref_out)

	# Construct the command line
	if parallel:
		command_line = "mpirun -np 2 ./pconvdiff -no-vis"
	else:
		command_line = "./convdiff -no-vis"

	if nx > 0:
		command_line += f" -nx {nx}"
		command_line += f" -ny {ny}"
	else:
		command_line += f" -m {mesh}"
		if parallel:
			command_line += f" -rs {ref}"
		else:
			command_line += f" -r {ref}"
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
		if trh1:
			command_line += ' -trh1'
		if trbc:
			command_line += ' -trbc'
	if pa:
		command_line += ' -pa'
	if upwind:
		command_line += ' -up'
	if nonlin:
		command_line += ' -nl'
	if nonlin_flux:
		command_line += ' -nlu'
	if nonlin_pot:
		command_line += ' -nlp'
	if nonlin_conv:
		command_line += ' -nlc'
	if nonlin_diff:
		command_line += ' -nld'
	if kappa != 1.:
		command_line += f' -k {kappa}'
	if hdg != 1:
		command_line += f' -hdg {hdg}'
	if nls != 0 and (nonlin or nonlin_flux or nonlin_pot or nonlin_diff):
		command_line += f' -nls {nls}'
	if pref != 0:
		command_line += f' -pref {pref}'
		if prefx != 0.5:
			command_line += f' -prefx {prefx}'
		if pface_max:
			command_line += ' -pmax'
	elif nc:
		# -pref implies it, so it is only worth passing on its own.
		command_line += ' -nc'
	if postproc:
		command_line += ' -pp'

	print(f"RUNNING: {command_line}", end='\r', flush=True)

	# Run test case
	cmd_out = subprocess.getoutput(command_line)
	split_cmd_out = cmd_out.splitlines()

	# Process the result
	fail = False
	try:
		test_solver, test_iters, test_L2_q, test_L2_t, test_L2_pp = parse_result(split_cmd_out)
		if (ref_L2_pp is None) != (test_L2_pp is None):
			raise ValueError('one side postprocessed and the other did not')
	except:
		fail = True

	if not fail:
		if test_solver == ref_solver and test_iters == ref_iters:
			if equal(ref_L2_t, test_L2_t) and equal(ref_L2_q, test_L2_q) \
			   and (ref_L2_pp is None or equal(ref_L2_pp, test_L2_pp)):
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
