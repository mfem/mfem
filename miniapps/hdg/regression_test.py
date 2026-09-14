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
update_local = False
filenames = []
args = sys.argv[1:]
if '-par' in args:
	parallel = True
	args.remove('-par')
# Regenerate this machine's own copy of each reference into the LOCAL
# directory instead of comparing against it. See the block below for why the
# local set exists at all.
if '--update-local' in args:
	update_local = True
	args.remove('--update-local')
filenames = args

# Resolve the reference directories against this script rather than against
# the working directory, so the suite can be run from an out-of-source build
# directory, where only the binaries are present.
base = os.path.dirname(os.path.abspath(__file__))
if parallel:
	path = os.path.join(base, 'regress_test_par') + '/'
	local_path = os.path.join(base, 'regress_test_par_local') + '/'
else:
	path = os.path.join(base, 'regress_test') + '/'
	local_path = os.path.join(base, 'regress_test_local') + '/'

# The local reference directories are gitignored, and that is the whole point
# of them.
#
# regress_test/ and regress_test_par/ are the UPSTREAM reference set. They were
# recorded on their author's machine and they depend on that machine's
# configuration -- which BLAS, which SuiteSparse, which compiler -- so a
# handful of cases will never reproduce here to the digits they store, and the
# iteration counts of a few more will not either. That is not a defect and it
# is not ours to fix: we have no authority to rewrite someone else's
# references, and rewriting them is exactly what makes a diff that must never
# be pushed. Only genuinely NEW references, for cases that did not exist
# before, belong in the tracked directories.
#
# So this machine's own numbers go in regress_test_local/ instead, which git
# never sees. A local file shadows the tracked one of the same name; a tracked
# case with no local file is compared as it always was. Regenerate the local
# set with --update-local.
def resolve(name):
	if os.path.isfile(local_path + name):
		return local_path + name, True
	return path + name, False

from_local = set()
if len(filenames) == 0:
	names = set(os.listdir(path))
	if os.path.isdir(local_path):
		names |= set(n for n in os.listdir(local_path) if n.endswith('.txt'))
	filenames = []
	for name in sorted(names):
		f, is_local = resolve(name)
		filenames.append(f)
		if is_local:
			from_local.add(f)

failed = 0
skipped = 0
generated = 0
refused = 0

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
	# The time advance. The defaults are the miniapp's own, so every reference
	# written before there were transient ones -- all of which record
	# --ntimesteps 0 -- reconstructs the command it always did.
	tf = float(get_ref_param(filename, '--time-final', "1"))
	nt = int(get_ref_param(filename, '--ntimesteps', "0"))
	ode = int(get_ref_param(filename, '--ode-solver', "1"))
	# An explicitly recorded preconditioner wins; where there is none it is
	# inferred from the solver line further down. Upstream references predate
	# the option and carry none; locally generated ones record it.
	prec = int(get_ref_param(filename, '--preconditioner', "0"))

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
	# A transient case. All three options go on the command line even where
	# they hold their default values: the time advance is the whole subject of
	# such a reference, so it is spelled out rather than inferred.
	if nt != 0:
		command_line += f' -tf {tf} -nt {nt} -ode {ode}'
	# A recorded -nls is passed back, full stop. This used to be guarded by a
	# hand-maintained list of "is the problem nonlinear" flags, and the list
	# could only ever be wrong in one direction: a reference recording a
	# non-default solver that the list judges linear is RE-RUN WITH A
	# DIFFERENT SOLVER than the one it records. p3_o1_dg_upwind_rd.txt is
	# exactly that -- it records --nonlinear-solver 1 on a linear problem --
	# and --update-local refuses it for that reason, which is the guard doing
	# its job and is also why the heuristic has to go rather than the guard.
	#
	# Six references are affected, all linear cases recording
	# --nonlinear-solver 1. Checked rather than assumed: their output with and
	# without -nls 1 is identical except for the wall-clock timings, a linear
	# problem never reaching SetupNonlinearSolver() where solver_type is read.
	if nls != 0:
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

	# A reference that records the ITERATIVE preconditioner is run with it
	# forced, whether or not it says so as an option.
	#
	# This is what removes the perpetual skips. 49 of the 157 serial references
	# record GMRES+GS or Newton+GMRES+GS -- all of them through the Schur
	# preconditioner -- and a SuiteSparse build produces UMFPack there, so the
	# solver strings did not match and every one of them was reported
	# "incompatible preconditioner" and never run. They were not incompatible,
	# only unreachable: with -prec 1 all 49 reproduce their upstream numbers,
	# iteration counts included.
	#
	# Only GS is inferred, never UMFPack, and the asymmetry is the point. GS
	# exists in every build, so forcing it can always be honoured; UMFPack
	# needs SuiteSparse, and a build without it would be asked for something
	# that aborts -- turning today's clean skip into a crash. A build that HAS
	# SuiteSparse already produces UMFPack by default, so there is nothing to
	# force. A UMFPack reference therefore still skips where it must, exactly
	# as before.
	#
	# The recorded preconditioner is the LAST '+'-separated field of the part
	# before any '/': the string is built as
	# solver[+prec][+lin_prec][/inner_solver] (darcyop.cpp). Matching 'GS'
	# anywhere in it instead matches the GS inside **LBFGS**, which is how
	# every LBFGS reference came to be re-run with a spurious `-prec 1`. That
	# happened to be inert -- checked, serial and parallel, with and without,
	# byte-identical output -- but it was passing an option the reference never
	# asked for, which is the thing this block exists to avoid.
	prec_run = prec
	if prec_run == 0 and ref_solver.split('/')[0].split('+')[-1] == 'GS':
		prec_run = 1
	if prec_run != 0:
		command_line += f' -prec {prec_run}'

	print(f"RUNNING: {command_line}", end='\r', flush=True)

	# Run test case
	cmd_out = subprocess.getoutput(command_line)
	split_cmd_out = cmd_out.splitlines()

	if update_local:
		# Refuse unless every option the reference records survives into the
		# regenerated one. The script rebuilds the command from a FIXED option
		# list, so a reference recording something outside that list at a
		# non-default value would be silently rewritten as a DIFFERENT case --
		# which is a worse outcome than not regenerating it. Subset, not
		# equality: convdiff has gained options since these were recorded, so
		# the new block is legitimately longer.
		def options(lines):
			out = []
			for l in lines:
				if l.startswith('Options used:'):
					out = []
					continue
				if l.startswith('   --'):
					out.append(l.rstrip())
				elif out:
					break
			return out
		ref_opts = set(options(ref_out))
		new_opts = set(options(split_cmd_out))
		# Differences this script creates on purpose are not drift. It always
		# passes -no-vis, so a reference recorded without it (every upstream
		# one) says --visualization where ours says --no-visualization; and the
		# preconditioner is the option this run exists to add.
		for o in ('   --visualization', '   --no-visualization'):
			ref_opts.discard(o)
			new_opts.discard(o)
		ref_opts.discard(f'   --preconditioner {prec}')
		new_opts.discard(f'   --preconditioner {prec_run}')
		lost = sorted(ref_opts - new_opts)
		if lost:
			print(f"{bcolors.WARN}REFUSED:{bcolors.RESET} {os.path.basename(filename)} → "
			      f"would drop {', '.join(l.strip() for l in lost)}")
			refused += 1
			continue
		if not os.path.isdir(local_path):
			os.makedirs(local_path)
		out_name = local_path + os.path.basename(filename)
		with open(out_name, 'w') as fh:
			fh.write(cmd_out + '\n')
		print(f"{bcolors.OKGREEN}WROTE:{bcolors.RESET} {out_name}")
		generated += 1
		continue

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
if update_local:
	print(f"{bcolors.OKGREEN}WROTE{bcolors.RESET} {generated} local references"
	      f" into {local_path}")
	if refused > 0:
		print(f"{bcolors.WARN}REFUSED{bcolors.RESET} {refused} -- see above;"
		      " each records an option this script does not reconstruct")
	sys.exit(0)
if from_local:
	print(f"{len(from_local)} / {len(filenames)} references came from"
	      f" {local_path}")
if skipped > 0:
	skipped_str = f" ({skipped} / {len(filenames)} skipped)"
else:
	skipped_str = ""
if failed == 0:
	print(f"{bcolors.OKGREEN}SUCCESS:{bcolors.RESET} all tests finished succesfully!" + skipped_str)
else:
	print(f"{bcolors.FAIL}FAIL:{bcolors.RESET} {failed} / {len(filenames)} tests failed!" + skipped_str)
