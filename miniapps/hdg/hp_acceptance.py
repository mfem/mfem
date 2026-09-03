#!/usr/bin/env python3

# Acceptance test for the hp-adaptive demonstrator in anisodiff.
#
# regression_test.py compares one run against one stored reference. This
# cannot: what is being defended is a RELATION between three runs -- that hp
# reaches a tolerance at all, and that it gets there on fewer globally coupled
# unknowns than the alternatives by a stated factor. A stored answer would pin
# the wrong thing, because the adaptive path is allowed to change; the ratios
# are not.
#
# Serial only, and that is a limitation rather than an oversight: a shared face
# cannot be coarsened, so DarcyHybridization::SetTraceOrders() refuses a
# partitioned run whose faces sit below the trace ceiling. See its note.
#
# Usage:  python3 hp_acceptance.py [path-to-anisodiff]
#
# HP_ARGS overrides the hp run, which is how the criteria are shown to be able
# to fail: with `-no-cap-trace-at-element` the loop plateaus at 8.8e-7 and the
# reach check trips.

import os
import subprocess
import sys

# The matrix. Problem 5 at this anisotropy is a boundary layer of thickness
# 1/(pi sqrt(ks)) = 1/31 -- analytic everywhere, unresolvable on the starting
# mesh, which is what makes it discriminate between h and hp at all.
COMMON = "-p 5 -ks 1e2 -o 2 -hb -dg -no-vis"
UNIFORM_NX = [16, 32, 64, 128]
H_ADAPT = "-nx 8 -amr 22 -dorf -theta 0.6 -ppest"
HP = os.environ.get("HP_ARGS", "-nx 8 -amr 34 -dorf -theta 0.6 -hp -ppest")

# What hp reached when this was written: 4.49e-10 at M = 3264. The bar is set
# an order above it, so that ordinary drift in the adaptive path does not trip
# the test and a real loss of reach does.
REQUIRED_TOL = 1.0e-9

# Tolerances at which the cost ratios are checked. Every method has to have
# reached these, so they are set by the weakest -- uniform refinement runs out
# at 8.3e-5 on the largest mesh the test is willing to run.
RATIO_TOLS = [1.0e-3, 1.0e-4]

MAX_OVER_H = 2.0 / 3.0      # hp must need at most this fraction of h-adaptive
MAX_OVER_UNIFORM = 1.0 / 5.0

class C:
    OK = '\033[92m'
    FAIL = '\033[91m'
    WARN = '\033[93m'
    OFF = '\033[0m'

def run(binary, args):
    env = dict(os.environ, MKL_THREADING_LAYER="GNU", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1")
    out = subprocess.run([binary] + args.split(), capture_output=True,
                         text=True, env=env, timeout=3600)
    if out.returncode != 0:
        raise RuntimeError(f"{binary} {args} exited {out.returncode}")
    return out.stdout

def curve_adaptive(text):
    """(M, relative t error) per cycle, from the adaptive loop's own report."""
    pts = []
    for line in text.splitlines():
        if not line.startswith("iter:"):
            continue
        f = line.split()
        pts.append((int(f[f.index("M:") + 1]), float(f[f.index("t_err:") + 1])))
    return pts

def point_uniform(text):
    """(M, relative t error) for a single non-adaptive run."""
    m = err = None
    for line in text.splitlines():
        if line.startswith("dim(M) = "):
            m = int(line.split("=")[1])
        if "|| t_h - t_ex ||" in line:
            err = float(line.split("=")[1])
    if m is None or err is None:
        raise RuntimeError("could not parse a uniform run")
    return (m, err)

def cost_at(pts, tol):
    """Smallest M at which the method is at or below tol, or None.

    The envelope, not the raw sequence: an adaptive run's error is allowed to
    rise on a cycle, and what is being asked is what it cost to GET there."""
    best = None
    for m, e in sorted(pts):
        if e <= tol and (best is None or m < best):
            best = m
    return best

def main():
    binary = sys.argv[1] if len(sys.argv) > 1 else "./anisodiff"
    if not os.path.isfile(binary):
        print(f"{C.FAIL}NOT FOUND{C.OFF}: {binary}")
        return 2

    print("hp acceptance: three runs of " + COMMON)
    uniform = [point_uniform(run(binary, f"{COMMON} -nx {nx}"))
               for nx in UNIFORM_NX]
    hadapt = curve_adaptive(run(binary, f"{COMMON} {H_ADAPT}"))
    hp = curve_adaptive(run(binary, f"{COMMON} {HP}"))

    best_hp = min(e for _, e in hp)
    print(f"  uniform      : {len(uniform)} meshes, best {min(e for _,e in uniform):.3g}")
    print(f"  h-adaptive   : {len(hadapt)} cycles, best {min(e for _,e in hadapt):.3g}")
    print(f"  hp-adaptive  : {len(hp)} cycles, best {best_hp:.3g}")
    print()

    failed = []

    # 1. Reach.
    if best_hp <= REQUIRED_TOL:
        print(f"{C.OK}PASS{C.OFF}  hp reaches {best_hp:.3g} <= {REQUIRED_TOL:.3g}")
    else:
        print(f"{C.FAIL}FAIL{C.OFF}  hp reaches only {best_hp:.3g}, "
              f"required {REQUIRED_TOL:.3g}")
        failed.append("reach")

    # 2 and 3. Cost, against each alternative, at every shared tolerance.
    print(f"\n  {'tol':>8} {'uniform M':>10} {'h M':>8} {'hp M':>8} "
          f"{'hp/h':>7} {'hp/unif':>8}")
    for tol in RATIO_TOLS:
        mu, mh, mp = (cost_at(uniform, tol), cost_at(hadapt, tol),
                      cost_at(hp, tol))
        if mp is None:
            print(f"  {tol:>8.1e} {'-':>10} {'-':>8} {'-':>8}   hp never reached it")
            failed.append(f"hp missed {tol:.1e}")
            continue
        r_h = mp / mh if mh else float("inf")
        r_u = mp / mu if mu else float("inf")
        bad = []
        if mh and r_h > MAX_OVER_H:
            bad.append(f"hp/h {r_h:.3f} > {MAX_OVER_H:.3f}")
        if mu and r_u > MAX_OVER_UNIFORM:
            bad.append(f"hp/uniform {r_u:.4f} > {MAX_OVER_UNIFORM:.3f}")
        tag = C.FAIL + "FAIL" + C.OFF if bad else C.OK + "ok  " + C.OFF
        print(f"  {tol:>8.1e} {str(mu):>10} {str(mh):>8} {str(mp):>8} "
              f"{r_h:>7.3f} {r_u:>8.4f}  {tag}" +
              ("  " + "; ".join(bad) if bad else ""))
        failed.extend(bad)

    print()
    if failed:
        print(f"{C.FAIL}FAIL:{C.OFF} " + "; ".join(failed))
        return 1
    print(f"{C.OK}SUCCESS:{C.OFF} hp reaches its tolerance and stays inside "
          f"both cost bounds")
    return 0

if __name__ == "__main__":
    sys.exit(main())
