# Coupling MFEM to SUNDIALS IDA

Design for an `mfem::IDASolver` alongside the existing `CVODESolver`,
`ARKStepSolver` and `KINSolver`. Branch `sundials-ida-integration`, off
`master`.

**This is now built** -- see §11 for what was measured on the way and what is
left. The design below is unchanged from before the implementation except
where §11 records a departure, so the two can be read against each other.

IDA integrates a DAE in fully implicit residual form,

    F(t, y, y') = 0,

by variable-order variable-step BDF, with a Newton solve per step on
`J = dF/dy + c_j dF/dy'`. It is the only SUNDIALS package MFEM does not
wrap, and the only one that can integrate a system whose mass matrix is
singular -- mixed and hybridized discretisations, saddle-point systems,
index-1 constraint equations -- without the caller first eliminating the
algebraic unknowns by hand.

**Versions.** Every `file:line` below is **SUNDIALS v7.9.0** (tagged
2026-09-03), and every measurement was taken twice, against a v7.9.0 build
and against the v7.5.0 install here. The two agree bit for bit -- same
answers, same step counts, same Jacobian-setup counts -- so nothing in this
design depends on which of them is present. The load-bearing lines are also
untouched on the current development HEAD (`v7.8.0-98-g23c718aa4`,
`feature/suncomplextype-merge`): checked by diffing `src/ida/ida_ls.c`
against the tag for `scalesol`, `matrixbased`, `iterative`, `nli_inc`,
`SUNLinSolResid`, `SUNMatZero` and `numiters`, which returns nothing.

## 1. What the three existing couplings share, and where they differ

All three derive from `SundialsSolver` (`linalg/sundials.hpp:373`), which
owns `sundials_mem`, the state `SundialsNVector *Y`, a `SUNMatrix A`, a
`SUNLinearSolver LSA`, and the last returned `flag`. The pattern is the
same in each:

* a `static` C-linkage trampoline per SUNDIALS callback, recovering the
  wrapper from `user_data` or from `SUNMatrix`/`SUNLinearSolver` `content`
  via the `GET_CONTENT` macro;
* an **empty custom `SUNMatrix`** -- `SUNMatNewEmpty()` with `content = this`
  and only `getid`/`destroy` filled in. It carries no data. It exists
  because SUNDIALS' matrix-based linear-solver interfaces refuse a NULL
  matrix, and because it is the channel through which the trampolines find
  the wrapper;
* an **empty custom `SUNLinearSolver`** -- `SUNLinSolNewEmpty()` with
  `gettype`/`solve`/`free`, `gettype` returning `SUNLINEARSOLVER_MATRIX_ITERATIVE`
  (`linalg/sundials.cpp:659`);
* the actual work delegated back to the MFEM operator, which sets up *and*
  solves its own linear system. Nothing is ever assembled on the SUNDIALS
  side.

Where they differ is only in which SUNDIALS entry point installs the setup
callback:

| package | attach | setup callback | residual/RHS source |
|---|---|---|---|
| CVODE | `CVodeSetLinearSolver` + `CVodeSetLinSysFn` | builds all of `A = I - γJ` | `TimeDependentOperator::Mult` |
| ARKODE | `MFEM_ARKode(SetLinearSolver)` + `MFEM_ARKode(SetLinSysFn)` -- a macro spelling `ARKStep*` before SUNDIALS 7.1 and `ARKode*` after (`sundials.cpp:161`) -- plus a second `M`/`LSM` pair for the mass matrix | builds `A = M - γJ` | `Mult`, or `ExplicitMult`+`ImplicitMult` for IMEX |
| KINSOL | `KINSetLinearSolver` + **`KINSetJacFn`** | fills the Jacobian only | `Operator::Mult` of a `NewtonSolver` operator |

IDA follows KINSOL here: there is no `IDASetLinSysFn`, only
`IDASetJacFn(ida_mem, IDALsJacFn)` with

```c
int jac(sunrealtype t, sunrealtype c_j, N_Vector y, N_Vector yp, N_Vector r,
        SUNMatrix J, void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);
```

so the wrapper receives `t`, `c_j`, `y`, `y'` and the current residual --
everything the MFEM operator needs.

## 2. IDA rejects the linear-solver object the other three wrappers build

This is the one hard constraint, and it was measured rather than reasoned
about. `IDASetLinearSolver` requires a non-DIRECT solver to supply `resid`
and `numiters` ops (`ida_ls.c:119`); CVODE and KINSOL guard those calls
instead (`cvode_ls.c:1813`, `kinsol_ls.c:1243`), and IDA is alone among the
four in demanding them. A standalone probe -- Robertson's index-1 DAE, with
a linear solver built to
exactly the shape `CVODESolver::UseMFEMLinearSolver()` builds:

| `gettype` | `numiters`/`resid` | `IDASetLinearSolver` | `tol` handed to `solve` |
|---|---|---|---|
| `MATRIX_ITERATIVE` | absent (MFEM's shape today) | **-3, rejected** | -- |
| `MATRIX_ITERATIVE` | supplied | 0, accepted | 4.95e-16 |
| `DIRECT` | absent | 0, accepted | **0** |

Both accepted variants integrate to SUNDIALS' own `idaRoberts_dns` answer
at `t = 4e10` (`5.2348e-08, 2.0939e-13, 9.99999948e-01` against a reference
of `5.2083e-08, 2.0834e-13, 9.9999995e-01`), in 1300 steps with 114
Jacobian setups -- the same three numbers on v7.9.0 and on v7.5.0.

Two consequences:

* **`numiters` must return a non-zero count.** When it returns 0, IDA takes
  a branch that copies `SUNLinSolResid(LS)` -- NULL for an empty solver --
  over the right-hand side (`ida_ls.c:1567`). Measured: segfault. It must
  return `1`; the reported linear-iteration statistic is then a placeholder
  and should say so in the doxygen. `resid` must merely be non-NULL as an
  *op*; the vector it returns is never read when `numiters >= 1`.
* **`MATRIX_ITERATIVE` is the right `gettype`, not `DIRECT`,** because IDA
  hands a real, error-weight-scaled tolerance to the solve in that mode and
  hands `0` in the other. `TimeDependentOperator::SUNImplicitSolve()`
  implementations use it -- `examples/sundials/ex16.cpp:535` does
  `T_solver.SetRelTol(tol)` -- so `DIRECT` would silently ask every existing
  operator's inner CG for a relative tolerance of zero.

The existing shared `LSGetType`, `LSFree`, `MatGetID` and `MatDestroy` are
reused unchanged; IDA adds two static ops of its own. No `zero` op is
needed: IDA calls `SUNMatZero` only for `SUNLINEARSOLVER_DIRECT`
(`ida_ls.c:1393`).

## 3. The residual: MFEM already has the form IDA wants

`TimeDependentOperator` is documented (`linalg/operator.hpp:355`) as

    F(u, k, t) = G(u, t),      k = du/dt,

which is IDA's residual with `res = F(y, y', t) - G(y, t)`. The
`Type` enum says which half is present, so the residual trampoline is a
three-way dispatch on facts the operator already declares:

| `Type` | residual | virtuals used |
|---|---|---|
| `EXPLICIT` (`F = k`) | `r = y' - f(y,t)` | `Mult` |
| `HOMOGENEOUS` (`G = 0`) | `r = F(y, y', t)` | `ImplicitMult` |
| `IMPLICIT` | `r = F(y, y', t) - G(y, t)` | `ImplicitMult`, `ExplicitMult` |

The `EXPLICIT` row matters out of proportion to its interest: it means
**every operator that already runs under CVODE runs under IDA**, as a DAE
with an identity mass matrix. That is the verification path in §7, and it
costs no new operator code.

`ImplicitMult`/`ExplicitMult` are exactly PETSc's TS `IFunction`/
`RHSFunction` pair, and `linalg/petsc.cpp:4435` and `:4461` call them for
that purpose. Honesty about the size of that claim: **`examples/petsc/ex9p.cpp`
is the only class in the tree that implements them.** So "operators written
for PETSc's implicit TS drive IDA unchanged" is true and has a sample of one.

## 4. The linear system: two routes, one of them free

IDA needs `J = dF/dy + c_j dF/dy'`, with `F` the *residual*, so including
`-dG/dy`.

### 4a. New virtuals (primary route)

```cpp
/// Setup the DAE linear system  J = dF/dy + cj dF/dy'  at (@a y, @a yp, t).
virtual int SUNImplicitSetupDAE(const Vector &y, const Vector &yp,
                                const Vector &res, real_t cj);
/// Solve  J @a x = @a b  with J from SUNImplicitSetupDAE().
virtual int SUNImplicitSolveDAE(const Vector &b, Vector &x, real_t tol);
```

on `TimeDependentOperator`, defaulting to `mfem_error()` like their
siblings. This is the general route and the one an operator written *for* a
DAE should take. It exists because the current `SUNImplicitSetup(y, v, jok,
jcur, gamma)` signature has no `y'` argument at all, so it cannot express
where `dF/dk` is to be evaluated when `F` is nonlinear in `y'` -- which is
the case IDA is for.

`jok` has no IDA analogue: IDA decides Jacobian lagging by not calling the
callback, so the setup is always a fresh one.

### 4b. Reuse of the CVODE/ARKODE hook (opt-in, exact)

`SUNImplicitSetup`'s own documentation (`linalg/operator.hpp:585`) defines
its matrix as

    A(γ) = dF/dk + γ (dF/du - dG/du),

and IDA's is `J = (dF/du - dG/du) + c_j dF/dk`. Hence

    J = c_j · A(1/c_j)

**identically, for any F and G** -- it is not a mass-matrix special case.
So an operator that already implements `SUNImplicitSetup`/`SUNImplicitSolve`
can serve IDA's linear system by being called with `gamma = 1/c_j` and
having the right-hand side scaled by `1/c_j`.

Verified entrywise on Robertson at four states and `c_j` spanning
`1e-6 … 1e6`: max relative difference `1.2e-16`. End to end, the two routes
integrate to the same answer in 1284 against 1300 steps and 1713 against
1725 Newton iterations -- the small spread is the round trip through
`1/c_j`, not a different matrix. (Comparing final *answers* would have been
the weak test: a wrong hybridized Jacobian costs Newton iterations, not
correctness.)

Exposed as `UseMFEMLinearSolverFromODEForm()`, never as a silent default,
with two documented limits:

* the `v` argument of `SUNImplicitSetup` is form-dependent (`inv(M)g` for
  form 1, `g` for forms 2 and 3) and is not always reconstructible from
  `(y, y', r)`. It is recovered where it is -- `v = y' - r` in the
  `EXPLICIT` case -- and passed as the residual otherwise. In practice
  MFEM's own implementations ignore it: `ex16.cpp` and `ex10.cpp` both use
  only `y` and `gamma`, and both ignore `jok` too;
* it presumes `dF/dk` is independent of `k`, which every form MFEM
  documents satisfies (`F = k`, `F = Mk`, `F = Mk - g`).

### 4c. `UseSundialsLinearSolver()`

`SUNLinSol_SPGMR` with IDA's internal difference-quotient
Jacobian-times-vector, matching the CVODE and ARKODE wrappers. Free.

## 5. What IDA needs that no other wrapper has: `y'`, and consistency

`ODESolver::Step(Vector &x, real_t &t, real_t &dt)` has no place for `y'`.
`IDASolver` therefore owns a second `SundialsNVector *YP` and exposes it
directly. This is the part of the design that is genuinely new rather than
a transcription of `CVODESolver`.

IDA requires `F(t0, y0, y'0) = 0` to start. `IDACalcIC` will find it:

* `IDA_YA_YDP_INIT` -- given the differential components of `y`, compute the
  algebraic components of `y` and all of `y'`. Needs an `id` vector marking
  which is which (`IDASetId`);
* `IDA_Y_INIT` -- given `y'`, compute `y`.

Measured through the MFEM-shaped linear solver, starting Robertson from a
deliberately inconsistent state (`y3 = 0.5`, `y' = 0`, `|F| = 5.0e-01`):
`IDACalcIC` returned 0 and gave `y3 = 4.2e-21`, `y' = (-0.04, 0.04, 0)`,
`|F| = 0` -- the exact consistent state. So the feature works through an
empty custom `SUNMatrix`, which was not obvious.

`IDASetSuppressAlg` excludes the algebraic components from the local error
test. For a FE constraint block this is not a nicety: without it the step
size is controlled by unknowns that have no dynamics.

## 6. Proposed interface

```cpp
class IDASolver : public ODESolver, public SundialsSolver
{
protected:
   int step_mode;                  ///< IDA_NORMAL (default) or IDA_ONE_STEP
   int root_components;
   SundialsNVector *YP;            ///< Derivative vector y'. Owned.
   SundialsNVector *ID;            ///< Differential/algebraic marker, or NULL.
   mutable Vector g_work;          ///< Scratch for G(u,t) in the residual.
   bool use_ode_form_linsys;       ///< Route 4b rather than 4a.

   static int Res(sunrealtype t, N_Vector yy, N_Vector yp, N_Vector rr,
                  void *user_data);
   static int LinSysSetup(sunrealtype t, sunrealtype cj, N_Vector yy,
                          N_Vector yp, N_Vector rr, SUNMatrix J,
                          void *user_data, N_Vector, N_Vector, N_Vector);
   static int LinSysSolve(SUNLinearSolver LS, SUNMatrix J, N_Vector x,
                          N_Vector b, sunrealtype tol);
   static int root(sunrealtype t, N_Vector yy, N_Vector yp,
                   sunrealtype *gout, void *user_data);

public:
   IDASolver();
#ifdef MFEM_USE_MPI
   IDASolver(MPI_Comm comm);
#endif
   void Init(TimeDependentOperator &f_) override;
   void Step(Vector &x, real_t &t, real_t &dt) override;

   // --- the DAE-specific half -------------------------------------------
   /// Set y'(t0). Defaults to zero, which is consistent only by accident.
   void SetInitialDerivative(const Vector &yp0);
   /// The derivative at the last time reached.
   const Vector &GetDerivative() const { return *YP; }
   /// Mark differential (true) against algebraic (false) components.
   void SetDifferentialComponents(const Array<bool> &is_differential);
   /// Exclude algebraic components from the local error test.
   void SetSuppressAlgebraic(bool suppress = true);
   /// Correct (y0, y'0) to satisfy F(t0,y0,y'0)=0; @a tout1 sets the sign
   /// and scale of the first step. Requires SetDifferentialComponents()
   /// for IDA_YA_YDP_INIT.
   void ComputeConsistentIC(real_t tout1, int icopt = IDA_YA_YDP_INIT);

   // --- the transcription of CVODESolver ---------------------------------
   void UseMFEMLinearSolver();             ///< route 4a, the default
   void UseMFEMLinearSolverFromODEForm();  ///< route 4b
   void UseSundialsLinearSolver();         ///< route 4c
   void SetStepMode(int itask);
   void SetSStolerances(real_t reltol, real_t abstol);
   void SetSVtolerances(real_t reltol, Vector abstol);
   void SetMaxStep(real_t dt_max);
   void SetMaxNSteps(int steps);
   void SetMaxOrder(int max_order);        ///< BDF, 1..5
   void SetLinearSolutionScaling(bool onoff);
   void SetRootFinder(int components, RootFunction func);
   long GetNumSteps();
   void PrintInfo() const;                 ///< IDAPrintAllStats
   virtual ~IDASolver();
};
```

`Init()` mirrors `CVODESolver::Init()` -- resize check against
`saved_global_size`, `IDACreate`/`IDAInit`, `IDASetUserData(this)`, default
tolerances, `UseMFEMLinearSolver()`, `reinit = true` -- with `IDAReInit`
taking both `Y` and `YP`.

**One guard worth its cost.** The first `Step()` evaluates
`F(t0, y0, y'0)` once and warns, naming `ComputeConsistentIC()`, if it is
large against the tolerances. A user may legitimately have `y' = 0`
consistent, so this is a warning and not an abort; but the default `y' = 0`
is otherwise a silent wrong start, and one residual evaluation is a cheap
way to make it loud.

`SetLinearSolutionScaling` exposes `IDASetLinearSolutionScaling`. It is
**on by default** for a matrix-based solver (`ida_ls.c:280`) and scales the
correction by `2/(1 + c_j/c_j^{old})` to account for a Jacobian formed at a
stale `c_j` (`ida_ls.c:1582`). That is the right default for both routes,
since both hold a factorisation from the last setup.

## 7. Verification

Nothing in `tests/unit` covers SUNDIALS today, so the first two items are
also the first SUNDIALS unit test in MFEM.

1. **Robertson**, `tests/unit/linalg/test_sundials_ida.cpp`, tag
   `[SUNDIALS]`, guarded by `MFEM_USE_SUNDIALS`. No finite elements: it
   pins the residual dispatch, the Jacobian mapping, and `IDACalcIC` against
   SUNDIALS' own published answer. The probe behind §2 and §5 is this test
   in C; it is rewritten clean rather than promoted.
2. **The two linear-system routes agree.** Same problem through 4a and 4b;
   the matrices must agree entrywise, not merely the answers -- §4b records
   why.
3. **`examples/sundials/ex16.cpp` gains IDA** next to CVODE and ARKODE. The
   heat equation with `M` non-singular is not a DAE, which is the point:
   IDA and CVODE must produce the same trajectory to the integration
   tolerance on an operator neither of them was written for. This is the
   `EXPLICIT`-row claim of §3 under test.
4. **A genuine DAE example.** Transient Stokes, `M v' + K v + B^T p = f`,
   `B v = 0` -- index 2, `dF/dy'` singular, and the case where
   `SetDifferentialComponents` and `SetSuppressAlgebraic` earn their place.
   No existing MFEM example is a DAE, so this has to be written. §9 is the
   use case that motivates the design, but it lives outside this repository
   and cannot be the upstream example. The two together cover both classes
   IDA is for: §9's hybridized system is **index 1** (its algebraic block is
   solvable for the algebraic unknowns), transient Stokes is **Hessenberg
   index 2** (the pressure does not appear in the constraint at all). An
   index-2 example is the one that exercises `SetSuppressAlgebraic`
   seriously, since without it the error test is applied to a variable whose
   error estimate is meaningless.
5. **Parallel.** `IDASolver(MPI_Comm)` over `SundialsNVector`'s parallel
   path, verified rank-count independent. Note the local build: SUNDIALS at
   `/home/ian/projects/sundials/install` has no MPI; `install-mpi` has
   `libsundials_ida` and `libsundials_nvecparallel` and is the tree to
   configure against for this item.

## 8. Build plumbing

* `config/defaults.mk:319` -- add `-lsundials_idas` to `SUNDIALS_LIB`.
  IDAS exports the whole `IDA*` API (checked with `nm -D`), exactly as
  MFEM already links `cvodes` rather than `cvode`, so one library covers
  IDA now and a future `IDASSolver` (adjoint sensitivity, the analogue of
  `CVODESSolver`) later.
* `config/cmake/modules/FindSUNDIALS.cmake:35` -- `ADD_COMPONENT IDAS
  "include" idas/idas.h "lib" sundials_idas`.
* `CMakeLists.txt:390` -- append `IDAS` to `SUNDIALS_COMPONENTS`.
* `linalg/sundials.hpp` -- `#include <ida/ida.h>`, the class, guarded as
  its siblings are.
* `linalg/operator.hpp/.cpp` -- the two new virtuals of §4a.

Minimum version: the design uses nothing newer than SUNDIALS 6, and nothing
it relies on has changed through v7.9.0. The only additions to IDA's public
API between v7.5.0 and v7.9.0 are `IDASetMaxNumConstraintFails`,
`IDAGetNumConstraintCorrections` and `IDAGetNumConstraintFails` -- all three
belonging to `IDASetConstraints`, which §10 leaves out.

## 9. The driving use case: a hybridized mixed system with an NPC-style solve

**No code from that work is on this branch and none should be.** A clean,
self-contained edit is what lands upstream; this section records why the
design is shaped the way it is, and what was checked.

### 9.1 The shape

A hybridized mixed discretisation carries a flux `q`, a potential `p` and a
skeleton trace `λ`, and the system is

```
| Mu(q,p)  ±Bᵀ  Cᵀ | | q |   | bu |
| B         D   E  | | p | = | bp |
| C         G   H  | | λ |   | br |
```

Only the potential (and sometimes the flux) carries a time derivative; **the
trace never does**. So

    dF/dy' = diag(δ_q M_q, δ_p M_p, 0)

is structurally singular. This is a DAE, not an ODE with a mass matrix, and
it is **index 1** exactly when the Jacobian block coupling the algebraic
unknowns to the algebraic equations is nonsingular -- for the common
potential-only-transient case, `H − C Mu⁻¹ Cᵀ`. That is a different matrix
from the one hybridization inverts, so it is a genuine extra condition and
not automatically satisfied; it is nonsingular under the same `τ > 0`
stabilization that makes the trace system solvable.

Today this is integrated by writing backward Euler out by hand. A
`FunctionCoefficient` returns a scalar `idt = 1/dt`, a mass integrator
carrying it is installed on the flux and potential forms, a *second* pair of
mass forms builds the old-time right-hand side `M yⁿ/dt`, and everything is
reassembled whenever `dt` changes. `ImplicitSolve` then returns the slope.

### 9.2 The mapping onto this design

It is closer than any other use case examined, because the NPC interface is
*already* a setup/solve pair for a DAE Jacobian -- it just does not know it:

| this design | what already exists there |
|---|---|
| `ImplicitMult(y, y', r)` | the full residual `F(q,u,λ)`, plus one mass apply on `y'` |
| `SUNImplicitSetupDAE(y, yp, res, cj)` | set the scalar to `cj`, reassemble, assemble+factor the Jacobian |
| `SUNImplicitSolveDAE(b, x, tol)` | reduce to the trace, solve, recover the local blocks |
| `id` vector | the per-block "does this carry a time derivative" flags |
| `SetSuppressAlgebraic` | -- |
| `ComputeConsistentIC` | -- |

Three things fall out of that table.

* **`c_j` enters through the knob that already exists.** The time term is
  introduced by a single scalar multiplying a mass integrator. IDA's `c_j`
  is that scalar, generalised from `1/Δt` to variable-order BDF. Nothing
  about the assembly path has to change.
* **The BDF history machinery disappears.** The second pair of mass forms
  exists only to build `M yⁿ/Δt`. IDA takes `y'` as an argument and keeps
  the history itself, so that half is not merely unnecessary, it would be
  wrong to keep.
* **Consistent initialisation is the feature with no counterpart.** Given
  an initial potential, the flux and the trace are *determined* by the two
  algebraic rows, and today the caller must arrange them. `IDACalcIC` with
  `id = (0, 1, 0)` computes them.

### 9.3 What was measured

A synthetic block DAE with exactly that structure -- `q` algebraic, `p`
differential, `λ` algebraic, a flux mass depending on the potential so the
local Jacobian has a nonzero `(0,1)` block -- integrated through IDA twice:
once factoring the full Jacobian, once through the hybridized elimination
(factor the local `(q,p)` block, form `S = H − [C G] Mloc⁻¹ [Cᵀ; E]`, solve
for the trace, recover the local blocks). No MFEM, no HDG code.

* Index-1 test on the data: `det[[A, Cᵀ],[C, H]] = 1.0`, nonsingular.
* `IDACalcIC(IDA_YA_YDP_INIT)` from a deliberately wrong `q0` and `λ0`,
  given only `p0`: `|F|` from **3.11e+01 to 3.33e-15**, recovering `q0`,
  `λ0` and `p0'`.
* The two routes on the **same `(J, b)`** at three states and `c_j` from
  `1e-5` to `1e6` -- 18 combinations -- agree to a worst relative
  difference of **3.7e-14**, and that worst case is the ill-conditioned one
  (`|x| = 3.4e3` at `c_j = 1e-2`); the rest are at `1e-16`.

Comparing the two *trajectories* would have been the weak test and it is
recorded here as such: they agree to 12 figures early and 9 late, but the
integrators took 493 against 465 steps, so the comparison was measuring the
step controller. The fixed-`(J, b)` comparison is the one that isolates the
elimination.

### 9.4 Two costs this design does not remove, named rather than discovered later

* **Reassembly frequency.** That code reassembles the forms whenever `Δt`
  changes, which under a fixed-step integrator is once. A variable-step BDF
  changes `c_j` continually. IDA bounds the damage rather than removing it:
  `ida_dcj` (default **0.25**, `ida_impl.h:70`) triggers a setup only when
  `c_j/c_j^{old}` leaves `[0.6, 1.67]`, and the probe above measured **61
  and 65 setups over 493 and 465 steps** -- about one in eight. Whether one
  full reassembly per eight steps is affordable is a question for that code,
  not for this interface, but it is the first thing to measure there.
* **Route 4b is unavailable.** Nothing in that code implements
  `SUNImplicitSetup`/`SUNImplicitSolve` -- it uses MFEM's own
  `ImplicitSolve` path. So the free reuse route of §4b buys it nothing, and
  the new virtuals of §4a are load-bearing rather than a convenience. That
  is the strongest argument in the design for adding them.

One thing that looks like an obstacle and is not: that interface's Jacobian
handle is **solve-only** -- its `Mult()` aborts, because the local blocks are
factored in place -- which rules out JFNK there. IDA never applies `J`, only
solves with it, so §4a is unaffected. §4c (SPGMR on difference quotients of
the residual) also still works, since it differences the *residual*, but it
would run unpreconditioned and is not the route to use here.

## 10. Deliberately not in this design

* **IDAS / adjoint sensitivity.** `CVODESSolver` is the template and the
  library link is already chosen to allow it, but it is a separate piece.
* **Preconditioner-only operation** (`IDASetPreconditioner`,
  `IDASetJacTimes`) for a genuinely matrix-free trace solve. `UseSundialsLinearSolver()`
  covers the unpreconditioned case; the preconditioned one wants an
  interface MFEM does not have for CVODE either.
* **`IDASetConstraints`** (positivity and sign constraints on `y`), and
  with it the three functions v7.9.0 added around it --
  `IDASetMaxNumConstraintFails`, `IDAGetNumConstraintCorrections`,
  `IDAGetNumConstraintFails`. Cheap to add, no analogue in the other three
  wrappers, and better added when something asks for it. That SUNDIALS is
  still extending this corner is a mild argument for waiting rather than
  guessing at the interface.
* **Changing the CVODE/ARKODE/KINSOL wrappers.** Adding `numiters`/`resid`
  to the shared linear solver would be harmless but is not needed by any of
  them; the two new ops stay IDA-local.


---

## 11. Built. What changed, what was measured, what is left

### Departures from the design above

* **`ComputeConsistentIC()` takes the state as an argument.** §6 had it
  acting on "the vector last passed to `Step()`", which cannot work: the
  class only learns which vector holds the state when `Step()` is called with
  it, and the whole point is to correct the pair *before* the first step.
* **`GetResidual()` was added**, with the dispatch moved into a `Residual()`
  member the SUNDIALS callback also uses. Without it the dispatch table of §3
  could only be tested by whether an integration converged, and its
  `IMPLICIT` row not at all.
* **`SetSVtolerances()` does not mirror `CVODESolver`'s.** IDA combines the
  tolerance vector with the state elementwise to build the error weights, and
  those operations dispatch on their first argument's type -- so a
  default-constructed (serial) `SundialsNVector` in a parallel run has a
  serial content struct read through a parallel accessor. The vector is built
  with the state's communicator instead. **`CVODESolver::SetSVtolerances()`
  has the same defect and was deliberately left alone**, being pre-existing
  and unrelated to this change.
* **`Init()` drops the differential/algebraic marker on a resize**, so the
  guards requiring it refuse loudly instead of passing while IDA holds
  nothing.

### Measured

* **Unit suite: 415 cases / 4,207,835 assertions, all passing.** Excluding
  `[IDA]` gives 410 / 4,207,678, so the delta is exactly the five new cases
  and their 157 assertions.
* **IDA against CVODE on `ex16`**, at a *common* final time -- the example
  runs its SUNDIALS solvers in one-step mode, where they step past `t_final`
  and do not interpolate back, so comparing the two final `.gf` files as they
  stand compares states at different times and shows a spurious 1e-4 that
  does not shrink:

  | `reltol` = `abstol` | CVODE vs IDA (EXPLICIT form) | vs IDA (IMPLICIT form) |
  |---|---|---|
  | 1e-4 | 7.0e-06 | 7.0e-06 |
  | 1e-6 | 8.5e-07 | 6.0e-07 |
  | 1e-8 | 8.5e-08 | 8.5e-08 |
  | 1e-10 | 8.5e-08 | 0.0 |

  The floor at 8.5e-08 is the `.gf` file's eight-significant-digit print
  resolution, not a discrepancy between the integrators.
* **The inconsistent-IC warning fires.** Robertson from `y = (1, 0, 0.5)`
  with `y' = 0`: it reports `|R| = 5.0e-01` against a `1.0e-06` threshold,
  and IDA then gives up at `t = 0` with "the error test failed repeatedly or
  with |h| = hmin" -- the symptom, with nothing about the cause. The same
  start with a consistent `y'` warns not at all and integrates.
* **Parallel is rank-count independent**, twelve Robertson systems
  distributed over the ranks so the weighted RMS norm is a reduction over
  every unknown. `y0[0]` at `t = 400`:

  | `reltol` | 1 rank | 2 ranks | 4 ranks | spread |
  |---|---|---|---|---|
  | 1e-8 | 0.4505186813852 | 0.4505186684390 | 0.4505186796108 | 1.3e-08 |
  | 1e-10 | 0.4505186687016 | 0.4505186686316 | 0.4505186686316 | 7.0e-11 |

  The spread tracks the requested tolerance, which is what says the ranks are
  solving one problem rather than agreeing by luck; at 1e-10 two and four
  ranks agree to all sixteen digits.

### Left

1. **Untested API**: `UseSundialsLinearSolver()`, `SetRootFinder()`,
   `SetMaxOrder()` and `SetLinearSolutionScaling()`.
2. **No accessor for the Newton or Jacobian-setup counts.** `GetNumSteps()`
   exists and `PrintInfo()` only prints, so "the two linear-system routes
   cost about the same" cannot be asserted.
3. **IDAS / adjoint sensitivity**, as §10 says. The library link is already
   chosen to allow it.
4. **A parallel index-2 example.** `ex5` here is serial; there is no `ex5p`.

Items 1 and 2 of the previous list are done -- see §12.

## 12. The parallel case and the index-2 example

### `tests/unit/linalg/test_sundials_ida.cpp` gained a `[Parallel]` case

Seven Robertson systems distributed over the ranks -- seven because it
divides evenly at neither two nor four, so the uneven local sizes exercise
`SundialsNVector`'s global-length reduction. It asserts on **every** rank's
own systems, `MPI_Allreduce`s the constraint defect so every rank asserts the
same number, and `MPI_Allreduce`s the pass/fail with `MPI_LAND` -- the last
because `punit_test_main.cpp` routes Catch2's output to a null stream off
root, so a `REQUIRE` failing on rank 1 prints nothing at all.

It uses `SetSVtolerances` deliberately: that is the method written to build a
tolerance vector of the state's N_Vector type (§11), and this case is the
only thing that would catch a regression in it.

Passes at 1, 2, 3 and 4 ranks. Parallel suite **100 cases / 70,961
assertions** on 2 ranks, against 99 / 70,938 excluding `[IDA]` -- the delta
is exactly the one new case and its 23 assertions.

**A rank owning no systems is refused, and the reason is a pre-existing
`SundialsNVector` defect** shared by all four wrappers, not anything of IDA's.
`_SetNvecDataAndSize_()`'s parallel branch guards its global-length reduction
with `glob_size == 0 && glob_size != size`, and `glob_size` is zero on entry
to that test, so the condition is just `size != 0`: an empty rank skips an
`MPI_Allreduce` every other rank enters. Measured rather than read -- two
systems over four ranks does **not** hang. The orphaned reduction pairs with
the next collective the empty ranks reach, which is the `MPI_Allreduce`
inside `N_VNewEmpty_Parallel()`, and SUNDIALS' own "global_length does not
equal the computed global length" check then returns NULL, so it aborts in
`SundialsNVector::MakeNVector()`. Loud, in a place that names nothing
relevant.

### `examples/sundials/ex5.cpp` is the index-2 example

Transient mixed Darcy, `M u' + Mk u - B^T p = F(t)`, `-B u = G(t)`, RT/L2 as
`examples/ex5.cpp` discretises the steady problem. `dR/dy'` is `diag(M, 0)`
and the pressure is absent from the constraint row: Hessenberg index 2, where
`ex16` is an ODE that IDA treats as a trivial DAE.

The exact solution is chosen to lie in the discrete spaces --
`u = a(t) x`, `p = a(t)(x_0 - x_1)`, `a(t) = 1 + sin(t)/2` -- so the spatial
error is zero by construction and what the example reports is the **time**
integration error alone. That also makes the consistent initial condition
available in closed form, which matters because for an index-2 system it is
two conditions and not one: the constraint `-B u_0 = G(0)`, and the hidden
constraint `-B u'_0 = G'(0)` obtained by differentiating it, which is what
determines `p_0` at all.

Measured, `-r 1 -o 1 -tf 1.0 -dt 0.1`, relative L2 errors:

| `-rtol` | `-atol` | u | p | steps |
|---|---|---|---|---|
| 1e-4 | 1e-6 | 7.86e-05 | 1.39e-04 | 16 |
| 1e-6 | 1e-8 | 1.09e-06 | 1.29e-07 | 46 |
| 1e-8 | 1e-10 | 1.32e-08 | 2.39e-09 | 66 |
| 1e-10 | 1e-12 | corrector convergence fails at t = 0 | | |

**The error tracks the tolerance over three decades and then the method
stops, and the linear solver is not why.** That was established by
elimination, not by argument: dropping the MINRES relative-tolerance floor
from 1e-12 to zero changes nothing, raising its iteration cap from 1e3 to
5e4 changes nothing, and a run that succeeds reports `LS fails = 0` beside
`NLS fails = 6`. What is left is intrinsic to index two --
`SetSuppressAlgebraic()` takes the algebraic block out of the **error** test,
but IDA's **corrector** test is a weighted norm over the whole Newton
correction, and the algebraic component of that correction scales like `1/h`.
Shrink the step and the test stops being satisfiable however well the linear
system is solved. Worth a reader's attention precisely because it presents
as a linear-solver problem and is not.

Two options exist to make the claims checkable rather than assertions:

* **`-no-sa`** leaves the algebraic block in the error test. Measured: aborts
  at `t = 0` with `h` at 3.9e-15. It is *expected* to abort; that is the
  demonstration. (On a 2x2 Hessenberg system the error test gives out first
  and on this one the corrector does -- which of the two goes first is not
  worth relying on.)
* **`-calcic`** discards the initial pressure and calls
  `ComputeConsistentIC()`. Measured: `IDA_LINESEARCH_FAIL`, "Newton/Linesearch
  algorithm failed to converge" -- `IDACalcIC` is documented for index-one
  systems and this is what that means in practice. The option discards the
  pressure first on purpose: a zero return from an already-consistent state
  proves nothing, so the demonstration has to hand it real work.

Also worth recording: my brief for this example specified `+B^T p` on the
momentum row with `-B u` on the constraint row, and called the result
symmetric. It is not -- integrating `(grad p, v)` by parts gives `-(p, div v)`
-- and the sign was corrected against the brief rather than copied from it.

### One thing found that is not ours

`config/defaults.mk` computes `SUNDIALS_CORE_PAT` with a `$(wildcard)` that
is expanded while `defaults.mk` is read -- which is *before* `config/user.mk`
sets `SUNDIALS_DIR`. So a build that sets `SUNDIALS_DIR` in `user.mk` never
appends `-lsundials_core`, the library builds, and the first link of anything
using it fails with `libsundials_core.so.7: DSO missing from command line`.
Pre-existing, affects every SUNDIALS >= 7 user with a `user.mk`, and nothing
to do with IDA; worked around in the build directory here rather than fixed
on this branch, which should stay a clean single-purpose edit.
