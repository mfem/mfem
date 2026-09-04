# H(div), RT and broken-RT: everything open, in one place

**All of it is optional and none of it blocks anything here.** This branch's
users want HDG spaces — a discontinuous L2 flux with a `DG_Interface` trace —
and every item below concerns a flux space that is not that: `RT_FECollection`
(H(div), `GetContType() == NORMAL`) or `BrokenRT_FECollection`. Nobody on this
side is asking for any of it.

It is collected here because the `fem/darcy` maintainer may want some of it and
should not have to find it by reading the branch. It is a **menu, not a to-do
list**, and it carries no priorities. Where something is a defect it says so,
where it is a documented refusal it says so, and where nobody knows it says
"unknown".

**These pathways were left alone on purpose.** The standing instruction is *do
not touch the RT or broken-RT pathways when the change is a default change for
the discontinuous spaces*, so where a change would have touched them it was
scoped away instead. Nothing below is an oversight.

**Most of the H(div) surface already works**, and the open items are a thin
edge around it: hybridization takes an H(div) flux and 16 of the 152 serial and
16 of the 121 parallel references exercise it, including a flux-nonlinear one;
the total-flux reconstruction reads a vector-range flux; the classic potential
postprocessing reads both flux layouts (contract in
`fem/darcy/postprocess_hdg.hpp`); threaded assembly is tested on RT; and RT is
what pins the flux-mass boundary-face term, because hybridizing the mixed
method is algebraically exact so the monolithic reference is right by
construction. Why broken-RT exists at all is in the class doxygen at
`fem/darcy/darcyhybridization.hpp:113-127`.

## 1. NPC refuses a flux space that is not discontinuous

`DarcyHybridization::NPCCheck()`, called from `NPCResidual()` and
`NPCGradient()` so both entry points are covered.

**No longer unknown, and the entry that stood here was wrong about what the
obstacle is.** It said the refusal was a refusal to guess about sign
conventions, and that nobody had written the scatter and compared. The scatter
has now been written and compared, and the sign conventions are not the issue:
NPC iterates on the *broken* state and a conforming space is one dof per
interior face too small to hold it, so both elements read the same value and
the trace row is annihilated for **every** conforming state, not just the ones
an iteration visits. The measurement is on `NPCCheck()` and in the `@note` on
the `NPCResidual` group.

**Nothing is left to investigate; what is left is a choice nobody has asked
for.** Lifting the refusal means carrying the flux unknown in the hat space,
which is a different operator size and a different caller contract, not a bug
fix. `BrokenRT_FECollection` reaches an H(div) element under NPC today.

The consequence in the suite is exactly one reference:
`regress_test/p2_o2_hb_nlu_newton.txt` is hybridized, nonlinear in the flux and
H(div), so it is the one `-nlu` case NPC can never take;
`p2_o2_dg_hb_upwind_nlu_newton` is the one it can, and has an `_npc` twin.

**Broken-RT is admitted and works** — settled, measured, and the measurement
now lives in the `-npc` help string of both `convdiff` and `pconvdiff`, which
used to claim the opposite, and in "An H(div) element reaches NPC through a
broken space" in `tests/unit/fem/test_darcy_npc.cpp`.
`BrokenRT_FECollection::GetContType()` returns `DISCONTINUOUS`, and its element
dof counts sum to its space size, which is the property the guard is really
about.

## 2. Broken-RT and the H(div) reduction have no reference coverage

The cheapest items here, and both are a reference set rather than new code.

* **Zero of the 152 serial and 121 parallel references use `--broken-RT`**, and
  the unit coverage is no longer quite zero at the `DarcyForm` level — the NPC
  case above hybridizes a broken-RT flux and checks it against a conforming-RT
  reduced solve — but no *reference* exercises it.
  What exercises it at all: `tests/unit/fem/test_brokenrt.cpp` (four cases, all
  collection-level, no `DarcyForm` anywhere), one solve in
  `tests/unit/fem/test_darcy_reduction.cpp` (`SolveBRT()`), and
  `miniapps/hdg/mfem-logo` through `make test`, which `config/test.mk` checks by
  **exit code alone**. So whether hybridized broken-RT produces the right
  numbers is **unknown** — not suspected wrong, never compared with anything.
  One `-brt` reference, with and without `-npc`, would close it.
* **Zero references use `--reduction` without `--discontinuous`**, so the
  H(div) reduction route is pinned only by `test_darcy_reduction.cpp`'s
  `SolveRT()`. `DarcyPotentialReduction` requires only the *potential* space to
  be discontinuous, which is what lets it take an H(div) flux;
  `DarcyFluxReduction` requires the flux space to be and additionally refuses
  essential flux dofs. Both refuse the nonlinear regime in `ComputeS()`.

A `-chk` tolerance, as `navierstokes-test-seq` uses, turns an exit-code smoke
run into a real check without a reference file.

## 3. The H(div) flux time mass refuses `vdim > 1`

`miniapps/hdg/darcyop.cpp:370` and `:396`, both `MFEM_VERIFY(vdim == 1)` and
both inside `if (btime_u)`. The discontinuous branch above each builds a
`VectorMassIntegrator` and calls `SetVDim(vdim)`; the H(div) branch builds a
`VectorFEMassIntegrator`, which has no `vdim` to set. So a *system* with an
H(div) flux and a time derivative is refused while the same system with a DG
flux runs. **A miniapp limitation, not a library one** — nothing in
`fem/darcy` refuses it. Roadmap §8's.

## 4. The essential-trace guard in seven drivers, and why it is over-broad

`examples/hdg/ex5.cpp:111`, `ex5p.cpp:134`, `miniapps/hdg/convdiff.cpp:345`,
`pconvdiff.cpp:347`, `anisodiff.cpp:272`, `anisodiff-hr.cpp:301`,
`panisodiff.cpp:291` — seven copies of `if (trace_ess_bc && !dg && !brt)`
refusing with "Essential trace BC does not work with continuous elements".

It was measured that the essential-trace route **does** work for RT and gives
the *identical* discrete solution to the natural route. The mechanism: on the
hybridized path `B->AddBdrFaceIntegrator()` supplies a **marker only** —
`DarcyForm::Assemble()` reads `B->GetBFBFI_Marker()` and installs
`constr_flux_integ` on those attributes (`darcyform.cpp:324-341`) — and only
`AssembleDivLDGFaces()`, which the *reduction* branch calls, ever evaluates
`B`'s face integrators. So an RT discretisation can carry a boundary trace
constraint at no cost to itself.

**The guards were left alone deliberately** — they are the branch author's
drivers and upstream's example, moving the miniapps onto the essential-trace
route is not ours to decide, and relaxing the condition would move regression
references. Recorded so the decision is available rather than lost.

## 5. `AssembleFluxMassBdrFaces()` serves a section that is not on this branch

`fem/darcy/darcyform.hpp:175`, definition at `darcyform.cpp:2183`, called from
`:426`. Its doxygen refers the reader to `extension_hdg.hpp`, which is on
`gf-hdg-subdomains-dev` and not here, and the routine exists to serve roadmap
§1's extension-from-subdomains boundary treatment. It is **not**
H(div)-specific — it is space-agnostic and sizes its block correctly for both
layouts. It is here only because its one test is an RT one and because anyone
auditing the RT paths will meet it.

## The adjacent case: an `H1_Trace` (EDG) trace space

**Not H(div), and not our spaces either.** Two items about the *trace* space
rather than the flux space, included because they fail in the same way — on a
space this branch does not use — and a maintainer looking at one will want the
other.

### 6. The trace prolongation is applied inconsistently

Pre-existing and not specific to any nonlinear ordering. Four places disagree
about whether the trace vector is in true dofs or L-dofs:

| what | where | which |
|---|---|---|
| `Operator::Height()` | `fem/hybridization.cpp:33` | `c_fes.GetVSize()`, i.e. L-dofs |
| `ReduceRHS()` | `darcyhybridization.cpp:3337` | sizes the reduced RHS to the *conforming* width |
| serial `Mult()` / `GetGradient()` | `:1829` into `:1971` | indexes `x` by face VDofs with no prolongation |
| `ParMultNL()` | `:2215` | does prolong |

For a `DG_Interface` trace space the conforming prolongation is null and all
four agree, which is every case in this tree. For an `H1_Trace` (EDG) trace
space with a nonlinear problem they would not. Also `HDG-ORDERING-API.md` §7.

### 7. `ProjectSolution()` refuses a continuous trace collection outright

`darcyhybridization.cpp:3523`: `MFEM_VERIFY(c_fes.FEColl()->GetContType() !=
CONTINUOUS, ...)`. `H1_Trace_FECollection` derives from `H1_FECollection`,
whose `GetContType()` returns `CONTINUOUS`, so EDG cannot use that entry point.
Documented by the message; whether the projection could be defined there is
unknown.

## Two negative facts, so nobody looks again

`tests/unit/fem/test_darcy_brokenrt.cpp` **does not exist** — the broken-RT
tests are `test_brokenrt.cpp` and one case in `test_darcy_reduction.cpp`. And
**no `MFEM_VERIFY`, `MFEM_ABORT` or comment anywhere in `fem/darcy/` refuses an
H(div) flux other than `NPCCheck()`**; the other continuity checks in the
directory are the two reductions' constructors and `ProjectSolution`'s, and the
"No flux diagonal!" aborts in `miniapps/hdg/darcyop.cpp` are about the block
preconditioner.
