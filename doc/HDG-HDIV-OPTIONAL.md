# H(div), RT and broken-RT: everything open, in one place

**All of it is optional and none of it blocks anything here.** This branch's
users want HDG spaces — a discontinuous L2 flux with a `DG_Interface` trace —
and every item below concerns a flux space that is not that: `RT_FECollection`
(H(div), `GetContType() == NORMAL`) or `BrokenRT_FECollection`. Nobody on this
side is asking for any of it.

It is collected here because the `fem/darcy` maintainer may want some of it and
should not have to find it by reading the branch. It is a menu, not a to-do
list, and it carries no priorities.

**These pathways were left alone on purpose.** The standing instruction on this
branch is *do not touch the RT or broken-RT pathways when the change is a
default change for the discontinuous spaces*, so where a change would have
touched them it was scoped away instead. Nothing below is an oversight, and
nothing below is a bug report — where something is a defect it says so, where
it is a documented refusal it says so, and where nobody knows it says
"unknown".

Everything here was checked against the code in this working tree. Nothing was
run: the tree was mid-rebuild (`libmfem.a` and the miniapp binaries were
absent), so every claim below is from reading source, not from a measurement,
and the two places where a measurement would settle a question say so.

## The baseline: what already works, so it is not on the list

Worth stating, because most of the H(div) surface is complete and the open
items are a thin edge around it.

* **Hybridization takes an H(div) flux and is exercised.** 16 of the 152 serial
  regression references and 16 of the 121 parallel ones are hybridized with the
  RT default (`--hybridization` present, `--discontinuous` absent), including
  two upwinded ones and `p2_o2_hb_nlu_newton`, which is nonlinear in the flux.
* **The total-flux reconstruction reads a vector-range flux.**
  `fem/darcy/darcyhybridization.cpp:4182` and `:4210` branch on
  `FiniteElement::VECTOR` and call `CalcVShape`; the reconstructed total flux
  is an RT space by construction (`fem/darcy/darcyform.cpp:1030`).
* **The classic potential postprocessing reads both flux layouts.**
  `fem/darcy/postprocess_hdg.cpp:38-50` checks `vdim` against the range type
  rather than assuming one, and the contract is in the class doxygen at
  `fem/darcy/postprocess_hdg.hpp:41-45`.
* **Threaded assembly is tested on RT** —
  `tests/unit/fem/test_darcy_threaded_assembly.cpp:80`.
* **A boundary face term on the flux mass reaches the hybridized solve, and RT
  is what pins it** — `tests/unit/fem/test_darcy_hybridization.cpp:539`, whose
  own comment says RT is the right form to test it in because hybridizing the
  mixed method is algebraically exact, so the monolithic reference is right by
  construction rather than by measurement.

The background for why broken-RT exists at all is the class doxygen at
`fem/darcy/darcyhybridization.hpp:113-127`: continuity of the flux space does
not allow stabilisation of the trace for advection in the potential equation,
so the flux space is broken to permit upwinding.

## 1. NPC refuses a flux space that is not discontinuous

`DarcyHybridization::NPCCheck()`, `fem/darcy/darcyhybridization.cpp:3523-3536`.
The guard is at `:3526-3530`:

    MFEM_VERIFY(fes.FEColl()->GetContType() ==
                FiniteElementCollection::DISCONTINUOUS,
                "NPC needs a discontinuous flux space; ...");

`fes` is the flux space (`Hybridization(fes_u_, fes_c_)`,
`darcyhybridization.cpp:28-33`). It is called from `NPCResidual()` (`:3542`)
and `NPCGradient()` (`:3633`), so both entry points are covered. The reasoning
is on the declaration (`darcyhybridization.hpp:678-681`) and in the `@note` on
the `NPCResidual` group (`darcyhybridization.hpp:1018-1022`): an H(div) flux
makes the local rows of the residual a conforming scatter with sign conventions
that have not been checked against.

**Unknown**: whether those sign conventions actually differ. The refusal is a
refusal to guess, not a finding that it is wrong. Nobody has written the
scatter and compared it.

The consequence in the suite is exactly one reference:
`miniapps/hdg/regress_test/p2_o2_hb_nlu_newton.txt` is hybridized, nonlinear in
the flux, and H(div) (`--no-discontinuous`, `--no-broken-RT`,
`--hybridization`), so it is the one `-nlu` case NPC can never take;
`p2_o2_dg_hb_upwind_nlu_newton` is the one it can, and has an `_npc` twin.

### 1a. The guard admits broken-RT; the miniapps' help text says it does not

`BrokenRT_FECollection::GetContType()` returns `DISCONTINUOUS`
(`fem/fe_coll.hpp:502`). So `-brt` passes `NPCCheck()`. But the `-npc` option
help says otherwise in both drivers — `miniapps/hdg/convdiff.cpp:276`
("needs a DISCONTINUOUS flux, so not with `-brt` or the H(div) default") and
`miniapps/hdg/pconvdiff.cpp:272-273`. Neither miniapp guards the combination
itself: `convdiff.cpp:1036` calls `op.SetNPC()` unconditionally and the `cerr`
validation block at `:344-380` has no NPC case, so a genuinely refused
combination aborts out of the library rather than being turned away by option
parsing.

Two things followed, and they were separate. **Both are now settled, and the
guard was right while the help text was wrong.**

The cheap check this entry proposed was run — `convdiff -no-vis -nx 8 -ny 8
-p 2 -o 1 -brt -hb -nl -nls 3`, with and without `-npc`:

| route | `|| q_h - q_ex || / || q_ex ||` | local nonlinear iterations |
|---|---|---|
| reduced trace operator | 0.00229857 | 256 |
| NPC | 0.00229857 | **0** |

Identical to every printed digit, and the iteration count says NPC really took
the other route rather than falling back. So **NPC works on a broken-RT flux**,
which is what the guard's own reasoning predicts: `NPCCheck()` refuses a
*conforming scatter*, and broken RT has no inter-element continuity, so it is
not one. The help text in both miniapps has been corrected and now carries this
measurement.

What is still true is the coverage: **no regression reference uses
`--broken-RT` at all** (0 of 152 serial, 0 of 121 parallel), so nothing in the
suite would notice if this stopped working. One `-brt` reference, with and
without `-npc`, would close that — it is the cheapest item in this document and
the only one with a measurement already behind it.

## 2. The H(div) flux time mass refuses `vdim > 1`

`miniapps/hdg/darcyop.cpp:370` and `:396`, both `MFEM_VERIFY(vdim == 1,
"Unsupported case")` and both inside `if (btime_u)`. The discontinuous branch
immediately above each one builds a `VectorMassIntegrator` and calls
`SetVDim(vdim)` (`:364` and `:391`); the H(div) branch builds a
`VectorFEMassIntegrator`, which has no `vdim` to set. So a *system* — more than
one equation — with an H(div) flux and a time derivative is refused, while the
same system with a DG flux runs.

This is roadmap §8's, cross-referenced from §4. It is a miniapp limitation
rather than a library one: nothing in `fem/darcy` refuses it.

## 3. The essential-trace guard in seven drivers, and why it is over-broad

Seven copies of the same three lines:

| file | line |
|---|---|
| `examples/hdg/ex5.cpp` | 111 |
| `examples/hdg/ex5p.cpp` | 134 |
| `miniapps/hdg/convdiff.cpp` | 345 |
| `miniapps/hdg/pconvdiff.cpp` | 347 |
| `miniapps/hdg/anisodiff.cpp` | 272 |
| `miniapps/hdg/anisodiff-hr.cpp` | 301 |
| `miniapps/hdg/panisodiff.cpp` | 291 |

each reading `if (trace_ess_bc && !dg && !brt)` and refusing with "Essential
trace BC does not work with continuous elements".

It was measured on this branch that the essential-trace route **does** work for
RT, and gives the identical discrete solution to the natural route. The
mechanism: on the hybridized path `B->AddBdrFaceIntegrator()` supplies a
*marker only* — `DarcyForm::Assemble()` reads `B->GetBFBFI_Marker()` and
installs `constr_flux_integ` on those attributes
(`fem/darcy/darcyform.cpp:324-341`) — and only `AssembleDivLDGFaces()`, which
the *reduction* branch calls, ever evaluates `B`'s face integrators. So an RT
discretisation can carry a boundary trace constraint at no cost to itself.

The guards were left alone deliberately: they are the branch author's drivers
and upstream's example, moving the miniapps onto the essential-trace route is
not ours to decide, and relaxing the condition would move regression
references. Recorded here so the decision is available rather than lost.

## 4. Broken-RT has almost no coverage, and the collection is branch-local

`BrokenRT_FECollection` (`fem/fe_coll.hpp:492-508`) does not exist on `master`
— it is new in this branch family. What exercises it:

* `tests/unit/fem/test_brokenrt.cpp`, four cases, all **collection-level**: it
  is RT with nothing shared (`:31`), the name round-trips (`:80`), it supplies
  a trace collection with the same face dof count as RT's (`:117`), and it
  registers everything on the element (`:144`). No `DarcyForm` anywhere in the
  file.
* One solve: `tests/unit/fem/test_darcy_reduction.cpp`, `SolveBRT()` at `:126`
  and the case at `:217`, which checks flux reduction against the monolithic
  broken-RT solve.
* `miniapps/hdg/mfem-logo`, through the `make test` target at
  `miniapps/hdg/makefile:83-85` (`-nx 50 -o 3 -hb -brt -a 1e3 -ks 1e-3
  -c 2e4`). That is the only *hybridized* broken-RT run anywhere in the
  automated tests, and `config/test.mk:62-71` checks it by **exit code alone**
  — there is no reference to compare against, so it catches a crash and
  nothing else.

Zero of the 152 serial and 121 parallel regression references use
`--broken-RT`. So: whether hybridized broken-RT produces the right numbers is
**unknown** — not suspected wrong, simply never compared with anything. The
cheapest repair is a reference set, not new code; the second cheapest is the
shape `navierstokes-test-seq` already uses at `miniapps/hdg/makefile:80-82`,
where a `-chk` tolerance turns an exit-code smoke run into a real check without
a reference file.

## 5. Potential reduction is the H(div) reduction, and is uncovered

`DarcyPotentialReduction` requires only the *potential* space to be
discontinuous (`fem/darcy/darcyreduction.cpp:972-974`), which is what lets it
take an H(div) flux; `DarcyFluxReduction` requires the *flux* space to be
discontinuous (`:359-361`) and additionally refuses essential flux dofs
(`:370-371`). The drivers encode the choice at
`miniapps/hdg/convdiff.cpp:870-882`: `dg || brt` takes flux reduction,
otherwise potential reduction and only when there is no convection and there
is a time term, otherwise "No possible reduction!".

Both reductions refuse the nonlinear regime in `ComputeS()`
(`darcyreduction.cpp:533-534` and `:1138-1139`).

Coverage: every `--reduction` regression reference is a `-dg` one — 0 of 152
serial and 0 of 121 parallel have `--reduction` without `--discontinuous`. The
only pin on the H(div) route is `tests/unit/fem/test_darcy_reduction.cpp`,
`SolveRT()` at `:62` and the case at `:188`.

## 6. A capability statement worth keeping, with its scope corrected

`doc/HDG-ORDERING-API.md` §6 (`:541-542` as this was written; that file is
under active edit, so trust the section) says the reduced trace operator "is
the only route that accepts an **H(div) flux**". Within that section, which
compares the two *nonlinear* routes, that is exactly right: NPC refuses it
(item 1) and potential reduction refuses nonlinear integrators (item 5), so
hybridization is the only one left. Read as an absolute statement about every
route it is not — potential reduction and the unreduced monolithic solve both
take an H(div) flux for a linear problem, and
`tests/unit/fem/test_darcy_reduction.cpp:62` runs both.

Worth restating precisely if that sentence ever migrates into doxygen.

## 7. `AssembleFluxMassBdrFaces()` serves a section that is not on this branch

`fem/darcy/darcyform.hpp:161-175` (declaration at `:175`; roadmap §1 still
cites `:174`), definition at `fem/darcy/darcyform.cpp:2183`, called from
`:426`. Its doxygen refers the reader to `extension_hdg.hpp`, which is on
`gf-hdg-subdomains-dev` and not here, and the routine exists to serve roadmap
§1's extension-from-subdomains boundary treatment.

It is **not** H(div)-specific — it is space-agnostic, and its
`MFEM_VERIFY` at `darcyform.cpp:2234-2237` sizes the returned block as
`fe1->GetDof() * fes_u->GetVDim()`, which is right for both layouts. It is here
only because its one test is an RT one (item in the baseline above) and because
anyone auditing the RT paths will meet it.

## The adjacent case: an `H1_Trace` (EDG) trace space

**Not H(div), and not our spaces either.** Two items about the *trace* space
rather than the flux space, included because they fail in the same way — on
a space this branch does not use — and a maintainer looking at one will want
the other.

### 8. The trace prolongation is applied inconsistently

Pre-existing and not specific to any nonlinear ordering. Four places disagree
about whether the trace vector is in true dofs or L-dofs:

| what | where | which |
|---|---|---|
| `Operator::Height()` | `fem/hybridization.cpp:33` | `c_fes.GetVSize()`, i.e. L-dofs |
| `ReduceRHS()` | `fem/darcy/darcyhybridization.cpp:3320-3345` | sizes the reduced RHS to the *conforming* width |
| serial `Mult()` / `GetGradient()` | `:1829` into `:1976-1981` | indexes `x` by face VDofs with no prolongation |
| `ParMultNL()` | `:2221-2238` | does prolong |

For a `DG_Interface` trace space the conforming prolongation is null and all
four agree, which is every case in this tree. For an `H1_Trace` (EDG) trace
space with a nonlinear problem they would not.

`doc/HDG-ORDERING-API.md` §7 item 10 (`:698` as this was written) carries this,
and its own line numbers are stale — it cites `cpp:3469-3494` for
`ReduceRHS()`, which is now at `:3293`, and `:3474` is `ProjectSolution()`.

### 9. `ProjectSolution()` refuses a continuous trace collection outright

`fem/darcy/darcyhybridization.cpp:3474-3479`:
`MFEM_VERIFY(c_fes.FEColl()->GetContType() != CONTINUOUS, "Continuous trace
collections are not supported in projection!")`. `H1_Trace_FECollection` derives
from `H1_FECollection` (`fem/fe_coll.hpp:354`), whose `GetContType()` returns
`CONTINUOUS` (`:313`), so EDG cannot use that entry point. Documented by the
message; whether the projection could be defined there is unknown.

## Claimed items that turned out not to exist

Checked and not found, recorded so nobody looks again:

* **`tests/unit/fem/test_darcy_brokenrt.cpp`** does not exist. The broken-RT
  tests are `tests/unit/fem/test_brokenrt.cpp` (collection-level) and one case
  in `tests/unit/fem/test_darcy_reduction.cpp` (item 4).
* **No `MFEM_VERIFY`, `MFEM_ABORT` or comment anywhere in `fem/darcy/` refuses
  an H(div) flux other than `NPCCheck()`.** The other continuity checks in the
  directory are `DarcyFluxReduction`'s and `DarcyPotentialReduction`'s
  constructors (item 5) and `ProjectSolution`'s (item 9);
  `darcyhybridization.cpp:1128`'s `MFEM_ABORT("TODO: algebraic definition of
  C")` and `:3917`'s non-conforming-mesh assert are unrelated to the flux
  space, as are the "No flux diagonal!" aborts in `miniapps/hdg/darcyop.cpp`
  (`:985`, `:1133`, `:1152`), which are about the block preconditioner.
* **`GradientMode::MatrixFree` and `LocalOpType::FluxNL` are no longer
  refused.** Both refusals were lifted in `469d7115aa`; the comment recording
  that is at `darcyhybridization.cpp:3531-3535`. They were never H(div)
  refusals in any case.
