# What of the device work belongs upstream, and how it has to be shaped

Scratch, like every `.md` here. This one is about a CONSTRAINT rather than a
task: everything the HDG branches touch outside `fem/darcy/` and
`miniapps/hdg/` has to become its own PR off `origin/master` eventually, and a
PR that cannot be cherry-picked cleanly is a PR nobody merges.

**The mechanical rule, and it costs nothing if applied from the start: an
out-of-darcy change goes in its own COMMIT that touches no `fem/darcy` file.**
Then `git cherry-pick <sha>` onto a branch off master is clean and the PR is
the commit. A commit that mixes `fem/bilininteg.hpp` with
`fem/darcy/darcyhybridization.cpp` has to be split by hand later, against a
master that has moved.

## The design question, and the measurement that settles it

Asked: since the divergence block will need `VectorDivergenceIntegrator`
extended anyway, should `VectorMassIntegrator` get an `AssembleEA` too, and
should the Darcy kernels be replaced by MFEM's `AssemblyLevel::ELEMENT`?

**No, and the reason is not taste.** MFEM's element assembly is
**tensor-product only, and a simplex is refused on both of its paths.**
`MassIntegrator::AssembleEA` goes through `AssemblePA`, which asks for either
`DofToQuad::TENSOR` or, on the Stroud path, `RAGGED_TENSOR`
(`fem/integ/bilininteg_mass_pa.cpp:57-64`). The first is serviced only by
`NodalTensorFiniteElement`; the generic `FiniteElement::GetDofToQuad` is
`MFEM_VERIFY(mode == DofToQuad::FULL, "invalid mode requested")`
(`fem/fe/fe_base.cpp:377`) — an abort, not a fallback. The second is refused
by `AssembleEA_` itself: `MFEM_VERIFY(maps->mode != DofToQuad::RAGGED_TENSOR,
"AssembleEA not implemented for ragged tensor bases")`
(`fem/integ/bilininteg_mass_ea.cpp:27`). So quads and hexes only, and that is
checked in both places rather than inferred from the kernel's indexing.

The HDG work is not tensor-only. Seven of the Darcy unit-test files build
triangle meshes, and the Darcy-local element-mass kernel runs on them:
`convdiff -m ../../data/inline-tri.mesh -p 1 -o 2 -dg -hb -rtol 1e-12` reports
the flux mass kernel taken and returns 1.14145e-05 / 2.79531e-07, identical to
the element loop's. Routing that through upstream EA would LOSE every simplex
case rather than accelerate it.

Two further mismatches, either of which would be enough on its own:

* **EA has no notion of `vdim`.** `EABilinearFormExtension` sizes `ea_data` as
  `ne*ndof*ndof` with a scalar `ndof`, so the flux space — L2 with
  `vdim = dim` — cannot go through it at all.
* **EA folds FACE terms into the element matrices for a DG space**
  (`factorize_face_terms` is set whenever `fes->IsDGSpace()`), which is exactly
  the work `DarcyForm` routes into the constraint blocks itself. It would
  double-count.

**So the divergence block does not need `VectorDivergenceIntegrator` extended
either — and it is written now**, as a Darcy-local kernel of the same shape as
the two mass ones: a host pass computing `(NQ, NE)` weights and shape tables,
then one `mfem::forall` over elements. Its extra ingredient is the flux basis
DIVERGENCE at quadrature points, and that turned out to be free: the divergence
of the vector basis function `(a, k)` is `d_k phi_a`, so the physical gradient
table IS the divergence table — which is what `DenseMatrix::GradToDiv()` says
by copying it verbatim.

For that block the argument is stronger than the simplex one, because there is
no upstream route on ANY element shape: `MixedBilinearForm::SetAssemblyLevel`
aborts on `AssemblyLevel::ELEMENT`, the two-space `AssembleEA` virtual is
commented out, and `VectorDivergenceIntegrator` has no EA. Nothing to fall
back to and nothing to wait for.

That is the whole answer to the question as asked: the mixed-EA edifice
(`EAMixedBilinearFormExtension`, the commented-out
`AssembleEA(trial_fes, test_fes, emat)` virtual at `fem/bilininteg.hpp:105`,
`MixedBilinearForm::SetAssemblyLevel`'s "stay tuned" abort) is not on the
critical path for HDG and should not be built for HDG's sake.

## What DOES belong upstream, and it is one small PR

Everything currently outside `fem/darcy` is additive and behaviour-preserving.
It is "expose what is already computed", and it exists because a batched
assembly has to ask an integrator what it would do rather than reconstruct it:

| addition | why |
|---|---|
| `SumIntegrator::NumIntegrators()`, `GetIntegrator(i)` | a wrapper that hides what it wraps forces every consumer to treat a sum of one as different from the thing itself — which made a whole assembly mode unreachable here |
| `DGTraceIntegrator::GetVelocity/GetDensity/GetAlpha/GetBeta` | the face kernels need the same `alpha`, `beta` and `u` the integrator uses |
| `VectorMassIntegrator::GetCoefficient/GetVectorCoefficient/GetMatrixCoefficient` | ditto for the element kernel |
| `MassIntegrator::GetElementIntRule`, `VectorMassIntegrator::GetElementIntRule` | ONE SOURCE OF TRUTH for the quadrature rule: `AssembleElementMatrix()` now calls it, so a kernel that wants the same operator asks instead of copying a formula with three inputs |

The rule methods are the only ones with a behavioural surface at all, and it is
a narrowing: the formula moves from inside `AssembleElementMatrix` to a method
that `AssembleElementMatrix` calls. Nothing else changes.

**Not yet split into its own commit.** Doing that is the next housekeeping step
and it is cheap now, expensive later.

## What upstream is already doing, checked rather than assumed

Two of these change the plan, so both were verified directly against
`mfem/mfem` rather than taken from a summary.

**`vdim` in element assembly is IN REVIEW and we must not write it.**
PR #5419, "Add vector-valued element assembly", Alex Lindsay, open, +682/-117,
last touched 2026-09-02. It puts a `vdim` on `EABilinearFormExtension` and
sizes `ea_data` as `ne*vdim*elemDofs*vdim*elemDofs` — the exact lines the first
draft of this note proposed writing. It also adds `VectorDiffusionIntegrator`'s
EA kernels, so `VectorMassIntegrator`'s would be a precedented follow-on rather
than a novelty.

**And it excludes our case by name.** Read from the PR's own diff:

```cpp
if (vdim > 1 && trial_fes->IsDGSpace())
{
   MFEM_VERIFY(a->GetBBFI()->Size() == 0 && a->GetFBFI()->Size() == 0 &&
               a->GetBFBFI()->Size() == 0,
               "vector-valued DG element assembly does not yet support "
               "boundary or face integrators");
}
```

A vector-valued DG space with face integrators is the HDG flux mass, refused
explicitly. So even after #5419 lands, upstream EA does not serve this branch,
on tensor meshes or otherwise. If we want to engage it, the useful contribution
is the half that guard excludes, offered *onto* #5419.

**Mixed element assembly is genuinely empty, and has been for six years.**
Verified on `origin/master` today: `MixedBilinearForm::SetAssemblyLevel` still
carries `MFEM_ABORT("Element assembly not supported yet... stay tuned!")` with
`// ext.reset(new EAMixedBilinearFormExtension(this));` beneath it
(`fem/bilinearform.cpp:1415`), and the mixed `AssembleEA(trial_fes, test_fes,
emat)` virtual is still commented out (`fem/bilininteg.hpp:105`). The "stay
tuned" traces to the original element-assembly PR in 2020 and was never
followed up.

It is still not worth building for HDG's sake, for the simplex reason above.
It is worth knowing it is unclaimed if anyone wants it on its own merits.

## The sequencing constraint, which is sharper than the cherry-pick one

**`fem/darcy/` itself is being upstreamed right now, by someone else.**
PR #5384, "Framework for mixed systems - phase 1 [darcy-hdg-phase1]", Jan Nikl,
open, **+26,602 across 68 files, last updated 2026-09-07** — the day this was
written. It carries `fem/darcy/{bilininteg_hdg,darcyform,darcyhybridization,
darcyreduction,estimators_hdg,pdarcyform}`, the `miniapps/hdg` codes, and eight
new `examples/hdg/`. Its parent, PR #4350, is where the `HDGStabilization` and
ordering conversations already happen.

Upstream's `fem/darcy/` is entirely host code — no `mfem::forall`, no
`Read()`/`Write()`, no `BatchedLinAlg` anywhere in it. So **nobody is doing HDG
on device and that half is ours**, but our device work sits on top of 26k lines
in review. Landing device changes into `fem/darcy/` upstream before phase 1
merges would conflict with it directly.

The sequencing that follows: let phase 1 land, then offer device support as
phase-2 material, and raise it in #4350 where the thread is already open,
rather than as a surprise PR.

## The cleanest contribution we have, and it is not the kernels

The two `linalg/batched/` defects. **Verified against `origin/master` today,
both still live and neither ever reported:**

| | master | ours |
|---|---|---|
| `GPUBlasBatchedLinAlg::AddMult` | `&alpha, d_A, m, m*n, ...` | `&alpha, d_A, lda, m*n, ...` |
| `NativeBatchedLinAlg::LUSolve` | `Reshape(x.Write(), ...)` | `Reshape(x.ReadWrite(), ...)` |

Both are silent wrong answers on a device — a rectangular batch returns garbage
from the first, and the second solves against whatever was in device memory.
Both were found from the Darcy side and neither is reachable from anything else
in the tree, which is why they survived. `MagmaBatchedLinAlg::AddMult` carries
the same `lda` defect and is untested here, MAGMA not being installed; an
outside contributor landed a batched fix on exactly those terms recently, so
that is an acceptable way to offer it rather than a reason to hold it back.

This is already on `batched-linalg-device-fixes` off `master` with its own
test, so it is PR-shaped today.

## Two process facts worth having written down

* **MFEM has an AI policy as of v4.10**, and it is the first line of that
  release's CHANGELOG: AI-assisted code is allowed but must be disclosed with
  an `AI-assisted` label on the PR, and the author remains responsible for
  correctness, licensing and attribution. It applies to everything here.
* **`CONTRIBUTING.md` prefers feature branches inside the MFEM organisation
  rather than a fork**, which is what `gf-hdg-*` already are. There is no PA/EA
  kernel-authoring guide — the GPU section is about a dozen lines of pointers —
  so a kernel PR carries its own explanation of layout and contract or it
  carries nothing.

## Already upstream-shaped, for reference

`batched-linalg-device-fixes` is the model: off `master`, touching only
`linalg/batched/`, carrying its own test, and merged INTO the HDG branches
rather than the other way round. Any future upstream piece should be built the
same way from the start.
