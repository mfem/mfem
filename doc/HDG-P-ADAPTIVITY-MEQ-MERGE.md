# Merging `gf-hdg-p-adaptivity` into `meq-integration`

Scratch, like everything in `doc/`. It goes when the merge is done and its
findings are in the code.

`meq-integration` at `/home/ian/projects/mfem/mfem-src` is the only place the
HDG branches meet; they are separate PRs and are never merged into each other.
Its `hdgdev` remote is a **local path** pointing at this tree, so `git fetch
hdgdev` sees unpushed commits — which is a trap, not a convenience: see
step 0.

## The one rule

**Never `git checkout --ours` or `--theirs` on
`fem/darcy/darcyhybridization.cpp`.** It takes the *whole* file. 51 of this
branch's 53 converted call sites in that file merge cleanly, along with all
seven new methods; only two are inside a conflict hunk. The last attempt did
exactly this and silently reverted
`tr_order`, `ess_tdof_user`, `TraceFE`, `TraceVDofs`, `TraceVDofsToTDofs`,
`SetTraceOrders`, `RetireSurplusTraceDofs`, `AllocTraceBlocks` and every
conversion — caught only by counting symbols afterwards. Resolve **hunks**, in
an editor, and count afterwards regardless.

## The conflict surface, measured

Read-only trial merge, which touches no ref, index or working tree:

```
cd /home/ian/projects/mfem/mfem-src
git fetch hdgdev
git merge-tree --write-tree --name-only HEAD hdgdev/gf-hdg-p-adaptivity
```

It prints the merged tree's OID first; `git show <oid>:<path>` then gives the
conflict-marked file without merging anything.

| file | hunks | kind | resolution |
|---|---|---|---|
| `fem/darcy/darcyhybridization.cpp` | 2 (74 + 49 lines) | structural | take **HEAD** in both, then port six sites by hand |
| `miniapps/hdg/anisodiff.cpp` | 1 (8 lines) | both appended | keep both |
| `miniapps/hdg/regression_test.py` | 3 | two appended, one genuine | see below |
| `doc/HDG-P-ADAPTIVITY.md` | 4 | add/add | take this branch's, after checking |

Auto-merged, and each still wants reading rather than trusting:
`fem/darcy/darcyhybridization.hpp`, `miniapps/hdg/convdiff.cpp`,
`tests/unit/CMakeLists.txt`. `fem/darcy/estimators_hdg.{hpp,cpp}` and
`tests/unit/fem/test_darcy_padapt.cpp` do not appear at all — lf never touched
them.

## Why it is a port and not a substitution

Call sites reading the constraint space directly in `darcyhybridization.cpp`:

| | lines | `c_fes.GetFaceElement` | `c_fes.GetFaceVDofs` | total |
|---|---|---|---|---|
| `gf-hdg-dev` (the merge base) | 4229 | 36 | 12 | **48** |
| `gf-hdg-linearise-first`, i.e. `HEAD` | 5215 | 37 | 15 | **52** |
| `gf-hdg-p-adaptivity` | 4529 | 4 | 2 | 6, deliberate |

The old plan's "four extra call sites" was right about the **count** and wrong
about the **work**: lf restructured `ComputeH` into `ComputeElementH` +
`ScatterElementH` + `GetElementTraceSize` with `Hel_offsets` and an OpenMP
chunk loop, so git aligns the trunk's inline Schur body against lf's three
calls. The whole 4-site delta is accounted for exactly:

| function | trunk | lf |
|---|---|---|
| `ComputeH` | 2 | 0 |
| `GetElementTraceSize` | 0 | 1 |
| `ScatterElementH` | 0 | 2 |
| `MarkEmptyTraceRows` | 0 | 1 |
| `NPCReduce` | 0 | 1 |
| `NPCRecover` | 0 | 1 |

Nothing was removed at the function level — lf is additive — so the port is
**six named substitutions in five named functions**, and nothing else.

## The six substitutions

| function | from | to |
|---|---|---|
| `GetElementTraceSize` | `c_fes.GetFaceElement(faces[f])->GetDof()` | `TraceFE(faces[f])->GetDof()` |
| `ScatterElementH` | `c_fes.GetFaceVDofs(faces[f1], c_dofs_1)` | `TraceVDofs(faces[f1], c_dofs_1)` |
| `ScatterElementH` | `c_fes.GetFaceVDofs(faces[f2], c_dofs_2)` | `TraceVDofs(faces[f2], c_dofs_2)` |
| `MarkEmptyTraceRows` | `c_fes.GetFaceVDofs(f, c_dofs)` | `TraceVDofs(f, c_dofs)` |
| `NPCReduce` | `c_fes.GetFaceVDofs(faces[f], c_dofs)` | `TraceVDofs(faces[f], c_dofs)` |
| `NPCRecover` | `c_fes.GetFaceVDofs(faces[f], c_dofs)` | `TraceVDofs(faces[f], c_dofs)` |

**`GetElementTraceSize` and `ScatterElementH` are one edit, not two.** The
first sizes `Hel_offsets` and `Hel_data`; the second writes the scattered rows
using the dof lists. Converting one and not the other gives a buffer sized at
the ceiling against a scatter at the face degree, or the reverse. Do them
together and read the `Hel_offsets` arithmetic once afterwards.

`MarkEmptyTraceRows` is not purely mechanical — see the semantic list below.

## Resolutions, file by file

**`fem/darcy/darcyhybridization.cpp`.** Both hunks are the trunk's inline
`ComputeH` tail against lf's `ComputeElementH(...)` / `ScatterElementH(...)`
calls. Take `HEAD` in both: that code has been *moved*, not deleted, and the
substitution has to be applied where it went. Then make the six edits above.

Exact counts, so the resolution is checkable rather than eyeballed (no line
carries two of these, so `grep -c` is enough):

| | after taking HEAD in both hunks | after the six substitutions |
|---|---|---|
| `c_fes.GetFaceElement` | 5 | **4** |
| `c_fes.GetFaceVDofs` | 7 | **2** |
| `TraceFE(` | 40 | **41** |
| `TraceVDofs(` | 11 | **16** |

The right-hand column is this branch's own 4 and 2 deliberate direct reads, and
its 40 and 13 conversions plus lf's four extra sites less the two that moved
into `ScatterElementH`. If the left column does not match before the edits, the
hunks were resolved the wrong way round.

The four `c_fes.GetFaceElement` and two `c_fes.GetFaceVDofs` that must
*survive* are all in code that has to see the whole slot, not the active part
of it: `RetireSurplusTraceDofs()` reading the full vdof list of a face,
`SetTraceOrders()` taking its ceiling from face 0 and checking each face's
element fits, and the fall-through inside `TraceFE()` and `TraceVDofs()`
themselves. Anything else reaching `c_fes` directly is a missed site.

```
for k in c_fes.GetFaceElement c_fes.GetFaceVDofs 'TraceFE(' 'TraceVDofs('; do
  printf '%-24s %s\n' "$k" \
    $(grep -c "$k" fem/darcy/darcyhybridization.cpp); done
for k in tr_order ess_tdof_user SetTraceOrders RetireSurplusTraceDofs \
         FaceOrdersFromElementOrders AllocTraceBlocks TraceVDofsToTDofs \
         ComputeElementH ScatterElementH GetElementTraceSize Hel_offsets; do
  printf '%-32s %s\n' $k $(grep -c $k fem/darcy/darcyhybridization.cpp); done
```

The second loop is the check that the `--ours` accident has not happened again,
and it has to cover **both** sides: the first seven are this branch's and the
last four are lf's, and a whole-file resolution either way zeroes one group.
Expected on a correct resolution: 10, 3, 8, 2, 2, 3, 4 and 4, 4, 4, 8.

**`miniapps/hdg/anisodiff.cpp`.** One hunk, and it is only the *Sample runs*
header block: subdomains appended two `-p 11` lines and p-adaptivity appended
three `-p 5` ones, after the same base line. Keep both, p-adaptivity's second.

**`miniapps/hdg/regression_test.py`.** Two hunks are appended option reads and
appended command-line clauses — keep both sides in the order they appear. The
third is a real combination: lf nested a `local_nl` check inside the success
branch and p-adaptivity extended the condition with the postprocessed error.
Both are wanted:

```python
if equal(ref_L2_t, test_L2_t) and equal(ref_L2_q, test_L2_q) \
   and (ref_L2_pp is None or equal(ref_L2_pp, test_L2_pp)):
    if ref_local_nl is not None and test_local_nl != ref_local_nl:
        ...FAILING, local nonlinear iterations...
    else:
        ...SUCCESS...
```

`parse_result()` itself does **not** conflict: lf put its local-nonlinear
count in a separate `get_local_nl()` helper, so p-adaptivity's rewritten
parser and lf's helper coexist. Check that both are present afterwards.

**`doc/HDG-P-ADAPTIVITY.md`.** add/add, because the plan was first written on
`gf-hdg-linearise-first` (`af6dab6349`) and re-added on the trunk-based branch
(`96f32f32b7`). Take this branch's copy — but *check* first rather than
asserting it, because whole-file resolutions are the thing that went wrong last
time:

```
git show HEAD:doc/HDG-P-ADAPTIVITY.md > /tmp/meq-plan.md
diff /tmp/meq-plan.md doc/HDG-P-ADAPTIVITY.md     # expect the meq side to be a subset
```

Then take theirs, and add this file alongside it.

## Semantic questions the merge creates

Mechanical resolution is the easy half. Each of these is a real interaction
between per-face trace degrees and something only lf has, none is covered by
any reference, and each is one measurement.

**1. `TraceFE()` inside an OpenMP region, and a lazy cache.**
`FiniteElementCollection::GetFE(geom, p)` — which is all `TraceFE()` is when
degrees are set — reads a `mutable Array<FiniteElementCollection*> var_orders`
and calls `InitVarOrder(p)` when the slot is empty. That is a data race if the
first call for a degree happens on a worker thread, and after the port
`GetCtFaceMatrix`, `GetE/G/HFaceMatrix` all call `TraceFE()` from inside
`ComputeElementH`, which lf runs under `#pragma omp parallel for`. **meq's
build has `MFEM_USE_OPENMP=YES` and `MFEM_THREAD_SAFE=YES`**, so this is
reachable there and nowhere else in the tree.

It appears to be closed already, by accident: `SetTraceOrders()` loops over
every face calling `TraceFE(f)` to verify the degree it got, so every degree in
use is warm before any assembly. That is a guess about a race, which is the
worst kind — measure it (`anisodiff -p 5 ... -hp` in a thread-safe OpenMP
build, several thread counts, answer independent of thread count) and then say
in `SetTraceOrders()` that the verification loop is also the warm-up, so nobody
optimises it away. `GetElementTraceSize` and `ScatterElementH` are both in
serial regions, so those two sites are not part of this.

**2. `MarkEmptyTraceRows()` against the retired surplus.** lf marks trace rows
that no face contributes to and gives them a unit diagonal in the matrix-free
operator. p-adaptivity retires a face's surplus slots into `ess_tdof_list`,
which also gives them a unit row. With `TraceVDofs()` substituted in, a surplus
slot is no longer marked live and so acquires a unit diagonal from *both*
mechanisms. Two identity rows on one dof is probably harmless and that is
exactly the kind of "probably" this branch has been wrong about before. Measure
it: `GradientMode::MatrixFree` with `-pref`, against the assembled answer.

**3. `CanBatchLocalFactor()` under variable order.** It refuses batching when
`Af_f_offsets` or `Df_f_offsets` are non-uniform, which variable-order element
spaces make them. So it should refuse by construction — predicted, not
measured. One run with the batched local factor mode and `-pref`.

**4. Is an unconverted site loud?** The old plan hoped so and never checked.
The local blocks are sized from `TraceFE()`, so a ceiling-length `c_dofs`
against a face-degree block is a size mismatch — an `MFEM_ASSERT` in a debug
build and possibly nothing in release. Cheap experiment while the merge is
open: convert five of the six, leave `NPCRecover` alone, run `convdiff` with
both `--npc` and `--p-refine` (both flags exist once merged, and no reference
combines them), and record what happens. Then convert it.

**5. `HDGFloorStabilization` is not installed on the estimator's integrator.**
On `meq-integration` `anisodiff` builds `amr_bfi` as a bare
`HDGDiffusionIntegrator(kcoeff, td)` while the potential mass form gets
`SetStabilization(*stab)` through its `stabilized()` lambda. So with
`--tau-floor > 0` the estimate measures a different stabilization than the
solve uses. This is the *same defect* the trunk already fixed once inside
`ComputeHDGFaceEnergy()`, which used to form the built-in `{h^-1 Q}` instead of
going through `StabValue()` — the library half is on the trunk and so on this
branch, and the caller half was missed. It is not created by this merge (only
`gf-hdg-subdomains-dev` has `HDGFloorStabilization` at all), but the merge is
when it becomes visible, because p-adaptivity moves the estimator's
construction into the loop and puts the reader right next to it. Install
`stab` on `amr_bfi` there.

And the interaction to write down while doing it:
`HDGErrorEstimator::TraceComparison::Projected` shifts the arguments the
`StabValue()` hook sees, so `-tf` and `--projected-trace-comparison` together
evaluate the hook off its own state. It is already said in
`SetTraceComparison()`'s doxygen; say it in the miniapp's option help too, or
refuse the combination.

**6. `GradientMode::MatrixFree` under a per-face trace** has never been run.
It reaches the trace space through the converted accessors, so it ought to
work, and item 2 is one specific reason it might not.

## Build and verification

The pipeline, which matters because a stale object in the middle of it is the
trap this file family has paid for four times:

```
mfem-src (meq-integration)  ->  /home/ian/projects/mfem/build  (CMake)
                            ->  /home/ian/projects/mfem/install
                            ->  /home/ian/projects/meq/build   (MFEM_DIR)
```

* **`make clean` / a fresh CMake build, not an incremental one.** p-adaptivity
  adds members to `DarcyHybridization` (`tr_order`, `ess_tdof_user`) and to
  `HDGErrorEstimator` (`excl_bdr`, `hyb`, `trcmp`), and `fem/fem.hpp` pulls
  `darcy/darcy.hpp`, so every TU that includes `mfem.hpp` sees both layouts.
  That includes meq's own objects — it holds a `DarcyHybridization` and uses
  its `AssemblyMode`, `LSsolveType` and `LPrecType` enums. meq does **not** use
  `HDGErrorEstimator`, so only the hybridization's layout reaches its code, but
  the rule is the same.
* **`MFEM_ENABLE_TESTING` is now `ON` in `/home/ian/projects/mfem/build`**, and
  the note in `CLAUDE.md` saying it is off is stale. So the unit tests do build
  there, and the merged `tests/unit/CMakeLists.txt` — verified to list all 15
  Darcy test files including `test_darcy_padapt.cpp` — is doing something.
* **`config/user.mk` does not exist in `mfem-src`**, so the recorded
  out-of-source recipe (`cp config/user.mk <scratch>/config/user.mk`) is stale
  too. Pass the TPL settings on the `make config` command line instead;
  `CLAUDE.md` records that command-line `make config VAR=...` was never
  affected by the `defaults.mk`-before-`user.mk` ordering bug.
* **`/home/ian/projects/mfem/build` is `MFEM_USE_MPI=NO`.** So acceptance item
  1's parallel half cannot be taken there; it needs a makefile out-of-source
  build with MPI, which is how both trees here are verified. It is
  `MFEM_USE_SUITESPARSE=YES`, so the 49 skips carry over unchanged — they are a
  property of how the references were generated, not of the build.
* **That install is already stale**, older than `meq-integration`'s current
  tip. meq therefore needs a reinstall and a rebuild whether or not this merge
  happens, which is worth separating from the merge's own deltas.

## Acceptance

Measure the first two **before** merging, so the deltas are the merge's.

1. **Regression references.** Merged serial is 157 files (129 shared + 23 NPC +
   5 p-adaptivity), parallel 121 (p-adaptivity adds no parallel reference).
   Target **2 / 157 with 49 skipped** and **15 / 121** — the same two drift
   failures and the same fifteen environmental ones as both parents.
2. **The NPC references specifically.** All 23 must still pass. Four of the six
   ported sites are in `ComputeH`'s path and so under every hybridized case in
   the suite, but `NPCReduce` and `NPCRecover` are reached only with `--npc`,
   and these 23 files are the whole of that coverage.
3. **Unit tests.** Expect meq's pre-merge count **plus 10 cases and 196
   assertions**, which is what `gf-hdg-p-adaptivity` adds over
   `gf-hdg-dev` (511 / 4,291,129 against 501 / 4,290,933).
4. **The null test, on the merged tree.** The `[PAdapt]` unit cases are the
   bit-for-bit ones -- every face at the uniform order must reproduce the
   never-configured answer with `Normlinf() == 0.0` -- and at miniapp level
   `p1_o2_dg_hb_pref1.txt` and `p1_o2_dg_hb_pref1_pmax.txt` must both still
   discriminate: strip `--p-refine` from either and the suite must get worse.
5. **`extension -o 1 -n 8 -r 1 -no-ctl`** still converges with its geometric
   check at 2.8e-11, which is the subdomain branch's own acceptance item and
   the cheapest proof the merge did not disturb it.
6. **`anisodiff -p 5 -ks 1e2 -o 2 -hb -dg -amr 16 -dorf -hp -pmax 5 -ppest`**
   reproduces the demonstrator's table from the miniapp's header comment.
7. **meq builds and its own suite passes** against the installed library.

## Order of work

0. **Push `gf-hdg-p-adaptivity` to `origin` first.** `hdgdev` is a local-path
   remote, so the merge will happily record commits that exist nowhere else; if
   the branch is later amended or rebased the merge commit dangles.
1. Measure meq's pre-merge baselines (items 1 and 3 above).
2. `git branch backup/meq-before-p-adaptivity` in `mfem-src`. The previous
   attempt was recovered by `git merge --abort`, which is free — but only
   before anything is committed.
3. `git merge hdgdev/gf-hdg-p-adaptivity`, resolve the four files as above.
4. Make the six substitutions. Run the verification greps.
5. Build clean, run acceptance 1–5.
6. Work through the six semantic questions, each as its own measurement, and
   put each answer in the code rather than here.
7. Rebuild and reinstall for meq, rebuild meq, run acceptance 7.
8. Delete this file.

## What not to do

Do not rebase `gf-hdg-p-adaptivity` onto `gf-hdg-linearise-first`. The
branches are separate PRs; a rebase would put 60-odd NPC commits inside
p-adaptivity's diff, and `--rebase-merges` on lf was already measured to
conflict while replaying its four merge commits. Merging in the integration
tree is the only place the two are supposed to meet.
