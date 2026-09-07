# Device offload for the element-local HDG work — the plan

Scratch, like every `.md` here. `doc/HDG-ELEMENT-LOCAL-PARALLELISM.md` is the
host-threading side of this and is **done**: every element-local loop in
`DarcyHybridization` threads, bit-for-bit, and the measurements there are what
bound everything below. This file is the device plan and nothing in it is
built.

## THE TARGET, and it is stricter than "offload the hot groups"

**A full-device code path, with no transfer back to the host until the outer
driver needs to write output.** Stated by the caller, and the first kernel's
measurement is the argument for it rather than against.

`HDGDiffusionFaceMatricesBatched()` (step 2's first kernel, built and correct)
computes every interior face matrix on the device. Its consumer is host code,
so every matrix comes straight back. Steady state, 2-D quads:

| n=64 order 3 | precompute (host) | kernel | copy-back |
|---|---|---|---|
| `-d cpu` | 16.8 ms | 26.3 ms | 0.0 ms |
| `-d cuda` | 16.9 ms | 32.7 ms | **10.9 ms** |

and at n=96 order 3 the copy-back is **183 ms against an 85 ms kernel**, being
193 MB of dense face matrices. So an isolated device kernel whose consumer is
on the host pays more in transfer than it saves in arithmetic. That is the
gate below, arriving from the integrator side instead of the linear-algebra
side, and it generalises the gate: **no step of this plan can be landed alone
and show a gain. The whole chain -- face assembly, the E/G/H/D storage, the
local factorisation and solves, the trace solve -- has to be device-resident
together.**

Two consequences worth stating now:

* **Step 4 stops being a configuration choice and becomes a requirement.**
  UMFPACK and KLU are host-only, so a direct trace solve forces exactly the
  transfer this target forbids. The trace solve has to be Krylov +
  cuSPARSE/AMG, or the chain is broken at its end.
* **Step 1's storage change and step 2's kernels have to land together.**
  Step 1 alone leaves the integrators on the host; step 2 alone leaves the
  local blocks there. Either alone is the "worse than doing nothing" case.

GPU timings here are from a consumer card shared with a desktop under WSL2 and
are indicative of shape, not of achievable performance.

### The scatter kernel, which is the shape the target needs

`HDGDiffusionFaceScatterBatched()` writes E, G, H and D straight into the
hybridization's own storage instead of into dense per-face matrices. **E, G
and H come out bit-exact against the per-face integrator; D to 8.7e-17**,
which is the atomic accumulation order. 2-D quads, n=64 order 3:

| | per-face (host) | dense kernel + copy-back | scatter, no readback |
|---|---|---|---|
| `-d cpu` | 38.0 ms | 26.3 + 0.0 | 60.5 ms |
| `-d cuda` | 41.6 ms | 32.7 + 10.9 | **43.5 ms** |

Two things to read off it. **The copy-back is gone by construction**, not
hidden -- the blocks are already where the local solves want them, which is
what a full-device chain needs. And **the atomics make this a device path and
not a host one**: D accumulates per element and two faces of an element
collide, so it goes through `AtomicAdd`, which on the host costs where a plain
`+=` does not (60.5 against 38.0). A host build wanting this shape should take
the colouring `DarcyHybridization` already builds.

Parity rather than a win on this hardware, which is a consumer card shared
with a desktop under WSL2 and not a verdict.

**Neither kernel is wired in.** Switching either on means the consumers --
`ComputeAndAssemblePotFaceMatrix`, then the local factorisation and solves --
moving with it, per the target above.

## The gate, before any of the steps

**Two of the four groups are nearly free and doing only those is worse than
doing nothing.** Groups 1 and 4 leave the integrators on the host, so every
iteration would copy the local blocks host↔device around host-side integrator
work — plausibly slower than staying on the host throughout, which is where
this already runs well. **The device story for HDG static condensation is worth
starting only if group 2 is going to be finished.** Group 1 first is for
proving the harness, not for the speedup.

## The four groups, with the shares that bound them

Measured on the host, from the threading work. Shares of an **NPC step**
(pedestal, `(n,k)` = (32,1), (48,2), (64,2), (32,3)) and of a **linear solve**
(order 2 quads, n = 96):

| group | what | NPC step | linear solve | what it needs |
|---|---|---|---|---|
| 1 | local dense LA — `InvertA`/`InvertD`, `MultInv`, `NPCReduce`/`NPCRecover`, `ComputeElementH`'s factor+Schur | 7–10% | ~6% | **a storage change, no kernels** |
| 2 | the integrators — `ConstructGrad`, `LocalResidual`, `DarcyForm::Assemble` | 46–53% | 44% | a partial-assembly rewrite |
| 3 | the scatter into the `SparseMatrix` | 12–17% | ~13% | a different algorithm, or `MatrixFree` |
| 4 | the trace solve | 26–31% | 51% | a configuration change |

## What you write, and what you do not

**You do not write CUDA, HIP or SYCL.** `mfem::forall(N, [=] MFEM_HOST_DEVICE
(int i) {...})` is the portability layer — one lambda, compiled for whatever
`mfem::Device` is configured. For the block-per-element shape a PA kernel wants
there are `forall_2D`, `forall_3D`, `forall_2D_batch`
(`general/forall.hpp:1226-1256`) and the `MFEM_FORALL_2D/3D/3D_GRID` macros,
with `MFEM_FOREACH_THREAD`, `MFEM_SHARED`, `MFEM_SYNC_THREAD` and
`MFEM_UNROLL` inside them; on the host those degrade to plain loops and empty
macros (`general/backends.hpp:71-77`), so **the same source is the CPU kernel
and the GPU kernel**. `kernels::` supplies `MFEM_HOST_DEVICE` dense linear
algebra, `MFEM_REGISTER_KERNELS` handles (dim, order) dispatch.

**Not Kokkos.** MFEM's backends are `CPU`, `OMP`, `CUDA`, `HIP`,
`RAJA_{CPU,OMP,CUDA,HIP}`, `OCCA_{CPU,OMP,CUDA}` and `CEED_{CPU,CUDA,HIP}`, so
RAJA and OCCA are reachable *through* MFEM and libCEED exists for exactly the
operator-evaluation problem group 2 poses. A second programming model inside a
library that has one would not be accepted upstream.

**There is no SYCL backend**, so an Intel GPU goes through OCCA or libCEED or
not at all.

## Step 0 — a CALLER CONTRACT, and the library is not at fault

**Withdrawn: an earlier version of this section said "configuring a CUDA
Device silently breaks the existing host code" and that finding the sites was
"mechanical but not small". That was wrong on both counts.** The symptom was
real and is reproduced below; the attribution was not, and the fix is one line
in the CALLER.

The symptom, on `-d cuda` and identically on `-d debug`: the reduced trace
right-hand side comes back **exactly zero**, the trace solve returns zero, the
recovered fields are quietly wrong, and nothing errors.

The cause is the MFEM alias-memory idiom, not `fem/darcy`. Writing a block of
a `BlockVector` through the block —

```cpp
rhs.GetBlock(1) += *darcy.GetPotentialRHS();   // device op on an ALIAS
```

— leaves the result in that alias's device buffer. A second view over the same
range (`BlockVector drhs(rhs, darcy.GetOffsets())`, which is what a caller
hands `FormLinearSystem`) gets a **fresh alias marked host-valid** whatever the
underlying state, so the host loops read stale zeros. `DarcyForm::Assemble()`
already does the right thing for its own `b_u` / `b_p` — `b_p->Assemble();
b_p->SyncAliasMemory(*block_b);` — and the caller owes the same:

```cpp
rhs.GetBlock(1) += *darcy.GetPotentialRHS();
rhs.GetBlock(1).SyncAliasMemory(rhs);          // <-- this
```

With that one line, `-d cpu` and `-d debug` agree **bit for bit, end to end**.

**Confirmed on the real GPU too, and the residue is worth reading carefully.**
Against the CUDA build, every FIELD value is bit-identical -- the trace
solution, the flux, the potential and each printed `u[i]`. Two printed numbers
still differ, in the last one or two hex digits: `|rhs_p|` and `|RHS|`. Both
are `Norml2()` results, and a norm is a reduction whose association order
differs on a device. So the data agrees exactly and only the *summation of it*
does not, which is the opposite of what the zero was and is not something to
chase.

Three things worth keeping from chasing it:

* **`-d debug` reproduces it exactly and needs no GPU.** MFEM's debug backend
  runs the device memory-validity machinery on the CPU, and it gave the same
  zeros with otherwise bit-identical arithmetic. So device-correctness work on
  this branch can be done and tested on any machine, in a serial build, with
  two-minute rebuild cycles instead of ten-minute CUDA ones. **Reach for it
  before configuring anything.**
* **The validity flags LIE, so no guard can catch this.** Measured at the
  entry to `ReduceRHS()` in the failing case: `hostvalid=1 devvalid=0` on the
  very block whose host buffer is zeros. `HostRead()` is therefore correctly a
  no-op, and an `MFEM_VERIFY` on `HostIsValid()` would pass. Three attempts to
  repair it from inside `DarcyHybridization` failed for this reason and are
  not in the tree.
* **There is a device test harness already**, contrary to what the first pass
  here assumed: `tests/unit/gpu_unit_test_main.cpp` builds `gpu_unit_tests`
  with `Device device("gpu")` when `USE_GPU` is set, and
  `tests/unit/miniapps/test_debug_device.cpp` is the precedent for a
  `Device`-constructing case. That is where a device regression for this
  belongs.

## Step 1 — group 1, and it is a storage change

`BatchedLinAlg` already wraps every operation these loops need
(`LUFactor`/`LUSolve`/`Mult`/`MultTranspose`/`AddMult`/`Invert`) and its NATIVE
backend *is* an `mfem::forall` over `MFEM_HOST_DEVICE` lambdas with
`Read()`/`Write()` discipline, with `GPU_BLAS` (cuBLAS/hipBLAS) and `MAGMA`
beside it. `SetLocalFactorMode(Batched)` already routes `InvertA`/`InvertD`
through it.

**What blocks the device is one line.** `InvertA()` builds
`DenseTensor A(Af_data.GetData(), n, n, NE)` — the raw-pointer constructor,
which goes to `Memory::Wrap()` and sets `VALID_HOST` with no device type. So
the batched path is device-ready as an algorithm and host-bound at the call
site.

Two ways, and the second is smaller:

1. Store the local blocks as a `DenseTensor` when `CanBatchLocalFactor()`
   holds, so the memory is the tensor's own.
2. Give `DenseTensor` a constructor that *aliases* an existing
   `Memory<real_t>`, and hand it `Af_data`'s. This touches `linalg/densemat.hpp`
   and is the sort of small, general addition upstream takes.

Then extend the same treatment to `MultInv`, `ComputeSolution`,
`NPCReduce`/`NPCRecover` and `ComputeElementH`'s factor+Schur, which are
`LUFactors`/`DenseMatrix` object code today and must become raw-pointer or
batched calls.

**DONE, and smaller than this plan expected.** `DenseTensor::NewMemoryAndSize(
const Memory<real_t> &, i, j, k, own_mem)` **already exists**
(`linalg/densemat.hpp:1187`), so option 2 needs no addition to `densemat.hpp`
and nothing has to go upstream. `InvertA()` and `InvertD()` now hand the
tensor `Af_data`/`Df_data`'s `Memory` instead of `GetData()`, and sync back
after. Verified: the assembled trace operator is **bit-for-bit identical**
between `-d cpu` and `-d cuda` (see step 0's table), and the host Darcy suite
is unmoved at 92 cases / 28,223 assertions.

What is NOT done from the paragraph above: extending the same treatment to
`MultInv`, `ComputeSolution`, `NPCReduce`/`NPCRecover` and `ComputeElementH`'s
factor+Schur. Those are step 0's problem as much as this step's, since they
are the raw-pointer readers that make the host/device split unsafe.

**Acceptance.** The NATIVE backend on device must be **bit-for-bit** the host's,
because it runs the identical `kernels::LUFactor`/`LUSolve` scalar code — the
same argument that makes `LocalFactorMode::Batched` exact without LAPACK. The
`GPU_BLAS` and `MAGMA` backends will **not** be, for the same reason LAPACK is
not, so a test asserting equality has to say which backend it is asserting
about. Plus: a serial build unchanged, and the host `Batched` path unchanged.

## Step 2 — group 2, the integrators, and this is the work

`ElementTransformation` and `Coefficient` carry **zero** `MFEM_HOST_DEVICE`
between them, so neither can appear in a device lambda, and every integrator in
`fem/darcy` is built on both. There is no `AssemblePA` or `AssembleEA` anywhere
in `fem/darcy`. So this is a rewrite of the integrators against a different
data model, not a port:

1. **Geometry** from `GeometricFactors` / `FaceGeometricFactors`, precomputed
   into device memory, instead of asking a transformation per quadrature point.
2. **Coefficients** evaluated into `QuadratureFunction`s up front.
3. **A restriction to gather dofs.** `L2FaceRestriction` and
   `ConformingFaceRestriction` exist (`fem/restriction.hpp`) — but **not one for
   an HDG trace space**, which has to be written. This is the prerequisite with
   nothing behind it.
4. **The kernels**, per integrator, in `forall_2D/3D` with
   `MFEM_FOREACH_THREAD`. The hot-path set is `MixedConductionNLFIntegrator`
   (element and HDG face), `HDGDiffusionIntegrator`, the two
   `HDGConvection*Integrator`s, and `HyperbolicFormIntegrator` for
   Navier-Stokes. `DarcyForm::Assemble`'s loop additionally goes through
   `BilinearForm::AssembleElementMatrix` in `fem/bilinearform.cpp`, so that
   part reaches outside `fem/darcy`.

**Acceptance.** Element matrices and residuals equal to the host's to
round-off, not bitwise — a PA kernel reassociates the quadrature sum. Compare
against the existing assembled path on the same problem, and pin it with the
convergence tables the branch already has rather than only with norms.

## Step 3 — group 3, the scatter

Two routes, and one already exists. `GradientMode::MatrixFree` **deletes this
group outright** — measured at 40–47% of `NPCGradient` — so a device path that
never assembles the trace matrix skips the problem. What it pays is an
unpreconditioned trace solve at 8x, which is exactly the open question in
`doc/HDG-JACOBIAN-FREE-TRACE.md`, and this is a second reason to want it
answered. Otherwise: an `AssembleEA`-style element-matrix array plus an
assembly kernel.

## Step 4 — SOLVED, by cuDSS, and it needed a compatibility fix

**cuDSS is installed here and MFEM already wraps it**, so the trace solve can
be direct AND on-device, which is what the full-device target needs. That
removes the "a direct trace solve forces a host round trip" problem entirely,
rather than trading it for an unpreconditioned Krylov method.

* cuDSS **0.8.0**, system-wide from the CUDA apt repository
  (`/usr/include/cudss.h`, `/usr/lib/x86_64-linux-gnu/libcudss.so`). Not part
  of the CUDA toolkit -- `/usr/local/cuda/version.json` lists no cudss
  component -- but already present, so nothing to obtain.
* `CuDSSSolver` (`linalg/cudss.hpp:37`) takes a serial `SparseMatrix`, which
  is exactly what `ComputeH()` produces.
* Configure with **both directory variables**, because `CUDSS_DIR=/usr` yields
  `/usr/lib`, which has no `libcudss.so` -- the libraries are multiarch:

```
make config MFEM_BUILD_DIR=<build> MFEM_USE_CUDA=YES CUDA_ARCH=sm_75 \
     MFEM_USE_CUDSS=YES CUDSS_INCLUDE_DIR=/usr/include \
     CUDSS_LIBRARY_DIR=/usr/lib/x86_64-linux-gnu
```

**MFEM's wrapper does not compile against 0.8.0 as it stands, AND UPSTREAM HAS
ALREADY FIXED IT** -- `1416665dc3`, "Add support for cuDSS 0.8.0", John
Pennycook of NVIDIA, on `origin/master` and not an ancestor of this branch.
Cherry-picked rather than kept as a parallel fix, and doing so caught a site
the parallel fix had missed.

0.8.0 broke three things at once, all documented in NVIDIA's own *cuDSS 0.8.0
Migration Guide*: `cudssMatrixCreateCsr` gained an `offsetType` parameter
before `indexType` (14 arguments where there were 13); `cudaDataType_t` became
`cudssDataType_t`, a distinct enum aliasing the CUDA values but not implicitly
convertible; and **`CUDSS_DATA_COMM` was removed** in favour of
`CUDSS_DATA_COMM_HOST` / `CUDSS_DATA_COMM_DEVICE`. That third one is in the
MPI constructor, so a serial build never sees it -- which is exactly why a fix
written from the compiler errors of a serial build misses it.

`#if CUDSS_VERSION >= 800` is the right guard and one guard suffices, verified
against real headers for 0.3.0 through 0.8.0 rather than inferred: the
signature is 13 arguments with `cudaDataType_t` in every release up to and
including 0.7.0. `CUDSS_VERSION` (`MAJOR*10000 + MINOR*100 + PATCH`) has
existed since at least 0.3.0. Version guards are thoroughly idiomatic here:
hypre carries 84, PETSc 27, SUNDIALS 23.

Two version floors worth knowing, both from the same audit: MFEM's serial path
already needs cuDSS >= 0.5.0 (`cudssSetThreadingLayer`) and its MPI path
>= 0.6.0 (`cudssMatrixSetDistributionRow1d`). And 0.6.0 changed
`cudssExecute`'s phase parameter and the `CUDSS_PHASE_*` values, so a header
and library from different minors must never be mixed.

**One thing to check before relying on cuDSS under MPI**, flagged rather than
asserted since it has not been run here: the upstream commit sets
`CUDSS_DATA_COMM_HOST` only, while the migration guide says MGMN users must
set `CUDSS_DATA_COMM_HOST` *and* `CUDSS_DATA_COMM_DEVICE` whenever GPU-side
communication takes place.

Measured on the hybridized trace system, 2-D quads n=16 order 2, 32,544 nnz,
both under `-d cuda`:

| | trace solve | potential \|u\| |
|---|---|---|
| UMFPACK, host | 67.1 ms | `0x1.c21ebb4fbad9p+0` |
| **cuDSS, device** | **52.1 ms** | `0x1.c21ebb4fbad97p+0` |

Agreeing to twelve significant hex digits, which is two different direct
factorisations of one matrix. The timing is not the point and this hardware is
not a verdict; **the point is that nothing is pulled to the host.**

One thing to know before wiring it into a miniapp: `regression_test.py`
compares the solver NAME first, so a cuDSS run reports "SKIPPING --
incompatible preconditioner" against every existing reference rather than
failing it. A cuDSS run tells you nothing about correctness until a reference
set is generated for it.

## Step 4 — group 4, the trace solve, by configuration (superseded above)

`SparseMatrix::Mult` has a cuSPARSE/hipSPARSE path and hypre's AMG has GPU
support, so a **Krylov** trace solve runs on device today. What does not is the
direct solve the tests and miniapps default to: UMFPACK and KLU are SuiteSparse
and host-only. So this step is a solver choice, not code — but note it changes
the answer to the tolerance, unlike everything else here.

## Recommended order, and it is not the cheap-first order

1. **Step 1**, only to prove the harness against work that cannot fail for an
   interesting reason — the same reason `NPCRecover` was the right first host
   loop.
2. **Step 2**, or stop. It is the majority of the time in both regimes and
   nothing else changes that.
3. Steps 3 and 4 fall out of choices made in 2.

## Prepared, and four things in this plan were wrong

The tree exists: **`/home/ian/projects/mfem-hdg-cuda-dev`**, configured out of
source, library builds clean (163 MB, zero errors), and **MFEM's own `ex1`
runs on `-d cuda` there and prints output identical to `-d cpu`** but for the
`--device` line. So the harness step 1 wants is proven before step 1 starts.

**There is a GPU, which this plan only implied.** NVIDIA RTX 2070 SUPER,
8 GB, driver 591.86, Turing, compute capability **7.5**. So correctness on
device is checkable here.

**Performance is NOT checkable here and a slow number from this box means
nothing** — WSL2 sharing the GPU with the desktop, a consumer graphics card
rather than a compute one, and an old one. Timing belongs on real hardware
later. Everything below is about correctness and shape.

Four corrections to the recipe, each of which costs a cycle to rediscover:

* **`CUDA_ARCH = sm_60`, MFEM's default (`config/defaults.mk:49`), is REJECTED
  by CUDA 13.3** — `nvcc fatal : Unsupported gpu architecture 'sm_60'`. So
  `make config MFEM_USE_CUDA=YES` as written below fails outright. Pass
  `CUDA_ARCH=sm_75`.
* **`cp config/user.mk` breaks the CUDA build.** It enables SUNDIALS at
  `/home/ian/projects/sundials/install`, which was built WITHOUT CUDA, and
  `sundials.hpp:39` then refuses: "MFEM_USE_CUDA=TRUE requires SUNDIALS to be
  built with CUDA support". Repoint it at
  `/home/ian/projects/sundials/cuda-install`, which exists and has
  `sunmemory/sunmemory_cuda.h`.
* **The out-of-source tree leaves `MFEM_INC_DIR` and `MFEM_LIB_DIR` empty**, so
  `make -C <build>/examples ex1` fails to link with undefined *MFEM* symbols
  (`mfem::MemoryManager::Copy_`) alongside libstdc++ ones — which reads as a
  host-compiler mismatch and is not one. Pass both explicitly. An empty
  `MFEM_INC_DIR` additionally leaves a dangling `-I` that swallows the next
  include path, so the failure surfaces as a missing SUNDIALS header.
* g++ 15.2 and CUDA 13.3 **do** work together here; nvcc's own
  `host_config.h` refuses only `__GNUC__ > 15`. The link errors above are not
  that, and g++-10/12/13/14 are installed if a fallback is ever needed.

## Step 2's items 1 and 3 are ALREADY AVAILABLE, measured

The plan casts step 2 as "a rewrite of the integrators against a different
data model, not a port", with four items. **Two of the four need no work at
all**, which is the difference between building a data model and writing
kernels against one that exists.

**Item 3, the restriction, is not merely present -- its layout is already the
hybridization's.** `L2InterfaceFaceRestriction` reproduces
`c_fes.GetFaceVDofs()` **exactly, face for face**, which is how every face
loop in `darcyhybridization.cpp` gathers the trace today:

| | order 0 | order 1 | order 2 |
|---|---|---|---|
| 2-D quads | matches on all 24 faces | all 24 | all 24 |
| 2-D triangles | all 40 | all 40 | all 40 |
| 3-D hexes | all 54 | all 54 | all 54 |

Compared value by value against a load carrying each dof's own index, so a
permutation or a sign flip would show as a value and not only as a norm. So
there is nothing to write and nothing to reconcile.

**Item 1, the geometry, is served by `FaceGeometricFactors`**
(`mesh/mesh.hpp:3155`), which carries `X`, `J`, `detJ` **and `normal`** in
`(NQ x SDIM x NF)` column-major `Vector`s -- device memory, laid out the way a
kernel wants, with `Mesh::GetFaceGeometricFactors(ir, flags, face_type)` the
accessor that `bilininteg_mass_pa.cpp` and `lininteg_boundary.cpp` already
use. `NORMALS` is the flag that matters for HDG and it is there.

**What is left of step 2 is items 2 and 4**: coefficients evaluated into
`QuadratureFunction`s, and the kernels themselves. That is still the bulk of
the project, and steps 3 and 4 still fall out of it -- but it is writing
kernels against an existing data model rather than building the model.

## The prerequisite with nothing behind it already exists

Step 2's item 3 says a restriction for an HDG trace space "has to be written.
This is the prerequisite with nothing behind it." **It is written.**
`L2InterfaceFaceRestriction` (`fem/restriction.hpp:1114`) is dispatched by
`FiniteElementSpace::GetFaceRestriction()` at `fem/fespace.cpp:1533` for
exactly `dynamic_cast<const DG_Interface_FECollection*>(fec)` — which is what
the HDG trace space is. And it is not merely present: its `Mult` is already an
`mfem::forall` over a `MFEM_HOST_DEVICE` lambda with `Read()`/`Write()`
discipline (`fem/restriction.cpp:2356`), so it is device-ready as it stands.
What is unverified is whether its dof ordering suits the hybridization's face
loops; that is a much smaller question than writing one.

The plan's other structural claims were checked and hold: `fem/eltrans.hpp`
and `fem/coefficient.hpp` carry **zero** `MFEM_HOST_DEVICE` between them, and
there is no `AssemblePA` or `AssembleEA` anywhere in `fem/darcy`.

## Where to build it

CUDA 13.3 is installed (`/usr/local/cuda`, `nvcc` 13.3.73) and
`/home/ian/projects/mfem/build` is an existing `MFEM_USE_CUDA=YES` MFEM build,
so the toolchain is proven on this machine. Configure a **fourth** tree out of
source rather than touching either HDG tree, the way the OpenMP one was:
`cp config/user.mk <build>/config/user.mk` first — without it `make config`
falls back to SuiteSparse defaults that do not exist here — then
`make config MFEM_BUILD_DIR=<build> MFEM_USE_CUDA=YES`. Nothing device-shaped
can be measured in either committed tree.

## What this plan does not cover

* **Parallel + device.** The flux and potential are L2 and rank-local, so only
  the trace needs communication, and hypre handles the device side of that.
  Nothing here changes it, but nothing here has been tried on more than one
  rank either.
* **The nonlinear local solve.** `LocalNLOperator` builds a solver per element
  and calls integrators inside a local Newton loop. NPC deletes that loop
  entirely, which is why NPC is the ordering a device path should target; the
  reduced trace operator's fused local Newton is a much harder device shape and
  is not planned here.
