# Device offload for the element-local HDG work — the plan

Scratch, like every `.md` here. `doc/HDG-ELEMENT-LOCAL-PARALLELISM.md` is the
host-threading side of this and is **done**: every element-local loop in
`DarcyHybridization` threads, bit-for-bit, and the measurements there are what
bound everything below. This file is the device plan and nothing in it is
built.

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

## Step 0 — THE PREREQUISITE THIS PLAN DID NOT HAVE, and it comes before everything

**Configuring a CUDA `Device` silently breaks the existing host code**, before
any offload is attempted. Measured on `gf-hdg-linearise-first`, order 1, 8x8
quads, hybridized, `LocalFactorMode::Batched`:

| quantity | `-d cpu` | `-d cuda` |
|---|---|---|
| assembled trace operator `H` (nnz, sum, maxabs) | identical | **identical, bit for bit** |
| the load, `\|rhs_p\|` | `0x1.eb851eb851ec3p-5` | `0x1.eb851eb851eb9p-5` |
| the REDUCED trace load `\|RHS\|` | `0x1.ec096b6d502eep-5` | **`0x0p+0`** |
| trace solution `\|X\|` | `0x1.2c1d0cfe578fcp-1` | **`0x0p+0`** |

**No error is raised anywhere.** The reduced right-hand side is exactly zero,
the trace solve returns zero, and the recovered fields are quietly wrong.

The mechanism, as far as it was chased: `DarcyHybridization`'s element loops
read and write through raw host pointers (`Array::operator[]`, `DenseMatrix`
over `&data[offset]`, `Vector::MakeRef`) and nothing in them tells the
`Memory` that the host copy is now the valid one. Interleaved with them are
ordinary `Vector` operations -- `b_tr = 0.`, `Operator::MultTranspose` -- which
under a Device run on the device and claim it. The host writes then land in
memory the vector believes is stale, and the next `HostRead()` copies the
device side back over them.

**Two attempts to retrofit this at single sites in `ReduceRHS()` did NOT fix
it and were reverted** rather than shipped unverified: claiming the host side
before the loop is undone by the `= 0.` that follows, and claiming it after
still leaves the answer zero, so at least one more site upstream is involved.
Finding the complete set is the work of this step. It is mechanical but it is
not small, and **nothing in steps 1 to 4 can be verified on a device until it
is done** -- a device run cannot be compared against a host one while the
device run is silently returning zeros.

Note what this does NOT say: the *assembly* path is fine. `H` comes back bit
for bit identical, which is why step 1 below could be verified at all.

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

## Step 4 — group 4, the trace solve, by configuration

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
