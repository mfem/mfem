# Contact Miniapp 
Frictionless contact mechanics examples using MFEM and the Tribol contact library.

## Description

The **Contact Miniapp** demonstrates the coupling between **MFEM** and 
[**Tribol**](https://github.com/LLNL/Tribol) for simulating 
*frictionless contact* in deformable solids. Tribol provides the contact gap 
function and its Jacobian through a mortar segment-to-segment formulation. 
A standalone **Interior Point (IP)** solver is used to solve an 
inequality-constrained minimization problem enforcing the non-penetration 
condition. Linear systems arising in the IP iterations are solved with 
**Conjugate Gradient (CG)** preconditioned by the recently introduced
 **AMG with Filtering (AMGF)** solver. Benchmark examples include the two-block 
 compression, ironing, beam-sphere, onion (nested spheres), folded-strip and ring
 self-contact tests.

---

### AMG with Filtering (AMGF)

The **AMGF** preconditioner extends classical AMG with a targeted small-subspace 
correction restricted to degrees of freedom involved in active contact constraints. 
This correction significantly improves solver convergence in late-stage IP iterations 
where contact constraints dominate and classical AMG performance typically degrades.

- **AMGF** requires a sparse direct solver when the filtered subspace is nonempty.
  Build MFEM with **MUMPS**, **CPardiso**, **SuperLU_DIST**, **STRUMPACK**, or **cuDSS**.
  Select one with `-amgf-fsolver mumps|cpardiso|superlu|strumpack|cudss`.
  `auto` prefers cuDSS on CUDA when available; otherwise it selects the first
  enabled host solver in the order MUMPS, SuperLU_DIST, STRUMPACK, CPardiso.
  With OpenMPI, `auto` also checks CUDA buffer support for multiple ranks.
- When the **global contact subspace is empty**, `-amgf` automatically uses
  ordinary BoomerAMG with the same AMG options. No filtered operator or direct
  solver is constructed. Once candidate contact rows appear, AMGF is enabled
  again. A rank with no local contact entries still participates if the global
  subspace is nonempty. The driver checks the selected direct backend only when
  it is needed.
- **AMG** may be used when no direct solver is available, though it often exhibits 
slower or stagnating convergence.

---

## Usage

The miniapp can be run only in parallel. Typical runs specify the problem ID, 
material settings, and number of incremental load steps.

Example (4 MPI ranks):

```bash
mpirun -np 4 contact -prob 0 -sr 0 -pr 0 -tr 2 -nsteps 4 -amgf
```

This example runs the **two-block** frictionless contact problem with linear 
elasticity and **AMG with Filtering (AMGF)** preconditioning.

### Notes

- `-prob` selects the problem: `0` two-block, `1` ironing, `2` beam–sphere,
`3` onion (9 nested spherical shells under equal inner and outer pressure),
`4` folded U-shaped strip, `5` compressed elastic ring (each a single connected
deformable body).
- The onion mesh is generated at startup; only the beam–sphere problem reads a
  mesh file from `meshes/`.
- `-nonlin`: enables nonlinear elasticity. This option is available only for the 
beam–sphere case (`-prob 2`). The self-contact examples (`-prob 4/5`) always
enable it.
- `-amgf` selects the AMGF preconditioner and requires one of the direct solvers
  listed above when the contact subspace is nonempty. Use `-no-amgf` to select
  standard HypreBoomerAMG.
- `-nsteps` sets the number of incremental load steps (pseudo time-stepping)    
- `-sr` sets the number of serial refinements
- `-pr` sets the number of parallel refinements
- `-tr` sets the Tribol proximity parameter
- `-vis`: real-time visualization using [GLVis](https://glvis.org) 
- `-paraview` enables ParaView visualization

Each load step reports optimizer termination status and final stationarity,
feasibility, complementarity, and optimality errors. A failed linear solve,
line search, curvature test, or optimizer convergence check stops the load
sequence and returns exit code 2.

### Folded U-shaped strip: `-prob 4`

The generated mesh is a single connected 3D hexahedral solid with two straight
arms joined by a semicircular bend. Each arm is 6 units long, the bend's
centerline radius is 0.72, the width is 0.9, and the thickness is 0.3. The initial
gap between the straight arms is 1.14. The coarse mesh has 24 cells along each
arm, 24 around the bend, and four across both width and thickness: 1152 hexes
and 5475 displacement DOFs. `-sr` and `-pr` refine it using the usual options.

A short patch at the center of the outer bend is clamped; both arm tips are
free. A downward reference-surface pressure loads the outer face of the upper
arm. Material parameters are
Young's modulus 1000 and Poisson ratio 0.3, using the existing device-enabled
Neo-Hookean model. The pressure is a dead load assembled on the reference mesh,
not a follower pressure updated with the deformed normal.

| Boundary attribute | Meaning |
|---|---|
| 1 | Clamped outer bend patch (all displacement components) |
| 2 | Outer upper-arm surface: downward loading |
| 3 | Facing surface of the lower arm: mortar |
| 4 | Facing surface of the upper arm: non-mortar |
| 5 | Free surfaces, including both tips, the remaining bend and width faces |

This is frictionless self-contact between two prescribed, disjoint surface
patches of the same body. The bend is excluded from contact pairing. It does
not perform arbitrary whole-surface self-contact or neighboring-face exclusion.
It uses the existing Tribol single-mortar formulation without library changes.

For this problem, `-nsteps` defaults to **20 loading steps**, `-u-pressure`
defaults to **0.06**, and `-tr` defaults to **2**. The other problems retain their
previous defaults. Optional `-msteps N` adds N unloading steps, ending at zero
pressure. A small noncontact test is `-u-pressure 0.0001 -nsteps 1 -sr 0`.

Each load step repeats the quadratic IP solve while refreshing the
Neo-Hookean tangent and the mortar geometry at the latest displacement. Updates
are limited to 0.1 in the global displacement infinity norm. The step finishes
when the undamped correction is at most `-contact-update-tol` (default 1e-6).
`-contact-max-updates` defaults to 50; exceeding it returns exit code 2. These
updates replace the beam-sphere displacement bounds, allowing contact release
during unloading. `-opt-maxit` limits each individual IP solve, so use its
default 100 for a complete physical loading cycle.

At print level 2 the output includes geometry updates, global gap-row count,
and minimum mortar gap. Gap rows count candidate constraints, including
separated surfaces within the search distance; they do not count faces in
physical contact. The minimum mortar gap is the normalized, integrated weak
gap evaluated in the final contact linearization, not the minimum pointwise
distance. `n/a` means there are no candidate pairs. A near-zero gap under load
indicates contact. IP/Newton/CG counts sum all geometry updates for that load
step; optimality errors are from the final IP solve, and the separate
displacement-update error checks nonlinear/geometry convergence. Reported strip
energies use the actual Neo-Hookean potential including the reference load.

Inside a GPU allocation, run a loading/unloading cycle with:

```bash
./run-contact-bound.sh --ranks 4 --cores 4 -- \
  -prob 4 -sr 0 -pr 0 -nsteps 12 -msteps 12 -u-pressure 0.06 \
  -amgf -amgf-fsolver mumps -d hip -hypre-gpu-aware-mpi \
  -no-vis -paraview -pl 2
```

Use `-d cpu` and omit `-hypre-gpu-aware-mpi` for a host run. Constitutive
evaluation, element-matrix kernels, vector operations, and HYPRE solves use
the selected device. Tangent assembly transfers ownership of the local CSR
storage once and subsequently writes through a non-owning view of those same
allocations. Each rebuild synchronizes the view's memory validity and restores
HYPRE's diagonal-first ordering before using the matrix. This applies both to
updates within a load step and to later load steps. Identity prolongation uses
that matrix directly; otherwise parallel assembly computes `P^T K P` on HYPRE's
active backend. There is no full CSR host staging copy or tangent clone. The
existing element-assembly routine still restores its row offsets on the host,
so the small row-offset array moves between host and device. HYPRE BIGINT builds
use the existing index-conversion wrapper and borrow the numerical values.
Tribol's implicit mortar search, integration, and contact CSR preparation also
use the host. ParaView output is written under `ParaView/contact-problem_4_*`.

The integration test uses four loading steps and two unloading steps. It checks
a nonzero-load empty-subspace solve against plain
AMG, contact onset, non-penetration of the weak mortar gap, geometry convergence,
and contact release on unloading. It also requests the unavailable cuDSS backend
in the empty case to verify that no direct solver is constructed:

```bash
python3 test-self-contact.py --device cpu
python3 test-self-contact.py --device hip --gpu-aware-mpi \
  --launcher './run-contact-bound.sh --ranks 4 --cores 4 --'
```

Use `--solver strumpack` to test the alternative filtered solver. Test logs are
saved to a temporary directory, or to the directory supplied by `--output-dir`.

### Compressed elastic ring: `-prob 5`

The generated circular ring is one closed hexahedral solid, with shared vertices
at the periodic seam. Its centerline radius is 1.0, radial thickness is 0.12, and
width is 0.6. The coarse mesh has 64 circumferential cells, two through the
thickness and two across the width: 256 hexes and 1728 displacement DOFs. The
initial inner diameter is 1.88. `-sr` and `-pr` refine the mesh.

The lower outer arc within 11.25 degrees of the bottom is clamped (attribute 1).
Inward pressure loads the upper outer arc within 22.5 degrees of the top
(attribute 6). This is a dead pressure load assembled using reference-surface
normals; it is predominantly downward and also has horizontal components.
The material is the same nonlinear Neo-Hookean material as the strip, with
Young's modulus 1000 and Poisson ratio 0.3.

The lower inner arc within 45 degrees of the bottom is the mortar patch
(attribute 3); the opposing upper inner arc is the non-mortar patch
(attribute 4). Lateral inner arcs and width faces are free (attribute 2), keeping
the contact patches disjoint. This example models frictionless bore closure
between prescribed patches, rather than unrestricted whole-surface self-contact.

`-ring-pressure` sets maximum pressure (default 2.5). The example defaults to
20 loading steps and search ratio `-tr 2`; `-msteps N` unloads to zero pressure.
It shares the strip's nonlinear material/contact geometry updates, convergence
checks, energy and weak-gap reporting, and empty-subspace AMG fallback. Geometry
corrections are capped at 0.04, one third of the radial thickness. A small
noncontact case is `-ring-pressure 0.005 -nsteps 1 -sr 0`.

```bash
./run-contact-bound.sh --ranks 4 --cores 4 -- \
  -prob 5 -sr 0 -pr 0 -nsteps 4 -msteps 2 \
  -amgf -amgf-fsolver mumps -d hip -hypre-gpu-aware-mpi \
  -no-vis -paraview -pl 2

python3 test-self-contact.py --problem ring --device cpu
python3 test-self-contact.py --problem ring --device hip --gpu-aware-mpi \
  --launcher './run-contact-bound.sh --ranks 4 --cores 4 --'
```

ParaView output is written under `ParaView/contact-problem_5_*`. The ring test
checks empty-contact AMG fallback, bore closure without significant penetration
of the weak mortar gap, convergence of every loading/unloading step, and release
to the unloaded elastic state.

The coarse four-loading/two-unloading cycles were verified with MUMPS: the
updated folded strip on four HIP GPUs, and the ring on one CPU rank and on one
and four HIP GPUs. Every physical step converged; both examples finished with
zero contact rows and absolute elastic energy below 1e-12. The strip's unloaded
energy was 1.90835e-13. The maximum-load results are:

| Example | Configuration | Maximum pressure | Gap rows at maximum load | Minimum weak mortar gap |
|---|---|---:|---:|---:|
| Folded strip | HIP, 4 ranks | 0.06 | 85 | 1.2254e-7 |
| Ring | CPU, 1 rank; HIP, 1 and 4 ranks | 2.5 | 45 | 4.38709e-9 |

The unavailable-direct-backend empty-contact tests also passed on CPU and HIP.
These are correctness checks on coarse meshes, rather than scaling benchmarks.

### GPU execution

BoomerAMG uses Chebyshev smoothing by default (`-amg-relax-type 16`) for both
AMG and AMGF. Select the previous l1-Jacobi smoother with
`-amg-relax-type 18`. Coarsening 8, interpolation 6, and zero aggressive levels
remain the defaults.

`-hypre-gpu-aware-mpi` enables HYPRE's runtime GPU-aware communication flag
after device initialization. Its default is disabled; use
`-no-hypre-gpu-aware-mpi` to disable it explicitly. The MPI installation must
accept device buffers. On Tuolumne, `run-contact-bound.sh` automatically sets
`MPICH_GPU_SUPPORT_ENABLED=1` when this option is enabled; for other launchers,
export it explicitly. The existing build links Cray MPICH's HSA GPU transport. MFEM's separate
GPU-aware MPI setting is controlled by `MFEM_GPU_AWARE_MPI`: leave that
environment variable unset for the measured HYPRE-only configuration.

To run the native-HIP version of the fastest tested solver/refinement settings,
first build only the miniapp:

```bash
cd /usr/workspace/socratis/nnl/GPU/mfem/miniapps/contact
make -j3 contact
```

Then, inside an allocated four-GPU Tuolumne node:

```bash
export MPICH_GPU_SUPPORT_ENABLED=1 HSA_XNACK=0
unset MFEM_DEVICE MFEM_MEMORY MFEM_GPU_AWARE_MPI
./run-contact-bound.sh --ranks 4 --cores 24 -- \
  -prob 0 -sr 0 -pr 3 -tr 2 -nsteps 1 -opt-maxit 1 \
  -amgf -amgf-fsolver mumps -fa -d hip -hypre-gpu-aware-mpi \
  -no-vis -no-paraview -pl 1
```

Use `--ranks 1`, `2`, or `4` with identical miniapp options for strong scaling.
The wrapper binds one GPU and its local CPU/NUMA domain per rank, uses one
OpenMP/BLAS thread, enables MPICH GPU support for the HYPRE option, and
prints the per-rank MPI settings plus `TOTAL_WALL_SECONDS`. To inspect the configuration once, use
`-pl 2 -amg-print-options`: HYPRE GPU-aware MPI should print `enabled` and
AMG down/up relaxation should be 16. With `-opt-maxit 1`, exit 2 after one
accepted optimizer step is expected; the inner CG solve must still converge.

The prior study's 9.09-second result was a synchronized program timer, excluding
launch/device initialization and final teardown; corresponding full elapsed
time was 12.18 seconds. It also explicitly allocated selected displacement
vectors as managed memory. The native `-d hip` command above reproduces the
solver, refinement, and communication settings with ordinary HIP allocation;
its performance has not yet been measured under identical conditions. For
the exact measured selective-memory command, see
`../../../contact-scaling-investigation-20261117/REPRODUCE.md`.

For host execution, specify `-d cpu`. The launcher accepts MPI rank counts
above four, defaults to one CPU core per rank, requests no GPUs, and uses
Flux per-task CPU affinity with local host-memory allocation. Rank count
times `--cores` must fit the single-node CPU allocation. OpenMP and BLAS
remain single-threaded per rank.

For example, inside a sufficiently large single-node allocation:

```bash
unset MFEM_DEVICE MFEM_MEMORY MFEM_GPU_AWARE_MPI
./run-contact-bound.sh --ranks 16 --cores 1 -- \
  -prob 0 -sr 0 -pr 3 -tr 2 -nsteps 1 -opt-maxit 1 \
  -amgf -amgf-fsolver mumps -fa -d cpu -no-vis -no-paraview -pl 1
```

Use 32, 48, or 96 ranks similarly when enough CPU cores are allocated.
Omit `-hypre-gpu-aware-mpi` for host runs. HIP runs retain the one-GPU-per-rank
binding and the four-rank limit on one Tuolumne node.

Use a CUDA-enabled MFEM and a CUDA-enabled HYPRE build for `-d cuda`. The HIP
equivalent is `-d hip`, with HIP enabled in both libraries. GPU HYPRE builds
require version 2.31 or newer for MFEM's runtime host/device memory policy
support used around Tribol. HYPRE 3.0 is the validated CUDA configuration.

For example, from `miniapps/contact`, on one GPU:

```bash
mpirun -np 1 ./contact -prob 2 -sr 0 -pr 0 -tr 2 -nsteps 2 \
  -lin -fa -d cuda -amgf -amgf-fsolver mumps -no-vis
```

`-fa` assembles the linear elasticity stiffness matrix on the device. Without
it, stiffness assembly uses the legacy host path, while the IP vector kernels,
contact filtering, and GPU HYPRE operations still use the device. Nonlinear
beam-sphere runs use the same portable Neo-Hookean implementation to evaluate
reference energy, gradient, and Hessian on CPU and device backends. `-fa` controls
only the linear elasticity model.

Pressure RHS assembly, lumped masses, bound Jacobian assembly, and contact
CSR filtering also have device paths. Tribol's implicit single-mortar residual
and Jacobian remain host operations: the miniapp synchronizes coordinates and
temporarily selects HYPRE's host policy around those calls. A device-enabled
Tribol build does not change that implicit-mortar limitation.

MUMPS and CPardiso perform filtered solves on the host. To keep the filtered
solve on CUDA, enable `MFEM_USE_CUDSS=YES` and use `-amgf-fsolver cudss` (or
`auto`). Set `CUDSS_COMM_LIB` to the cuDSS OpenMPI communication layer matching
your CUDA-aware MPI installation, unless MFEM already records that path.
The communication layer must support device buffers. For example:

```bash
export CUDSS_COMM_LIB=<cudss-prefix>/lib/libcudss_commlayer_openmpi.so
mpirun -np 1 ./contact -prob 2 -sr 0 -tr 2 -nsteps 2 \
  -fa -d cuda -amgf -amgf-fsolver cudss -no-vis
```

For multiple GPUs, use the MPI launcher's device binding to assign ranks to
GPUs. Multiple ranks sharing one GPU can have substantially different timing.

The contact and Tribol miniapp targets select C++20 for GPU builds because
their device-enabled Tribol dependencies require it; MFEM itself can remain
C++17. With NVCC 13.4, the header-only fmt 12 configuration exported by some
Umpire builds can fail in `fmt/format-inl.h`. Use compatible dependency headers
or a Umpire build using compiled fmt. An example CUDA CMake configuration is:

```bash
cmake -S . -B <mfem-build-dir> \
  -DMFEM_USE_MPI=YES -DMFEM_USE_CUDA=YES -DMFEM_USE_TRIBOL=YES \
  -DCMAKE_CUDA_ARCHITECTURES=<gpu-architecture-number> \
  -DHYPRE_DIR=<path/to/cuda-hypre> -DMETIS_DIR=<path/to/metis> \
  -DAXOM_DIR=<path/to/axom> -DTRIBOL_DIR=<path/to/tribol> \
  -DRAJA_DIR=<path/to/raja> -DUMPIRE_DIR=<path/to/umpire> \
  -DCAMP_DIR=<path/to/camp>
cmake --build <mfem-build-dir> --target contact -j
```

Add `-DMFEM_USE_MUMPS=YES -DMUMPS_DIR=<path/to/mumps>` or
`-DMFEM_USE_CUDSS=YES -DCUDSS_DIR=<path/to/cudss>` to enable an AMGF filtered
solver. The architecture number is hardware-specific (for example, `86` for
an RTX A4000). For Make builds, enable `MFEM_USE_CUDA=YES`, set `CUDA_ARCH` and
the dependency prefixes, and compile native device kernels with the CUDA
compiler (`CUDA_CXX=nvcc`, with the compatible `MPICXX` host compiler).

---

## Accompanying presentation

[MFEM Community workshop 2025 - AMG with Filtering](https://mfem.org/pdf/workshop25/29_Petrides_AMG_Filtering.pdf)

---

## References:
S Petrides, T Hartland, T Kolev, CS Lee, M Puso, J Solberg, EB Chin, J Wang, C Petra.
***AMG with Filtering: An Efficient Preconditioner for Interior Point Methods*** 
***in Large-Scale Contact Mechanics Optimization***. 
In: (2025). [DOI: 10.48550/arXiv.2505.18576](https://arxiv.org/abs/2505.18576) 

---

## Installation

Tribol depends on [Axom](https://github.com/LLNL/axom) and MFEM. 
Although Tribol can be built automatically via **uberenv** and **Spack**, 
for this miniapp it is simpler to build **Axom** and **MFEM** manually and 
point Tribol to them. The steps are as follows:

### Using pre-built Tribol/Axom installs

If you already have compatible installs of Tribol and Axom, point MFEM to the install prefixes.

- Hypre install prefix: `<path/to/hypre>`
- METIS install prefix: `<path/to/metis>`
- Axom install prefix: `<path/to/axom>`
- Tribol install prefix: `<path/to/tribol>`

**MFEM make build (configure):**
```bash
make config MFEM_USE_MPI=YES MFEM_USE_METIS=YES MFEM_USE_METIS_5=YES MFEM_USE_TRIBOL=YES \
  CXXFLAGS="-std=c++20" \
  HYPRE_DIR=<path/to/hypre> METIS_DIR=<path/to/metis> \
  AXOM_DIR=<path/to/axom> TRIBOL_DIR=<path/to/tribol> ADIAK_DIR=<path/to/adiak> CAMP_DIR=<path/to/camp> \
  RAJA_DIR=<path/to/raja> UMPIRE_DIR=<path/to/umpire> FMT_DIR=<path/to/fmt> CALIPER_DIR=<path/to/caliper>
```

**MFEM CMake build (configure):**
```bash
cmake -S . -B <mfem-build-dir> -DCMAKE_CXX_STANDARD=20 \
  -DMFEM_USE_MPI=YES -DMFEM_USE_METIS=YES -DMFEM_USE_TRIBOL=YES \
  -DHYPRE_DIR=<path/to/hypre> -DMETIS_DIR=<path/to/metis> \
  -DAXOM_DIR=<path/to/axom> -DTRIBOL_DIR=<path/to/tribol> -DADIAK_DIR=<path/to/adiak> -DCAMP_DIR=<path/to/camp> \
  -DRAJA_DIR=<path/to/raja> -DUMPIRE_DIR=<path/to/umpire> -DCALIPER_DIR=<path/to/caliper>
```

Note: RAJA/UMPIRE/CALIPER are optional for MFEM itself, but many Tribol builds
enable them. If your Tribol install does not depend on them, you can omit the
corresponding `*_DIR` entries above.

Note: `FMT_DIR` only needs to be added for the make-based build (and only when
the Umpire install uses `fmt`). If `FMT_DIR` is not set and a sibling `fmt-*`
directory exists next to your `UMPIRE_DIR`, MFEM's make configuration will try
to pick it up automatically.

Note: when using pre-built Tribol/Axom, you typically need to use a compatible
compiler/MPI wrapper (same C++ standard library ABI).

### Manual Build Steps

1. Pull axom and tribol (starting from the mfem folder):
   ```bash
   cd ..
   git clone --recursive https://github.com/LLNL/axom.git axom-repo
   git clone --recursive https://github.com/LLNL/Tribol.git tribol-repo
   ```
2. **MFEM:** Build with `MFEM_USE_MPI`, `MFEM_USE_METIS`, `MFEM_USE_METIS_5`, 
   and `MFEM_USE_TRIBOL` enabled, plus MUMPS or CPardiso if AMGF is desired.
   Note: it's important to do `make install`.
   ```
   MFEM_USE_TRIBOL = YES
   AXOM_DIR = @MFEM_DIR@/../axom-repo/axom
   TRIBOL_DIR = @MFEM_DIR@/../tribol-repo/tribol
   TRIBOL_OPT = -I$(TRIBOL_DIR)/include -I$(AXOM_DIR)/include
   TRIBOL_LIB = -L$(TRIBOL_DIR)/lib -ltribol -ltribol_shared -lredecomp -L$(AXOM_DIR)/lib \
   -laxom_quest -laxom_mint -laxom_slam -laxom_slic -laxom_lumberjack -laxom_core
   ```
3. [**Axom:**](https://github.com/LLNL/axom.git) Starting from the MFEM root
   directory (we assume this directory is named mfem):
   ```bash
   cd ../axom-repo
   python3 ./config-build.py -hc ../mfem/miniapps/contact/axom.cmake -bt Release -DCMAKE_INSTALL_PREFIX=../axom
   cd build-axom-release && make -j install
   ```
4. [**Tribol:**](https://github.com/LLNL/Tribol.git) Starting from the MFEM root
   directory:
   ```bash
   cd ../tribol-repo
   python3 ./config-build.py -hc ../mfem/miniapps/contact/tribol.cmake -bt Release -DCMAKE_INSTALL_PREFIX=../tribol 
   cd build-tribol-release && make -j install
   ```
   > **tribol.cmake file**: Modify as needed to match `axom` and `mfem` paths.

5. **MFEM Contact Miniapp**: Build from inside the directory `mfem/miniapps/contact`:  
     ```bash
      make contact
      ```  
