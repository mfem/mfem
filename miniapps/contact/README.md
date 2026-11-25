# Contact Miniapp 
Frictionless contact benchmarks and Coulomb-friction ironing using MFEM and Tribol.

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
 compression, ironing, and beam-sphere contact tests. 

Both frictionless and Coulomb contact use Tribol's formulation-backed single
mortar with frozen geometry within each load increment. Frictionless contact
uses the same nonmortar/multiplier surface as Coulomb contact, passed first to
the formulation API. The bulk model can still use nonlinear elasticity
for beam-sphere contact. Enzyme is not required for either mortar mode.

---

### AMG with Filtering (AMGF)

The **AMGF** preconditioner extends classical AMG with a targeted small-subspace 
correction restricted to degrees of freedom involved in active contact constraints. 
This correction significantly improves solver convergence in late-stage IP iterations 
where contact constraints dominate and classical AMG performance typically degrades.

- **AMGF** requires a user-specified solver for the filtered subspace. In this 
miniapp a  parallel sparse direct solver is used, so MFEM must be built with 
either **MUMPS** (`MFEM_USE_MUMPS=YES`) or **CPardiso** (`MFEM_USE_CPARDISO=YES`). 
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

### Coulomb friction in the ironing problem

A positive `-mu` enables Coulomb friction for `-prob 1`. The original ironing
loads run on two MPI ranks with:

```bash
mpirun -np 2 contact -prob 1 -sr 0 -pr 0 -tr 2 -mu 0.1 \
  -nsteps 4 -msteps 6 -no-vis
```

A shorter regression case uses `-nsteps 1 -msteps 1 -indent 0.02 -travel 0.04`.
Use Tribol's 3D vector-LM frictional single mortar API (`e7212777` or later on
`feature/ebchin/coulomb-formulation`), with MPI and Redecomp. Enzyme is not
required. The parent topology and partition remain fixed, and the bulk model
remains small-strain linear elasticity.

`OptContactProblem` owns the mechanics and frozen mortar data for both contact
modes. `IPSolver` and `CoulombSolver` implement the same single-increment
`ContactSolver::Solve()` interface; the load driver sets boundary conditions,
subdivides increments, and updates coordinates after acceptance. Solver failure
returns the initial displacement and restores persistent multiplier input.
The existing `IPSolver::Mult()` API and external linear-solver support remain.

Coulomb mortar holds weights and normals fixed within each increment, using
`k = k_start + G * (u - u_start)` and contact residual `G^T * lambda`.
MFEM enforces the normal complementarity and Coulomb disk projection with a
Newton/FGMRES solve. Compression-positive pressure is `-n . lambda`, the
opening-positive weighted gap is `n . k`, and tangential slip is `k - (n . k)*n`.
Previous-step coordinates advance only after accepting an increment. Both
frictionless and Coulomb ironing use the flat substrate as the nonmortar surface
carrying the multipliers, and the curved iron as the mortar surface. Both modes
use formulation-backed mortar. Reported open/stick/slip counts
include supported contact nodes; unsupported dual nodes are listed separately.
`-amgf` is supported by the IP
solver; Coulomb runs use `-no-amgf` (the default).

`-indent` and `-travel` set total downward and horizontal displacement;
negative travel reverses the sliding direction. Defaults preserve the original
loads `1.0/1.4` and `3.0/1.4`. `-fstep` limits a Coulomb displacement increment
(default `0.05`). Failed solves retry with smaller increments and return a
nonzero exit status if subdivision is exhausted. Acceptance checks pressure,
the linearized gap, complementarity, the friction cone, and dissipation; it
does not certify the exact geometric gap after a finite increment.

`-mu 0.0` selects scalar frictionless mortar with the IP solver. Use
`-coulomb -mu 0.0` to compare zero and positive friction through the same
Coulomb solver and surface assignment.
After building `contact`, run the projection derivative checks and the
friction/frictionless regressions with:

```bash
ctest --test-dir <mfem-build-dir> -R 'contact_(coulomb|ironing)' --output-on-failure
```

### Notes

- `-prob` selects the problem: `0` two-block, `1` ironing, `2` beam–sphere.  
- `-nonlin`: enables nonlinear elasticity. This option is available only for the 
beam–sphere case (`-prob 2`). 
- `-amgf` selects the AMGF  preconditioner and requires MFEM to be built with 
**MUMPS** or **CPardiso**. Use `-no-amgf` to fall back to standard HypreBoomerAMG. 
- `-nsteps` sets the number of incremental load steps (pseudo time-stepping)    
- `-sr` sets the number of serial refinements
- `-pr` sets the number of parallel refinements
- `-tr` sets the Tribol proximity parameter
- `-vis`: real-time visualization using [GLVis](https://glvis.org) 
- `-paraview` enables ParaView visualization

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
