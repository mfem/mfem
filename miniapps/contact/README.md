# Contact Miniapp 
Frictionless contact mechanics examples using MFEM and the Tribol contact library.

## Description

The **Contact Miniapp** demonstrates the coupling between **MFEM** and [**Tribol**](https://github.com/LLNL/Tribol) for simulating *frictionless contact* in deformable solids. Tribol provides the contact gap function and its Jacobian through a mortar segment-to-segment formulation. A standalone **Interior Point (IP)** solver is used to solve an inequality-constrained minimization problem enforcing the non-penetration condition. Linear systems arising in the IP iterations are solved with **Conjugate Gradient (CG)** preconditioned by the recently introduced **AMG with Filtering (AMGF)** solver. Benchmark examples include the two-block compression, ironing, and beam-sphere contact tests. 

---

### AMG with Filtering (AMGF)

The **AMGF** preconditioner extends classical AMG with a targeted small-subspace correction restricted to degrees of freedom involed in active contact constraints. This correction significantly improves solver convergence in late-stage IP iterations where contact constraints dominate and classical AMG performance typically degrades.

- **AMGF** requires a user-specified solver for the filtered subspace. In this miniapp a  parallel sparse direct solver is used, so MFEM must be built with either **MUMPS** (`MFEM_USE_MUMPS=YES`) or **CPardiso** (`MFEM_USE_CPARDISO=YES`). 
- **AMG** may be used when no direct solver is available, though it often exhibits slower or stagnating convergence.

---

## Usage

The miniapp can be run only in parallel. Typical runs specify the problem ID, material settings, and number of incremental load steps.

Example (4 MPI ranks):

```bash
mpirun -np 4 contact -prob 0 -sr 0 -pr 0 -tr 2 -nsteps 4 -amgf
```

This example runs the **two-block** frictionless contact problem with linear elasticity  
and **AMG with Filtering (AMGF)** preconditioning.

### Notes

- `-prob` selects the problem: `0` two-block, `1` ironing, `2` beam–sphere.  
- `-nonlin`: enables nonlinear elasticity. This option is available only for the beam–sphere case (`-prob 2`). 
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
S Petrides, T Hartland, T Kolev, CS Lee, M Puso, J Solberg, EB Chin, J Wang, C Petra. ***AMG with Filtering: An Efficient Preconditioner for Interior Point Methods in Large-Scale Contact Mechanics Optimization***. In: (2025). [DOI: 10.48550/arXiv.2505.18576](https://arxiv.org/abs/2505.18576) 

---

## Installation

Tribol depends on [Axom](https://github.com/LLNL/axom) and MFEM. Although Tribol can be built automatically via **uberenv** and **Spack**, for this miniapp it is simpler to build **Axom** and **MFEM** manually and point Tribol to them. The steps are as follows:

### Manual Build Steps

1. **MFEM:** Build with `MFEM_USE_MPI`, `MFEM_USE_METIS`, and `MFEM_USE_TRIBOL`  
   enabled, plus MUMPS or CPardiso if AMGF is desired.  
 
2. [**Axom:**](https://github.com/LLNL/axom.git) Starting from the MFEM root directory (we assume this directory is named mfem) type
   ```bash
   cd .. && git clone --recursive https://github.com/LLNL/axom.git axom-repo
   cd axom-repo
   python3 ./config-build.py -hc ../mfem/miniapps/contact/axom.cmake -bt Release -DCMAKE_INSTALL_PREFIX=../axom
   cd build-axom-release && make -j install
   ```
3. [**Tribol:**](https://github.com/LLNL/Tribol.git) Starting from the MFEM root directory type
   ```bash
   cd .. && git clone --recursive https://github.com/LLNL/Tribol.git tribol-repo
   cd tribol-repo
   python3 ./config-build.py -hc ../mfem/miniapps/tribol/tribol.cmake -bt Release -DCMAKE_INSTALL_PREFIX=../tribol 
   cd build-tribol-release && make -j install
   ```
   > **C++ Standard**: Starting with version 4.8.1, MFEM requires a C++17 compiler. Ensure that Tribol uses the same standard by adding `-DCMAKE_CXX_STANDARD=17 -DBLT_CXX_STD=c++17`.

4. **MFEM Contact Miniapp**: Build from inside the directory `mfem/miniapps/contact`:  
     ```bash
      make contact
      ```  

