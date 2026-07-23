# Plan: Create MFEM-SAMRAI Coupled Example (ex16)

## Context

This plan creates a new example in `examples/samrai/ex16.cpp` that demonstrates **basic interoperability** between MFEM and SAMRAI by running both solvers side-by-side in an alternating time loop. The example shows that MFEM and SAMRAI can coexist and be built together in the same program.

**What it demonstrates:**
- MFEM solves the **nonlinear heat equation** (from original examples/ex16.cpp)
- SAMRAI solves **linear advection** on structured AMR grid
- Both advance in an alternating time loop
- **No data sharing** - each library operates independently
- Purpose: Demonstrate basic interoperability and foundation for future coupling

**Note:** This creates a NEW file `examples/samrai/ex16.cpp`, separate from the existing `examples/ex16.cpp`. MFEM conventionally reuses example numbers across subdirectories (e.g., `sundials/ex16.cpp` also exists).

## Requirements & Constraints

Based on user clarifications:

**Integration Approach:**
- Coupled MFEM-SAMRAI demonstration
- MFEM keeps all objects from original ex16.cpp (Mesh, GridFunction, ConductionOperator)
- SAMRAI uses LinAdv model with structured AMR hierarchy
- Alternating execution: MFEM timestep → SAMRAI timestep → repeat
- No data transfer between libraries in this version

**Dimensionality:**
- Provide both 2D and 3D input files
- Create `samrai_input.2d` and `samrai_input.3d`
- MFEM heat equation works in both 2D and 3D

**MPI Requirements:**
- SAMRAI requires MPI support
- MFEM must be built with: `MFEM_USE_MPI=ON`
- May need: `MFEM_FETCH_TPLS=ON` to auto-fetch HYPRE dependency

**Build Configuration:**
- Location: `examples/samrai/` subdirectory
- Example name: `ex16.cpp` (generates executable `ex16`)
- CMake target: `samrai_ex16`
- Dependencies: SAMRAI (MFEM_USE_SAMRAI=ON), MPI (MFEM_USE_MPI=ON)
- SAMRAI components: tbox, hier, xfer, pdat, math, mesh, geom, solv, algs, appu

## Implementation Steps

### 1. Create examples/samrai/ Directory Structure

Create the following directory and files:
- `examples/samrai/` - new directory
- `examples/samrai/CMakeLists.txt` - build configuration
- `examples/samrai/ex16.cpp` - main example source (merged MFEM + SAMRAI)
- `examples/samrai/LinAdv.h` - copy from SAMRAI source
- `examples/samrai/LinAdv.cpp` - copy from SAMRAI source
- `examples/samrai/LinAdvFort.h` - copy from SAMRAI source
- `examples/samrai/samrai_input.2d` - SAMRAI 2D configuration file
- `examples/samrai/samrai_input.3d` - SAMRAI 3D configuration file
- `examples/samrai/README` - brief documentation

### 2. Create examples/samrai/CMakeLists.txt

Follow the pattern from `examples/sundials/CMakeLists.txt`:
- Define SAMRAI_EXAMPLES_SRCS list with ex16.cpp
- Include project binary directory for mfem.hpp
- Call add_mfem_examples with "samrai_" prefix
- Remove prefix from executable name (target: samrai_ex16, executable: ex16)
- Link LinAdv.cpp as additional source to samrai_ex16 target
- Add test configuration for both 2D and 3D inputs if MFEM_ENABLE_TESTING

### 3. Update Main examples/CMakeLists.txt

Add conditional subdirectory inclusion:
- Add `if (MFEM_USE_SAMRAI) add_subdirectory(samrai) endif()`
- Place after other TPL example subdirectories (after moonolith, around line 264)

### 4. Implement examples/samrai/ex16.cpp

**Structure: Merge original examples/ex16.cpp with SAMRAI LinAdv**

#### 4.1 Include Headers
- MFEM headers: `#include "mfem.hpp"` for all MFEM functionality
- SAMRAI headers: SAMRAI_config.h, SAMRAIManager, SAMRAI_MPI, InputManager
- SAMRAI hierarchy: PatchHierarchy, CartesianGridGeometry
- SAMRAI integrators: TimeRefinementIntegrator, HyperbolicLevelIntegrator
- SAMRAI mesh: GriddingAlgorithm, BergerRigoutsos, StandardTagAndInitialize, CascadePartitioner
- SAMRAI visualization: VisItDataWriter
- Application: LinAdv.h

#### 4.2 Keep MFEM ConductionOperator Class
- Copy entire ConductionOperator class from original examples/ex16.cpp
- Implements nonlinear heat equation: du/dt = div((kappa + alpha*u) grad u)
- Provides Mult() and ImplicitSolve() for time integration
- No modifications needed - operates independently of SAMRAI

#### 4.3 Command-Line Arguments
Parse arguments for both solvers:
- MFEM options: mesh file, refinement levels, order, ODE solver, dt, t_final, alpha, kappa
- SAMRAI options: input file (default: samrai_input.3d), visualization intervals
- Shared options: visualization enable/disable
- Add `-i` flag for SAMRAI input file

#### 4.4 Initialization Sequence
1. Initialize MPI and SAMRAI:
   - `tbox::SAMRAI_MPI::init(&argc, &argv)`
   - `tbox::SAMRAIManager::initialize()` and `startup()`

2. Initialize MFEM (from original ex16.cpp):
   - Parse command-line options
   - Read mesh file
   - Refine mesh uniformly
   - Create H1_FECollection and FiniteElementSpace
   - Create GridFunction for temperature
   - Set initial conditions
   - Create ConductionOperator
   - Select and initialize ODESolver

3. Initialize SAMRAI (from LinAdv main.cpp):
   - Parse SAMRAI input file using InputManager
   - Extract configuration databases (Main, LinAdv, CartesianGeometry, etc.)
   - Create SAMRAI objects in order:
     - CartesianGridGeometry
     - PatchHierarchy
     - LinAdv model
     - HyperbolicLevelIntegrator
     - StandardTagAndInitialize
     - BergerRigoutsos
     - CascadePartitioner
     - GriddingAlgorithm
     - TimeRefinementIntegrator
     - VisItDataWriter (if HDF5 available)
   - Initialize hierarchy: `time_integrator->initializeHierarchy()`

#### 4.5 Alternating Time Loop
```
// Synchronize time parameters
mfem_time = 0.0
samrai_time = 0.0
final_time = min(mfem_t_final, samrai_t_final)

while (mfem_time < final_time && samrai_time < final_time):
  
  // MFEM heat equation step
  mfem_ode_solver->Step(u, mfem_time, mfem_dt)
  Update MFEM visualization if needed
  
  // SAMRAI advection step
  samrai_dt_new = samrai_time_integrator->advanceHierarchy(samrai_dt, rebalance)
  samrai_time += samrai_dt
  samrai_dt = samrai_dt_new
  Update SAMRAI visualization if needed
  
  Print status for both solvers
```

#### 4.6 Cleanup
- Deallocate SAMRAI objects (following LinAdv main.cpp cleanup)
- Delete MFEM objects (ConductionOperator, Mesh)
- Shutdown SAMRAI: `SAMRAIManager::shutdown()` and `finalize()`
- Finalize MPI: `SAMRAI_MPI::finalize()`

#### 4.7 Output and Visualization
- MFEM output: Write mesh and solution GridFunction files (ex16.mesh, ex16-final.gf)
- MFEM visualization: GLVis socketstream (if enabled)
- SAMRAI output: VisIt data files in specified directory
- Print summary: both solvers completed successfully

### 5. Copy LinAdv Application Files

Copy SAMRAI's LinAdv files to examples/samrai/:
- Source: `/home/vogl2/local/src/samrai/source/test/applications/LinAdv/`
- Files: LinAdv.h, LinAdv.cpp, LinAdvFort.h
- These provide SAMRAI's linear advection model implementation

### 6. Create examples/samrai/samrai_input.3d

Based on `sphere.3d.input` from SAMRAI examples:

**Main section:**
- dim = 3
- base_name = "ex16_samrai_3d"
- log_all_nodes = FALSE
- viz_dump_interval = 5 (output every 5 timesteps)
- viz_dump_dirname = "visit_samrai_3d"

**LinAdv section:**
- advection_velocity = 2.0, 1.0, 1.0
- godunov_order = 2
- corner_transport = "CORNER_TRANSPORT_1"
- data_problem = "SPHERE"
- Initial_data: radius, center, uval_inside, uval_outside
- Refinement_data: UVAL_GRADIENT and UVAL_SHOCK criteria
- Boundary_data: FLOW boundaries on all faces/edges/nodes

**CartesianGeometry section:**
- domain_boxes = [(0,0,0), (29,19,19)]
- x_lo = 0.0, 0.0, 0.0
- x_up = 30.0, 20.0, 20.0

**PatchHierarchy section:**
- max_levels = 3
- ratio_to_coarser: 2x refinement on each level
- largest_patch_size and smallest_patch_size constraints

**Algorithm sections:**
- BergerRigoutsos: efficiency tolerances
- GriddingAlgorithm: using defaults
- StandardTagAndInitialize: GRADIENT_DETECTOR tagging
- HyperbolicLevelIntegrator: CFL = 0.9
- TimeRefinementIntegrator: start_time = 0, end_time = 100, max_integrator_steps = 25

**LoadBalancer section:**
- Using default CascadePartitioner configuration

### 7. Create examples/samrai/samrai_input.2d

Based on `sphere.2d.input` from SAMRAI examples:

**Key differences from 3D:**
- dim = 2
- base_name = "ex16_samrai_2d"
- viz_dump_dirname = "visit_samrai_2d"
- advection_velocity = 2.0, 1.0 (2D vector)
- domain_boxes = [(0,0), (59,39)]
- x_up = 60.0, 40.0 (2D bounds)
- ratio_to_coarser: 2x refinement in 2D
- Initial sphere center adjusted for 2D domain
- Remove z-direction boundary conditions (only x/y faces and edges)
- Adjust patch sizes for 2D

### 8. Update examples/samrai/README

Documentation should include:

**Purpose:**
- Demonstrates MFEM-SAMRAI interoperability
- MFEM solves nonlinear heat equation
- SAMRAI solves linear advection with AMR
- Both run side-by-side in alternating time loop

**Build Requirements:**
- MFEM with SAMRAI and MPI support
- SAMRAI library with HDF5 (for visualization)
- Build command: `cmake -DMFEM_USE_SAMRAI=ON -DMFEM_USE_MPI=ON -DMFEM_FETCH_TPLS=ON`

**Running:**
- 2D case: `./ex16 -m ../data/inline-quad.mesh -i samrai_input.2d`
- 3D case: `./ex16 -m ../data/star.mesh -i samrai_input.3d`
- Command-line options for both MFEM and SAMRAI parameters

**Output:**
- MFEM: GLVis visualization, mesh/solution files
- SAMRAI: VisIt files in specified directory
- Instructions for viewing with GLVis and VisIt

**Notes:**
- This demonstrates basic interoperability without data coupling
- Foundation for future examples with actual data exchange
- Both solvers operate independently on their own domains

### 9. Testing Configuration

In CMakeLists.txt, add tests for both dimensions:
- `samrai_ex16_2d`: Run with 2D mesh and samrai_input.2d
- `samrai_ex16_3d`: Run with 3D mesh and samrai_input.3d
- Test options: reduced timesteps for faster testing
- Verify successful completion (exit code 0)
- Can run in serial or parallel with mpirun

## Critical Files

**Files to create:**
- `examples/samrai/CMakeLists.txt` - build configuration
- `examples/samrai/ex16.cpp` - merged MFEM + SAMRAI example
- `examples/samrai/LinAdv.h` - copied from SAMRAI
- `examples/samrai/LinAdv.cpp` - copied from SAMRAI
- `examples/samrai/LinAdvFort.h` - copied from SAMRAI
- `examples/samrai/samrai_input.2d` - SAMRAI 2D configuration
- `examples/samrai/samrai_input.3d` - SAMRAI 3D configuration
- `examples/samrai/README` - documentation

**Files to modify:**
- `examples/CMakeLists.txt` - add samrai subdirectory inclusion

**Files to reference:**
- `examples/ex16.cpp` - original MFEM heat equation (keep all MFEM code)
- `examples/sundials/CMakeLists.txt` - build pattern
- `/home/vogl2/local/src/samrai/source/test/applications/LinAdv/main.cpp` - SAMRAI reference
- `/home/vogl2/local/src/samrai/source/test/applications/LinAdv/example_inputs/sphere.2d.input`
- `/home/vogl2/local/src/samrai/source/test/applications/LinAdv/example_inputs/sphere.3d.input`

## Verification Steps

### Incremental Testing Approach

Test at each major milestone during implementation, not just at the end.

### 1. Build Requirements Check
- Verify MFEM build configuration includes MFEM_USE_SAMRAI and MFEM_USE_MPI
- Confirm SAMRAI_DIR points to valid SAMRAI installation
- Check that required TPLs (HYPRE, HDF5) are available if needed

### 2. After Directory and Build Setup
- Verify CMake detects the new examples/samrai subdirectory
- Check that samrai_ex16 target is recognized in the build system
- Confirm examples/CMakeLists.txt correctly includes samrai subdirectory

### 3. After Copying LinAdv Files
- Verify LinAdv.cpp, LinAdv.h, and LinAdvFort.h are in place
- Check LinAdv dependencies (especially HDF5 requirement for visualization)
- Confirm SAMRAI header files are accessible

### 4. After Creating Minimal ex16.cpp Skeleton
- Build with just includes and empty main function
- Catch any include path or header issues early
- Verify MFEM and SAMRAI headers can coexist without conflicts

### 5. After Full Implementation
- Build the complete samrai_ex16 target
- Verify compilation succeeds for both MFEM and SAMRAI components
- Confirm LinAdv.cpp links correctly with all required libraries

### 6. Runtime Testing - 2D Case
- Run with 2D mesh and samrai_input.2d configuration
- Verify both MFEM and SAMRAI initialize without errors
- Check that alternating timesteps complete successfully
- Confirm both solvers reach final time

### 7. Runtime Testing - 3D Case
- Run with 3D mesh and samrai_input.3d configuration
- Verify 3D SAMRAI hierarchy works correctly
- Confirm similar successful behavior as 2D case

### 8. Output Verification
- **MFEM**: Verify mesh and solution files are created and can be loaded in GLVis
- **SAMRAI**: Check VisIt files are generated in output directory with AMR hierarchy
- Confirm visualization files show expected physical behavior (heat diffusion, sphere advection)

### 9. Automated Test Suite
- Run test suite for both 2D and 3D configurations
- Verify all tests complete successfully with exit code 0

This incremental approach catches issues when they're easier to fix and provides confidence at each implementation step.

## Notes

**Integration approach:**
- This is a "loose coupling" - both solvers coexist but don't exchange data
- Demonstrates that MFEM and SAMRAI can be built and run together
- Foundation for future examples with actual data coupling

**MPI considerations:**
- SAMRAI requires MPI (uses SAMRAI_MPI wrapper)
- MFEM must be built with MFEM_USE_MPI=ON
- Example works in serial (1 process) or parallel
- Both libraries handle MPI independently

**Mesh compatibility:**
- MFEM uses unstructured mesh (read from file)
- SAMRAI uses structured Cartesian grid (defined in input file)
- No geometric relationship between the two meshes in this version

**Time integration:**
- Both solvers use their own timestep sizes
- No synchronization of timesteps required
- Each advances independently to its target time

**Dimensionality handling:**
- MFEM mesh file determines MFEM problem dimension
- SAMRAI input file determines SAMRAI problem dimension
- User responsible for matching dimensions appropriately

**Visualization:**
- MFEM: GLVis for interactive viewing
- SAMRAI: VisIt for AMR hierarchy visualization
- Both can be viewed simultaneously in separate tools

**Fortran dependencies:**
- LinAdv may use Fortran routines via LinAdvFort.h
- SAMRAI_LIBRARIES should include necessary Fortran objects
- If issues arise, check SAMRAI was built with Fortran support

**Future extensions:**
- Add data transfer between MFEM and SAMRAI
- Couple the physics problems (e.g., heat affects advection)
- Use SAMRAI's AMR to drive MFEM mesh refinement
- Solve on overlapping or coincident domains
