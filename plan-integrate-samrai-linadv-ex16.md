# Plan: Integrate SAMRAI LinAdv into MFEM ex16

## Context

The goal is to modify MFEM's `examples/ex16.cpp` to create a combined simulation that runs both:
1. MFEM's nonlinear heat equation (du/dt = C(u), existing functionality)
2. SAMRAI's linear advection problem (du/dt + div(a*u) = 0, new addition)

The two simulations will run in the same executable but operate independently with separate meshes. They will use sequential time-stepping (alternate advancing each solver) without data coupling. Each solver will produce its own visualization output.

This demonstrates interoperability between MFEM's finite element methods and SAMRAI's structured AMR approach for time-dependent PDEs.

## Requirements & Constraints

Based on user clarifications:
- **Integration level**: Combined simulation (both problems running together in one executable)
- **Dependency**: Always require SAMRAI (no conditional compilation needed)
- **Meshes**: Separate meshes - MFEM uses its Mesh, SAMRAI uses PatchHierarchy
- **Time-stepping**: Sequential - alternate advancing MFEM then SAMRAI at each iteration
- **Coupling**: None - problems are independent
- **Output**: Separate files - GLVis/VisIt for MFEM, VisIt for SAMRAI
- **SAMRAI components needed**: All core libraries already configured in FindSAMRAI.cmake

## Implementation Steps

### 1. Update examples/makefile
- Add conditional build support for ex16 when MFEM_USE_SAMRAI=YES
- Link SAMRAI libraries (SAMRAI_LIB from config) to ex16 build

### 2. Modify examples/ex16.cpp Structure

#### 2.1 Add SAMRAI Headers
- Include SAMRAI core headers at the top (after MFEM headers):
  - SAMRAI/SAMRAI_config.h
  - SAMRAI/tbox/SAMRAIManager.h, SAMRAI_MPI.h, InputManager.h, InputDatabase.h
  - SAMRAI/hier/PatchHierarchy.h, VariableDatabase.h
  - SAMRAI/geom/CartesianGridGeometry.h
  - SAMRAI/algs/TimeRefinementIntegrator.h, HyperbolicLevelIntegrator.h
  - SAMRAI/mesh/GriddingAlgorithm.h, BergerRigoutsos.h, StandardTagAndInitialize.h, CascadePartitioner.h
  - SAMRAI/appu/VisItDataWriter.h (if HAVE_HDF5)

#### 2.2 Include LinAdv Application Files
- Copy or reference LinAdv.h and LinAdv.cpp from SAMRAI examples
  - Option A: Copy files to examples/ directory
  - Option B: Add SAMRAI example path to include directories
- Include "LinAdv.h" header after SAMRAI headers

#### 2.3 Add SAMRAI Input File Parameter
- Add command-line option for SAMRAI input file (e.g., --samrai-input)
- Default to a simple embedded configuration or example file

### 3. Initialize SAMRAI Infrastructure in main()

#### 3.1 SAMRAI Initialization (before MFEM setup)
- Initialize SAMRAI_MPI and SAMRAIManager at program start
- Call SAMRAIManager::startup() at loop beginning

#### 3.2 Parse SAMRAI Input File
- Create InputDatabase and parse SAMRAI input file
- Extract Main, LinAdv, CartesianGeometry, PatchHierarchy, and integrator databases

#### 3.3 Create SAMRAI Objects
After MFEM objects are created (around line 185), add:
- Create CartesianGridGeometry for SAMRAI's Cartesian mesh
- Create PatchHierarchy with the geometry
- Create LinAdv model object
- Create HyperbolicLevelIntegrator with LinAdv model
- Create StandardTagAndInitialize for AMR error detection
- Create BergerRigoutsos box generator
- Create CascadePartitioner load balancer
- Create GriddingAlgorithm with error detector, box generator, and load balancer
- Create TimeRefinementIntegrator with hierarchy, level integrator, and gridding algorithm
- Register VisItDataWriter with LinAdv model (if HDF5 available)

#### 3.4 Initialize SAMRAI Hierarchy
- Call TimeRefinementIntegrator::initializeHierarchy() to set up AMR levels
- Get initial SAMRAI timestep

### 4. Modify Time-Stepping Loop

#### 4.1 Interleave MFEM and SAMRAI Steps
Current loop structure (lines 241-268) advances only MFEM. Modify to:
- Advance MFEM one step: `ode_solver->Step(u, t, dt)`
- Advance SAMRAI one step: `samrai_time_integrator->advanceHierarchy(samrai_dt, rebalance_coarsest)`
- Update parameters for both: `oper.SetParameters(u)` and SAMRAI internal updates
- Increment time by the minimum of dt and samrai_dt

#### 4.2 Synchronization Options
- Track separate time variables: `mfem_time` and `samrai_time`
- Advance sequentially, allowing each to progress at its own rate
- Or: use common timestep (min of both dt's) for simpler synchronization

#### 4.3 Visualization Updates
- Keep existing MFEM GLVis output (lines 255-258)
- Add SAMRAI VisItDataWriter calls when SAMRAI visualization is due
- Check both `ti % vis_steps` for MFEM and SAMRAI's viz_dump_interval

### 5. Finalize and Cleanup

#### 5.1 Save Final Solutions
- Keep existing MFEM solution save (line 273-276)
- SAMRAI automatically handles its output through VisItDataWriter

#### 5.2 Cleanup
- Deallocate SAMRAI objects (use reset() for shared_ptrs)
- Deallocate LinAdv model (delete)
- Call SAMRAIManager::shutdown() at loop end
- Call SAMRAIManager::finalize() before program exit (after MFEM cleanup)

### 6. Create SAMRAI Input File
- Create `examples/ex16_samrai.input` with configuration for LinAdv problem
- Base it on SAMRAI's sphere.3d.input but adapt for ex16's domain
- Include: advection velocity, godunov_order, boundary conditions, refinement criteria
- Configure Main section: dimension, visualization, restart parameters
- Configure PatchHierarchy: domain size, refinement ratios
- Configure TimeRefinementIntegrator: start_time, end_time, max_integrator_steps

### 7. Update Build System
- Ensure examples makefile links SAMRAI_LIB when building ex16
- May need to add SAMRAI include paths to CXXFLAGS for examples

## Critical Files

Files to modify:
- `examples/ex16.cpp` - main implementation (all changes above)
- `examples/makefile` - build system updates for SAMRAI linking

Files to create:
- `examples/ex16_samrai.input` - SAMRAI configuration file
- `examples/LinAdv.h` and `examples/LinAdv.cpp` - copy from SAMRAI or add include path

Files to reference:
- `/home/vogl2/local/src/samrai/source/test/applications/LinAdv/main.cpp` - reference implementation
- `/home/vogl2/local/src/samrai/source/test/applications/LinAdv/LinAdv.{h,cpp}` - model class
- `/home/vogl2/local/src/samrai/source/test/applications/LinAdv/example_inputs/sphere.3d.input` - input file template
- `config/cmake/modules/FindSAMRAI.cmake` - SAMRAI component libraries

## Reference Patterns from SAMRAI's LinAdv main.cpp

Key initialization sequence (lines 181-415 in SAMRAI main.cpp):
1. SAMRAI_MPI::init() and SAMRAIManager::initialize()
2. Loop with SAMRAIManager::startup()/shutdown()
3. Parse input file with InputDatabase and InputManager
4. Create geometry → hierarchy → model → integrators → gridding
5. initializeHierarchy() before time loop
6. Loop: advanceHierarchy() + writePlotData() + writeRestartFile()
7. Cleanup and SAMRAIManager::finalize()

Key objects and their roles:
- **CartesianGridGeometry**: Defines structured Cartesian coordinate system
- **PatchHierarchy**: Container for AMR levels and patch data
- **LinAdv**: Application-specific patch operations (fluxes, boundary conditions, tagging)
- **HyperbolicLevelIntegrator**: Manages explicit time integration per level
- **TimeRefinementIntegrator**: Coordinates time stepping across AMR levels
- **GriddingAlgorithm**: Manages regridding and load balancing
- **VisItDataWriter**: Outputs visualization files

## Verification Steps

1. **Build**: Compile ex16 with MFEM_USE_SAMRAI=ON
   - Verify all SAMRAI headers are found
   - Verify all SAMRAI libraries link correctly

2. **Run**: Execute modified ex16
   - Should accept both MFEM and SAMRAI command-line parameters
   - Should initialize both MFEM mesh and SAMRAI hierarchy
   - Should advance both simulations through time loop
   - Should produce MFEM output: ex16.mesh, ex16-init.gf, ex16-final.gf
   - Should produce SAMRAI output: VisIt files in specified directory

3. **Verify Outputs**: Check that both solvers produce valid results
   - MFEM: View with `glvis -m ex16.mesh -g ex16-final.gf`
   - SAMRAI: View with VisIt using generated .visit files
   - Compare timestep counts and final times

4. **Test Parameters**: Try different options
   - Different timesteps for each solver
   - Different mesh refinement levels
   - Different visualization intervals

## Notes

- **Namespace handling**: SAMRAI uses `using namespace SAMRAI;` while MFEM uses explicit namespacing. Be careful with conflicts (e.g., `Vector` exists in both).

- **MPI considerations**: Both MFEM and SAMRAI can use MPI. SAMRAI initializes its own SAMRAI_MPI which wraps MPI_COMM_WORLD. If MFEM also uses MPI, ensure compatible initialization.

- **Memory management**: SAMRAI uses shared_ptr extensively, MFEM uses raw pointers in examples. LinAdv model uses `new`/`delete`.

- **Dimension abstraction**: SAMRAI uses tbox::Dimension objects, MFEM uses integer dim. Create Dimension from MFEM's dim when initializing SAMRAI objects.

- **Input file location**: Store ex16_samrai.input in examples/ and provide default path in code.

- **Fortran dependencies**: LinAdv may require Fortran routines (LinAdvFort.h). Ensure these are available or handle conditionally.

- **Simplified first version**: For initial implementation, consider starting with 2D only and minimal SAMRAI features to validate the integration approach before adding full 3D and advanced features.

- **Alternative: Separate executable**: If integration into ex16 proves too complex, consider creating a new example (e.g., ex16s.cpp) specifically for SAMRAI integration, leaving ex16.cpp unchanged.
