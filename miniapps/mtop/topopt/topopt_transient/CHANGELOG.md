# Changelog

## Current Working State

The active transient topology-optimization path is:

- `TopOptTransient.cpp`
- `ProblemSpecification.hpp`
- `ObjectiveFunctional.hpp`
- `ElastodynamicsSolver.hpp`
- `test_adjoint_verification.cpp`

The older split headers `BoundaryLoadSpec.hpp`, `MaterialParams.hpp`, and
`TransientTopOptConfig.hpp` were consolidated into `ProblemSpecification.hpp`.
That consolidation is intentional for now so the problem layer stays easy to
read while the interface is still evolving.

## Recent Structural Changes

- Added `-problem` selection in `TopOptTransient`.
- Added the `wave`, `band-waveguide`, and `cantilever-compliance` problem
  definitions.
- Moved material, load, damping, config, and concrete problem definitions into
  `ProblemSpecification.hpp`.
- Added `BoundaryLoadSpec::domain_load` so the same solver path supports both
  boundary tractions and body-force loads.
- Added rectangular indicators/load coefficients for generated waveguide
  experiments.
- Added a modulated Gaussian load profile and symmetric receiver support for
  the band-waveguide experiment.
- Extended the generated band-waveguide to an `8 x 0.5` domain with a longer
  central propagation/filtering path.
- Raised the band-waveguide carrier frequency to sharpen the target wavelength
  and encourage more repeated material interfaces.
- Added `DampingField` to own damping coefficients and absorbing-boundary
  impedance.
- Made the coordinate sponge damping selectable by side so different generated
  problems can reuse the same damping machinery.
- Added `TransientDesignSolver`, which exposes the optimization sequence as:
  `FilterFSolve`, `PhysicsFSolve`, `PhysicsASolve`, and `FilterASolve`.
- Updated solver prints, including `Inverse lumped mass range: [min, max]`.
- Extended adjoint verification to cover lumped and consistent mass design
  gradients, clamped-BC projection, and the compliance objective path.

## Documentation

`README.md` is now the canonical directory documentation.
