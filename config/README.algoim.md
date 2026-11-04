# ALGOIM cut quadrature

MFEM uses the unmodified header-only ALGOIM repository at revision
`da1d81499608e1d499695d255f0233140b8c81e8`. Upstream HEAD could not be checked
because GitHub DNS was unavailable; this baseline is not claimed to be latest.
No dependency patch, header replacement, or solver interposition is required.

## Configuration and selection

```sh
cmake -S . -B build -DMFEM_USE_ALGOIM=ON -DALGOIM_DIR=/path/to/algoim \
  -DMFEM_USE_LAPACK=OFF -DMFEM_USE_LAPACKE=OFF
```

This builds general implicit-domain quadrature with **no LAPACK dependency**.
For polynomial quadrature, additionally enable `MFEM_USE_LAPACKE=ON`. Set
`LAPACKE_DIR` or `LAPACKE_INCLUDE_DIRS` and `LAPACKE_LIBRARIES` as needed.
The provider must supply a standard `lapacke.h`, `LAPACKE_dgesvd`, and
`LAPACKE_dggevx`, with LP64 integer ABI and the appropriate LAPACK/BLAS libraries.
A separate LAPACKE library or a provider such as OpenBLAS exporting those
symbols is supported. Explicitly enabling an unusable provider is a configure
error, not a request to silently fall back.

`MFEM_USE_LAPACK` and `MFEM_USE_LAPACKE` are independent: enabling Fortran LAPACK
alone does not select polynomial quadrature. Both ALGOIM implementations
currently require double-precision MFEM. Neither requires Blitz++.

GNU make supports the same `MFEM_USE_ALGOIM` and `MFEM_USE_LAPACKE` switches
(with `YES`/`NO` values). Set `ALGOIM_DIR` to the checkout; when using LAPACKE,
set `LAPACKE_OPT` for includes and `LAPACKE_LIB` for LAPACKE/LAPACK/BLAS linkage.
No ALGOIM archive needs to be built.

ALGOIM-specific MFEM code lives in one optional public adapter header,
`fem/cut_quadrature_algoim.hpp`, and one implementation source,
`fem/cut_quadrature_algoim.cpp`. The source keeps all backend helpers private
and guards polynomial code and headers with `MFEM_USE_LAPACKE`, preserving
the general-only build's independence from LAPACK/LAPACKE.

| Factory name | Behavior |
| --- | --- |
| `algoim` | Prefer polynomial quadrature for polynomial inputs when LAPACKE is enabled; otherwise use general quadrature |
| `algoim-general` | Force general implicit-domain quadrature |
| `algoim-polynomial` | Force polynomial quadrature; available only with LAPACKE |

Selection considers active constraints, not unused entries in the level-set
array. Numerical failures never trigger an algorithm switch. Analytic functions
are not implicitly approximated by polynomials. The factory returns null for an
unavailable name; `GetCutQuadratureBackends()` lists compiled choices.

## Inputs and domains

Include `mfem.hpp` or `fem/cut_quadrature.hpp` for the common API. Create a
backend, construct level sets, and call `CutQuadratureGenerator::Generate`.
The backend and polynomial FE are borrowed and must outlive their users;
coefficient vectors are copied. There is no per-element cache or batch API.

Both algorithms support square/cube reference elements and whole-element,
positive/negative volume, and single-interface surface requests. General
quadrature supports one active level set. Polynomial quadrature additionally
supports two polynomial sign constraints for volume integration: both interfaces
partition the hierarchy before nodes are selected by the conjunction of signs.
The adapter uses upstream Bernstein degree reduction at roundoff tolerance
before building the polynomial hierarchy, retaining at least linear storage in
each coordinate. This removes degree-elevation artifacts in resultants without
changing the input FE space or the requested integration order. Already-positive
MFEM coefficients are copied and permuted directly, not numerically reprojected.
For 3D polynomials independent of a common coordinate, MFEM generates a 2D
rule and tensors it with a one-dimensional rule. Independence is checked at
coefficient-roundoff scale; this avoids an upstream degree-reduction assertion
for extruded multi-polynomial geometries without modifying the dependency.
Mixed equality/inequality requests, two equality constraints, more than two
constraints, facets, simplices, and prescribed nodes remain unsupported.
In particular, a codimension-two intersection is not a union of two surfaces.

`PolynomialLevelSet` works with both algorithms. The general path performs
MFEM-owned scalar-templated tensor Bernstein evaluation, including derivative
polynomials, without ALGOIM's LAPACKE-dependent Bernstein header. Data is local
to the generation call.

For analytic functions, explicitly include `fem/cut_quadrature_algoim.hpp` and
use `AlgoimFunctionLevelSet<Dimension, Functor>`. The functor must provide
scalar-templated `operator()` and `grad()` accepting
`algoim::uvector<Scalar, Dimension>` for both real and `algoim::Interval` scalars.
For example:

```cpp
struct Wave
{
   template <typename Scalar>
   Scalar operator()(const algoim::uvector<Scalar, 2> &point) const
   {
      using std::sin;
      return point(1) - sin(point(0));
   }

   template <typename Scalar>
   algoim::uvector<Scalar, 2> grad(
      const algoim::uvector<Scalar, 2> &point) const
   {
      using std::cos;
      algoim::uvector<Scalar, 2> result;
      result(0) = -cos(point(0));
      result(1) = Scalar(1);
      return result;
   }
};

mfem::AlgoimFunctionLevelSet<2, Wave> level_set(Wave{});
```

The wrapper owns the functor, but referenced captures must remain alive. Plain
`FunctionLevelSet` callbacks and whole-element scalar bounds do not supply the
required interval value/gradient arithmetic and are rejected. Intervals carry
first-order Taylor data with bounded remainders; corner samples and finite
differences do not satisfy that contract. Callbacks must not invoke nested
interval algorithms. The general backend restores ALGOIM's thread-local interval
radius state after each call, including on exceptions.

## Order, mapping, and limitations

Both algorithms use `order / 2 + 1` one-dimensional points. General quadrature
currently supports nominal orders 0--19, following the documented upstream
limit of 10 points; polynomial quadrature supports 0--199. Orders are nominal,
not certificates of exactness on curved domains, and are never silently reduced.

Reference weights and conormals belong to `CutQuadrature`; physical/integrator
weights belong to `MappedCutQuadrature`. Conormals point toward increasing
original level-set values, even when the general algorithm negates a function
to integrate its positive phase. Positive-phase outward normals have the
opposite sign. Mapping supports embedded elements through the metric and
Nanson factor. Integrator weights use the parent element's measure.

Known zero equality constraints, coincident facets, singular gradients at
returned nodes, and singular mappings are checked. This does not globally
certify regularity. General ALGOIM has an internal subdivision limit and fallback
without a convergence-status output. Unmodified polynomial ALGOIM checks some
LAPACKE failures with assertions. MFEM translates catchable C++ exceptions and
invalid output into reports, but does not claim to intercept upstream assertions
or detect every internal convergence failure.

The legacy cut API, old ALGOIM adapter, and old moment-fitting engine are gone.
A new moment backend remains future work.

## Examples and validation

`ex38` integrates a circle/sphere. `lsf_integral` and distributed `lsf_integralp`
accept `-cq algoim`, `-cq algoim-general`, or `-cq algoim-polynomial`.
Use `-ls 2` for a unit-box plane (volume 0.37, surface measure 1) or `-ls 3` for
intersection of two positive half-spaces (volume 0.42; LAPACKE required).
Use `-n 1` with four MPI ranks to exercise empty partitions.

The `[CutQuadrature]` tests cover analytic interval callbacks, polynomial basis
conversion, explicit/automatic dispatch, independent algorithms, two-level-set
phase conservation and first moments, curved intersections, mappings, failure
cleanup, and unsupported requests. Registered MPI tests cover 1, 2, and 4 ranks,
empty partitions, and collective failures. No MPI calls occur in the backend.
