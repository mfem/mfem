# Discontinuous Galerkin MFEM mini-app using partial assembly

This mini-app demonstrates using partial-assembly to solve hyperbolic
conservation laws using discontinuous Galerkin methods and explicit
time integration.

The main object is a `PartialAssembly` object, which provides:

- local interpolation and differentiation operators (including at faces)
- face access to metric terms at quadrature data

On top of this object, there are several operators that are provided
(but more are possible). All of these operators allow for a "coefficient"
to be evaluated at quadrature points, which is referred to as D. These
operators are:

- `BtDB`, which represents mass or source terms with coefficient `D`
- `GtDB`, which represents dot product with the gradient of test functions
- `BtDB_face`, which represents integrating against test functions on faces

Using any of these operators simply requires templating on a class `D` which
provides an operator to evaluate the coefficient at a quadrature point.

There is a `ConservationLaw` object which is built on these three operators.
Given a flux function and numerical flux function, it will assemble the 
corresponding DG residual.

Examples are provided in the `apps` directory for solving the scalar advection
equation, Burgers' equation and the Euler equations of gas dynamics.

This mini-app is still incomplete. Improvements are needed for:

[ ] Handing of mixed meshes
[ ] AMR and non-conforming meshes
[ ] Second-order operators and viscous terms
