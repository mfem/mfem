// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details

#ifndef SPDE_SOLVERS_HPP
#define SPDE_SOLVERS_HPP

#include "mfem.hpp"
//#include "boundary.hpp"

#include <unordered_map>
#include <ostream>

namespace mfem {
namespace materials {

enum BoundaryType { kNeumann, kDirichlet, kRobin, kPeriodic, kUndefined };

struct Boundary {
  Boundary() = default;

  /// Print the information specifying the boundary conditions.
  void PrintInfo(std::ostream &os = mfem::out) const;

  /// Verify that all defined boundaries are actually defined on the mesh, i.e.
  /// if the keys of boundary attributes appear in the boundary attributes of
  /// the mesh.
  void VerifyDefinedBoundaries(const Mesh& mesh) const;

  /// Computes the error for each defined boundary attribute by calling back to
  /// the IntegrateBC.
  void ComputeBoundaryError(const ParGridFunction& solution);

  /// Helper function to compute the coefficients alpha, beta, gamma (see in
  /// `IntegrateBC`) for a given boundary attribute.
  void UpdateIntegrationCoefficients(int i, double& alpha, double& beta,
                                     double& gamma);

  /// Add a homogeneous boundary condition to the boundary.
  void AddHomogeneousBoundaryCondition(int boundary, BoundaryType type);

  /// Add a inhomogeneous Dirichlet boundary condition to the boundary.
  void AddInhomogeneousDirichletBoundaryCondition(int boundary,
                                                  double coefficient);

  /// Set the robin coefficient for the boundary.
  void SetRobinCoefficient(double coefficient);

  /// Map to assign homogeneous boundary conditions to defined boundary types.
  std::map<int, BoundaryType> boundary_attributes;
  /// Coefficient for inhomogeneous Dirichlet boundary conditions.
  std::map<int, double> dirichlet_coefficients;
  /// Coefficient for Robin boundary conditions (n.grad(u) + coeff u = 0) on
  /// defined boundaries.
  double robin_coefficient = 1.0;
};

/// IntegrateBC function from ex27p.cpp. For boundary verification.
/// Compute the average value of alpha*n.Grad(sol) + beta*sol over the boundary
/// attributes marked in bdr_marker. Also computes the L2 norm of
/// alpha*n.Grad(sol) + beta*sol - gamma over the same boundary.
double IntegrateBC(const ParGridFunction &x, const Array<int> &bdr,
                   double alpha, double beta, double gamma,
                   double &glb_err);

}
}




namespace mfem {
namespace materials {

/// Solver for the SPDE method based on a rational approximation with the AAA
/// algorithm. The SPDE method is described in the paper
/// Lindgren, F., Rue, H., Lindström, J. (2011). An explicit link between
/// Gaussian fields and Gaussian Markov random fields: the stochastic partial
/// differential equation approach. Journal of the Royal Statistical Society:
/// Series B (Statistical Methodology), 73(4), 423–498.
/// https://doi.org/10.1111/j.1467-9868.2011.00777.x
///
/// The solver solves the SPDE problem defined as
/// (A)^-\alpha u = b
/// where A is
/// A = div ( Theta(x) grad + Id ) u(x)
/// and \alpha is given as
/// \alpha = (2 nu + dim) / 2.
/// Theta (anisotropy tensor) and nu (smoothness) can be specified in the
/// constructor. Traditionally, the SPDE method requires the specification of
/// a white noise right hands side. SPDESolver accepts arbitrary right hand
/// sides but the solver has only been tested with white noise.
class SPDESolver {
public:
  /// Constructor.
  /// @param diff_coefficient The diffusion coefficient \Theta.
  /// @param nu The coefficient nu, smoothness of the solution.
  /// @param bc Boundary conditions.
  /// @param fespace Finite element space.
  SPDESolver(MatrixConstantCoefficient &diff_coefficient, double nu,
             const Boundary& bc, ParFiniteElementSpace *fespace);

  /// Constructor.
  /// @param diff_coefficient The diffusion coefficient \Theta.
  /// @param nu The coefficient nu, smoothness of the solution.
  /// @param fespace Finite element space.
  SPDESolver(MatrixConstantCoefficient &diff_coefficient, double nu,
             ParFiniteElementSpace *fespace);

  /// Solve the SPDE for a given right hand side b. May alter b if
  /// the exponent (alpha) is larger than 1. We avoid copying be default. If you
  /// need b later on, make a copy of it before calling this function.
  void Solve(ParLinearForm &b, ParGridFunction &x);

private:
  /// The rational approximation of the SPDE results in multiple
  /// reactio-diffusion PDEs that need to be solved. This call solves the PDE
  /// (div \Theta grad + \alpha I)^exponent x = \beta b.
  void Solve(const ParLinearForm &b, ParGridFunction &x, double alpha, double beta,
             int exponent = 1);

  /// Lift the solution to satisfy the inhomogeneous boundary conditions.
  void LiftSolution(ParGridFunction &x);

  // Each PDE gives rise to a linear system. This call solves the linear system
  // with PCG and Boomer AMG preconditioner.
  void SolveLinearSystem();

  /// Activate repeated solve capabilities. E.g. if the PDE is of the form
  /// A^N x = b. This method solves the PDE A x = b for the first time, and
  /// then uses the solution as RHS for the next solve and so forth.
  void ActivateRepeatedSolve() { repeated_solve_ = true; }

  /// Single solve only.
  void DeactivateRepeatedSolve() { repeated_solve_ = false; }

  /// Writes the solution of the PDE from the previous call to Solve() to the
  /// linear from b (with appropriate transformations).
  void UpdateRHS(ParLinearForm &b);

  // Compute the coefficients for the rational approximation of the solution.
  void ComputeRationalCoefficients(double exponent);

  // Bilinear forms and corresponding matrices for the solver.
  ParBilinearForm k_;
  ParBilinearForm m_;
  HypreParMatrix stiffness_;
  HypreParMatrix mass_bc_;

  // Transformation matrices (needed to construct the linear systems and
  // solutions)
  const SparseMatrix *restriction_matrix_;
  const Operator *prolongation_matrix_;

  // Members to solve the linear system.
  Vector X_;
  Vector B_;
  HypreParMatrix *Op_;

  // Information of the finite element space.
  Array<int> ess_tdof_list_;
  ParFiniteElementSpace *fespace_ptr_;

  // Boundary conditions
  const Boundary bc_;
  Array<int> dbc_marker_; // Markers for Dirichlet boundary conditions.
  Array<int> rbc_marker_; // Markers for Robin boundary conditions.

  // Coefficients for the rational approximation of the solution.
  Array<double> coeffs_;
  Array<double> poles_;

  // Exponents of the operator
  double nu_ = 0.0;
  double alpha_ = 0.0;
  int integer_order_of_exponent_ = 0;

  // Member to switch to repeated solve capabilities.
  bool repeated_solve_ = false;
  bool integer_order_ = false;
  bool apply_lift_ = false;
};

} // namespace materials
} // namespace mfem

#endif // MATERIALS_SOLVERS_HPP
