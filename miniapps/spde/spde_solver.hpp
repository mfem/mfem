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

#include "boundary.hpp"
#include "mfem.hpp"

namespace mfem {
namespace spde {

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
  SPDESolver(double nu, const Boundary &bc, ParFiniteElementSpace *fespace,
             double l1 = 0.1, double l2 = 0.1, double l3 = 0.1, double e1 = 0.0,
             double e2 = 0.0, double e3 = 0.0);

  /// Destructor.
  ~SPDESolver(); 

  /// Solve the SPDE for a given right hand side b. May alter b if
  /// the exponent (alpha) is larger than 1. We avoid copying be default. If you
  /// need b later on, make a copy of it before calling this function.
  void Solve(ParLinearForm &b, ParGridFunction &x);

  /** Set up the random field generator. If called more than once it resets the generator */
  void SetupRandomFieldGenerator(int seed=0);

  /// Generate a random field. Calls back to solve but generates the stochastic
  /// load b internally. 
  void GenerateRandomField(ParGridFunction &x);

  /// Construct the normalization coefficient eta of the white noise right hands
  /// side.
  static double ConstructNormalizationCoefficient(double nu, double l1,
                                                  double l2, double l3,
                                                  int dim);

  /// Construct the second order tensor (matrix coefficient) Theta from the
  /// equation R^T(e1,e2,e3) diag(l1,l2,l3) R (e1,e2,e3).
  static DenseMatrix ConstructMatrixCoefficient(double l1, double l2, double l3,
                                                double e1, double e2, double e3,
                                                double nu, int dim);

 private:
  /// The rational approximation of the SPDE results in multiple
  /// reactio-diffusion PDEs that need to be solved. This call solves the PDE
  /// (div \Theta grad + \alpha I)^exponent x = \beta b.
  void Solve(const ParLinearForm &b, ParGridFunction &x, double alpha,
             double beta, int exponent = 1);

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
  const Boundary &bc_;
  Array<int> dbc_marker_;  // Markers for Dirichlet boundary conditions.
  Array<int> rbc_marker_;  // Markers for Robin boundary conditions.

  // Coefficients for the rational approximation of the solution.
  Array<double> coeffs_;
  Array<double> poles_;

  // Exponents of the operator
  double nu_ = 0.0;
  double alpha_ = 0.0;
  int integer_order_of_exponent_ = 0;

  // Correlation length
  double l1_ = 0.1;
  double l2_ = 0.1;
  double l3_ = 0.1;
  double e1_ = 0.0;
  double e2_ = 0.0;
  double e3_ = 0.0;

  // Member to switch to repeated solve capabilities.
  bool repeated_solve_ = false;
  bool integer_order_ = false;
  bool apply_lift_ = false;

  WhiteGaussianNoiseDomainLFIntegrator *integ = nullptr;
  ParLinearForm * b = nullptr;
};

}  // namespace spde
}  // namespace mfem

#endif  // SPDE_SOLVERS_HPP
