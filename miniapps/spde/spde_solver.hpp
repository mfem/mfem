// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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

#include <ostream>
#include <unordered_map>
#include "mfem.hpp"

namespace mfem
{
namespace spde
{

enum class BoundaryType { kNeumann, kDirichlet, kRobin, kPeriodic, kUndefined };

struct Boundary
{
   Boundary() = default;

   /// Print the information specifying the boundary conditions.
   void PrintInfo(std::ostream &os = mfem::out) const;

   /// Verify that all defined boundaries are actually defined on the mesh, i.e.
   /// if the keys of boundary attributes appear in the boundary attributes of
   /// the mesh.
   void VerifyDefinedBoundaries(const Mesh &mesh) const;

   /// Computes the error for each defined boundary attribute by calling back to
   /// the IntegrateBC.
   void ComputeBoundaryError(const ParGridFunction &solution);

   /// Helper function to compute the coefficients alpha, beta, gamma (see in
   /// `IntegrateBC`) for a given boundary attribute.
   void UpdateIntegrationCoefficients(int i, real_t &alpha, real_t &beta,
                                      real_t &gamma);

   /// Add a homogeneous boundary condition to the boundary.
   void AddHomogeneousBoundaryCondition(int boundary, BoundaryType type);

   /// Add a inhomogeneous Dirichlet boundary condition to the boundary.
   void AddInhomogeneousDirichletBoundaryCondition(int boundary,
                                                   real_t coefficient);

   /// Set the robin coefficient for the boundary.
   void SetRobinCoefficient(real_t coefficient);

   /// Map to assign homogeneous boundary conditions to defined boundary types.
   std::map<int, BoundaryType> boundary_attributes;
   /// Coefficient for inhomogeneous Dirichlet boundary conditions.
   std::map<int, real_t> dirichlet_coefficients;
   /// Coefficient for Robin boundary conditions (n.grad(u) + coeff u = 0) on
   /// defined boundaries.
   real_t robin_coefficient = 1.0;
};

/// IntegrateBC function from ex27p.cpp. For boundary verification.
/// Compute the average value of alpha*n.Grad(sol) + beta*sol over the boundary
/// attributes marked in bdr_marker. Also computes the L2 norm of
/// alpha*n.Grad(sol) + beta*sol - gamma over the same boundary.
real_t IntegrateBC(const ParGridFunction &x, const Array<int> &bdr,
                   real_t alpha, real_t beta, real_t gamma, real_t &glb_err);

/// Solver for the SPDE method based on a rational approximation with the AAA
/// algorithm. The SPDE method is described in the paper
/// Lindgren, F., Rue, H., Lindström, J. (2011). An explicit link between
/// Gaussian fields and Gaussian Markov random fields: the stochastic partial
/// differential equation approach. Journal of the Royal Statistical Society:
/// Series B (Statistical Methodology), 73(4), 423–498.
/// https://doi.org/10.1111/j.1467-9868.2011.00777.x
///
/// The solver solves the SPDE problem defined as
/// (A)^-alpha u = b
/// where A is
/// A = div ( Theta(x) grad + Id ) u(x)
/// and alpha is given as
/// alpha = (2 nu + dim) / 2.
/// Theta (anisotropy tensor) and nu (smoothness) can be specified in the
/// constructor. Traditionally, the SPDE method requires the specification of
/// a white noise right hands side. SPDESolver accepts arbitrary right hand
/// sides but the solver has only been tested with white noise.
class SPDESolver
{
public:
   /// Constructor.
   /// @param nu The coefficient nu, smoothness of the solution.
   /// @param bc Boundary conditions.
   /// @param fespace Finite element space.
   /// @param l1 Correlation length in x
   /// @param l2 Correlation length in y
   /// @param l3 Correlation length in z
   /// @param e1 Rotation angle in x
   /// @param e2 Rotation angle in y
   /// @param e3 Rotation angle in z
   SPDESolver(real_t nu, const Boundary &bc, ParFiniteElementSpace *fespace,
              real_t l1 = 0.1, real_t l2 = 0.1, real_t l3 = 0.1,
              real_t e1 = 0.0, real_t e2 = 0.0, real_t e3 = 0.0);

   /// Destructor.
   ~SPDESolver();

   /// Solve the SPDE for a given right hand side b. May alter b if
   /// the exponent (alpha) is larger than 1. We avoid copying by default. If
   /// you need b later on, make a copy of it before calling this function.
   void Solve(ParLinearForm &b, ParGridFunction &x);

   /// Set up the random field generator. If called more than once it resets the
   /// generator.
   void SetupRandomFieldGenerator(int seed=0);

   /// Generate a random field. Calls back to solve but generates the stochastic
   /// load b internally.
   void GenerateRandomField(ParGridFunction &x);

   /// Construct the normalization coefficient eta of the white noise right hands
   /// side.
   static real_t ConstructNormalizationCoefficient(real_t nu, real_t l1,
                                                   real_t l2, real_t l3,
                                                   int dim);

   /// Construct the second order tensor (matrix coefficient) Theta from the
   /// equation R^T(e1,e2,e3) diag(l1,l2,l3) R (e1,e2,e3).
   static DenseMatrix ConstructMatrixCoefficient(real_t l1, real_t l2, real_t l3,
                                                 real_t e1, real_t e2, real_t e3,
                                                 real_t nu, int dim);

   /// Set the print level
   void SetPrintLevel(int print_level) {print_level_ = print_level;}

private:
   /// The rational approximation of the SPDE results in multiple
   /// reaction-diffusion PDEs that need to be solved. This call solves the PDE
   /// (div Theta grad + alpha I)^exponent x = beta b.
   void Solve(const ParLinearForm &b, ParGridFunction &x, real_t alpha,
              real_t beta, int exponent = 1);

   /// Lift the solution to satisfy the inhomogeneous boundary conditions.
   void LiftSolution(ParGridFunction &x);

   // Each PDE gives rise to a linear system. This call solves the linear system
   // with PCG and Boomer AMG preconditioner.
   void SolveLinearSystem(const HypreParMatrix *Op);

   /// Activate repeated solve capabilities. E.g. if the PDE is of the form
   /// A^N x = b. This method solves the PDE A x = b for the first time, and
   /// then uses the solution as RHS for the next solve and so forth.
   void ActivateRepeatedSolve() { repeated_solve_ = true; }

   /// Single solve only.
   void DeactivateRepeatedSolve() { repeated_solve_ = false; }

   /// Writes the solution of the PDE from the previous call to Solve() to the
   /// linear from b (with appropriate transformations).
   void UpdateRHS(ParLinearForm &b) const;

   // Compute the coefficients for the rational approximation of the solution.
   void ComputeRationalCoefficients(real_t exponent);

   // Bilinear forms and corresponding matrices for the solver.
   ParBilinearForm k_;
   ParBilinearForm m_;
   HypreParMatrix stiffness_;
   HypreParMatrix mass_bc_;

   // Transformation matrices (needed to construct the linear systems and
   // solutions)
   const SparseMatrix *restriction_matrix_ = nullptr;
   const Operator *prolongation_matrix_ = nullptr;

   // Members to solve the linear system.
   Vector X_;
   Vector B_;

   // Information of the finite element space.
   Array<int> ess_tdof_list_;
   ParFiniteElementSpace *fespace_ptr_;

   // Boundary conditions
   const Boundary &bc_;
   Array<int> dbc_marker_;  // Markers for Dirichlet boundary conditions.
   Array<int> rbc_marker_;  // Markers for Robin boundary conditions.

   // Coefficients for the rational approximation of the solution.
   Array<real_t> coeffs_;
   Array<real_t> poles_;

   // Exponents of the operator
   real_t nu_ = 0.0;
   real_t alpha_ = 0.0;
   int integer_order_of_exponent_ = 0;

   // Correlation length
   real_t l1_ = 0.1;
   real_t l2_ = 0.1;
   real_t l3_ = 0.1;
   real_t e1_ = 0.0;
   real_t e2_ = 0.0;
   real_t e3_ = 0.0;

   // Print level
   int print_level_ = 1;

   // Member to switch to repeated solve capabilities.
   bool repeated_solve_ = false;
   bool integer_order_ = false;
   bool apply_lift_ = false;

   WhiteGaussianNoiseDomainLFIntegrator *integ = nullptr;
   ParLinearForm * b_wn = nullptr;
};

}  // namespace spde
}  // namespace mfem

#endif  // SPDE_SOLVERS_HPP
