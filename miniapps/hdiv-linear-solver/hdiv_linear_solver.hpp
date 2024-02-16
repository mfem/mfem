// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef HDIV_LINEAR_SOLVER_HPP
#define HDIV_LINEAR_SOLVER_HPP

#include "mfem.hpp"
#include "change_basis.hpp"
#include <memory>

namespace mfem
{

/// @brief Solve the H(div) saddle-point system using MINRES with matrix-free
/// block-diagonal preconditioning.
///
/// See HdivSaddlePointSolver::HdivSaddlePointSolver for the problem
/// description.
class HdivSaddlePointSolver : public Solver
{
public:
   /// Which type of saddle-point problem is being solved?
   enum Mode
   {
      GRAD_DIV, ///< Grad-div problem.
      DARCY     ///< Darcy/mixed Poisson problem.
   };
private:
   MINRESSolver minres;

   static constexpr int b1 = BasisType::GaussLobatto;
   static constexpr int b2 = BasisType::IntegratedGLL;
   static constexpr int mt = FiniteElement::INTEGRAL;

   const int order;

   // L2 and RT spaces, using the interpolation-histopolation bases
   L2_FECollection fec_l2;
   ParFiniteElementSpace fes_l2;

   RT_FECollection fec_rt;
   ParFiniteElementSpace fes_rt;

   const Array<int> &ess_rt_dofs; ///< Essential BCs (in the RT space only).

   // Change of basis operators
   ChangeOfBasis_L2 basis_l2;
   ChangeOfBasis_RT basis_rt;

   /// Whether conversion from map type VALUE to INTEGRAL is required.
   const bool convert_map_type;

   ParBilinearForm mass_l2, mass_rt;

   // Components needed for the block operator
   OperatorHandle L, R, R_e; ///< Mass matrices.
   std::unique_ptr<HypreParMatrix> D, Dt, D_e; ///< Divergence matrices.
   std::shared_ptr<DGMassInverse> L_inv; ///< Inverse of the DG mass matrix.
   std::shared_ptr<Operator> A_11; ///< (1,1)-block of the matrix

   /// Diagonals of the mass matrices
   Vector L_diag, R_diag, L_diag_unweighted;

   // Components needed for the preconditioner

   /// Jacobi preconditioner for the RT mass matrix.
   std::unique_ptr<OperatorJacobiSmoother> R_inv;
   std::unique_ptr<HypreParMatrix> S; ///< Approximate Schur complement.
   HypreBoomerAMG S_inv; ///< AMG preconditioner for #S.

   Array<int> offsets, empty;
   /// The 2x2 block operator.
   std::unique_ptr<BlockOperator> A_block;
   /// The block-diagonal preconditioner.
   std::unique_ptr<BlockDiagonalPreconditioner> D_prec;

   Coefficient &L_coeff, &R_coeff;

   const Mode mode;
   bool zero_l2_block = false;
   QuadratureSpace qs;
   QuadratureFunction W_coeff_qf, W_mix_coeff_qf;
   QuadratureFunctionCoefficient W_coeff, W_mix_coeff;

   ConstantCoefficient zero = ConstantCoefficient(0.0);

   // Work vectors
   mutable Vector b_prime, x_prime, x_bc, w, z;
public:
   /// @brief Creates a solver for the H(div) saddle-point system.
   ///
   /// The associated matrix is given by
   ///
   ///     [  L    B ]
   ///     [ B^T  -R ]
   ///
   /// where L is the L2 mass matrix, R is the RT mass matrix, and B is the
   /// divergence form (VectorFEDivergenceIntegrator).
   ///
   /// Essential boundary conditions in the RT space are given by @a
   /// ess_rt_dofs_. (Rows and columns are eliminated from R and columns are
   /// eliminated from B).
   ///
   /// The L block has coefficient @a L_coeff_ and the R block has coefficient
   /// @a R_coeff_.
   ///
   /// The parameter @a mode_ determines whether the block system corresponds to
   /// a grad-div problem or a Darcy problem. Specifically, if @a mode_ is
   /// Mode::GRAD_DIV, then the B and B^T blocks are also scaled by @a L_coeff_,
   /// and if @a mode_ is Mode::DARCY, then the B and B^T blocks are unweighted.
   ///
   /// Mode::GRAD_DIV corresponds to the grad-div problem
   ///
   ///     alpha u - grad ( beta div ( u )) = f,
   ///
   /// where alpha is @a R_coeff_ and beta is @a L_coeff_.
   ///
   /// Mode::DARCY corresponds to the Darcy-type problem
   ///
   ///     alpha p - div ( beta grad ( p )) = f,
   ///
   /// where alpha is @a L_coeff and beta is @a R_coeff_. In this case, the
   /// coefficient alpha is allowed to be zero.
   HdivSaddlePointSolver(ParMesh &mesh_,
                         ParFiniteElementSpace &fes_rt_,
                         ParFiniteElementSpace &fes_l2_,
                         Coefficient &L_coeff_,
                         Coefficient &R_coeff_,
                         const Array<int> &ess_rt_dofs_,
                         Mode mode_);

   /// @brief Creates a linear solver for the case when the L2 diagonal block is
   /// zero (for Darcy problems).
   ///
   /// Equivalent to passing ConstantCoefficient(0.0) as @a L_coeff_ and
   /// Mode::DARCY as @a mode_ to the primary constructor (see above).
   HdivSaddlePointSolver(ParMesh &mesh_,
                         ParFiniteElementSpace &fes_rt_,
                         ParFiniteElementSpace &fes_l2_,
                         Coefficient &R_coeff_,
                         const Array<int> &ess_rt_dofs_);

   /// @brief Build the linear operator and solver. Must be called when the
   /// coefficients change.
   void Setup();
   /// Sets the Dirichlet boundary conditions at the RT essential DOFs.
   void SetBC(const Vector &x_rt) { x_bc = x_rt; }
   /// @brief Solve the linear system for L2 (scalar) and RT (flux) unknowns.
   ///
   /// If the problem has essential boundary conditions (i.e. if @a ess_rt_dofs
   /// is not empty), then SetBC() must be called before Mult().
   void Mult(const Vector &b, Vector &x) const override;
   /// No-op.
   void SetOperator(const Operator &op) override { }
   /// Get the number of MINRES iterations.
   int GetNumIterations() const { return minres.GetNumIterations(); }
   /// Eliminates the BCs (called internally, not public interface).
   void EliminateBC(Vector &) const;
   /// Return the offsets of the block system.
   const Array<int> &GetOffsets() const { return offsets; }
   /// Returns the internal MINRES solver.
   MINRESSolver &GetMINRES() { return minres; }
};

} // namespace mfem

#endif
