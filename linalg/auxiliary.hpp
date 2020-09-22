// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_AUXILIARY
#define MFEM_AUXILIARY

#include "../config/config.hpp"
#include "../general/tic_toc.hpp"

#include "solvers.hpp"

namespace mfem
{

// forward declarations (can probably be reduced/simplified
class Coefficient;
class ParMesh;
class ParBilinearForm;
class ParDiscreteLinearOperator;

/**
   The basic idea is that given an operator A and a transfer
   G, this will create a solver that approximates (G^T A G)^{-1}

   In practice we only use this for an AMS cycle, so some of
   the notation and algorithmic choices are specific to that.
*/
class MatrixFreeAuxiliarySpace : public mfem::Solver
{
public:
   /**
      Pi space constructor (two coefficients)

      cg_iterations = 0 means a single V-cycle
      otherwise we wrap BoomerAMG in CG

      rap_in_lor does a RAP product in the LOR space
      for building the matrix
   */
   MatrixFreeAuxiliarySpace(
      mfem::ParMesh& mesh_lor,
      mfem::Coefficient* alpha_coeff,
      mfem::Coefficient* beta_coeff, Array<int>& ess_bdr,
      mfem::Operator& curlcurl_oper, mfem::Operator& pi,
      int cg_iterations = 0);

   /**
      G space constructor (one coefficient)

      cg_iterations = 0 means a single V-cycle
      otherwise we wrap BoomerAMG in CG

      rap_in_lor does a RAP product in the LOR space
      for building the matrix
   */
   MatrixFreeAuxiliarySpace(
      mfem::ParMesh& mesh_lor,
      mfem::Coefficient* beta_coeff, Array<int>& ess_bdr,
      mfem::Operator& curlcurl_oper, mfem::Operator& g,
      int cg_iterations = 1);

   ~MatrixFreeAuxiliarySpace();

   void Mult(const mfem::Vector& x, mfem::Vector& y) const;

   void SetOperator(const mfem::Operator& op) {}

private:
   void SetupBoomerAMG(int system_dimension);
   void SetupVCycle();

   /// inner_cg_iterations > 99 applies an exact solve here
   void SetupCG(
      mfem::Operator& curlcurl_oper, mfem::Operator& conn,
      int inner_cg_iterations, bool very_verbose=false);

   mfem::Array<int> ess_tdof_list_;
   mfem::HypreParMatrix * aspacematrix_;
   // mfem::HypreBoomerAMG * aspacepc_;
   Solver * aspacepc_;
   mfem::Operator* matfree_;
   mfem::CGSolver* cg_;
   mfem::Operator* aspacewrapper_;

   mutable int inner_aux_iterations_;
};

/**
   Perform AMS cycle with generic Operator objects.

   Most users should use MatrixFreeAMS, which wraps this.
*/
class GeneralAMS : public mfem::Solver
{
public:
   /**
      pi and g should have Mult() and MultTranspose()

      the rest just nead Mult()
   */
   GeneralAMS(const mfem::Operator& A,
              const mfem::Operator& pi,
              const mfem::Operator& g,
              const mfem::Operator& pispacesolver,
              const mfem::Operator& gspacesolver,
              const mfem::Operator& smoother,
              const mfem::Array<int>& ess_tdof_list);
   virtual ~GeneralAMS();

   /// in principle this should set A_ = op;
   void SetOperator(const mfem::Operator &op) {}

   virtual void Mult(const mfem::Vector& x, mfem::Vector& y) const;

private:
   const mfem::Operator& A_;
   const mfem::Operator& pi_;
   const mfem::Operator& g_;
   const mfem::Operator& pispacesolver_;
   const mfem::Operator& gspacesolver_;
   const mfem::Operator& smoother_;
   const mfem::Array<int> ess_tdof_list_;

   mutable mfem::StopWatch chrono_;

   mutable double residual_time_;
   mutable double smooth_time_;
   mutable double gspacesolver_time_;
   mutable double pispacesolver_time_;

   void DebugVector(const mfem::Vector& vec, const std::string& tag) const;

   void FormResidual(const mfem::Vector& rhs, const mfem::Vector& x,
                     mfem::Vector& residual) const;
};

/**
   An auxiliary Maxwell solver for high-order finite element operators without
   high-order assembly.

   The auxiliary space solves are done using a low-order refined approach, but
   all the interpolation operators, residuals, etc. are done in a matrix-free
   manner.
*/
class MatrixFreeAMS : public mfem::Solver
{
public:
   /// ess_bdr is the boundary attributes that are essential (not the dofs, the attributes)
   MatrixFreeAMS(ParBilinearForm& aform, mfem::Operator& oper,
                 mfem::ParFiniteElementSpace& nd_fespace, mfem::Coefficient* alpha_coeff,
                 mfem::Coefficient* beta_coeff, mfem::Array<int>& ess_bdr,
                 int inner_pi_iterations = 0, int inner_g_iterations = 1);
   ~MatrixFreeAMS();

   void SetOperator(const mfem::Operator &op) {}

   void Mult(const mfem::Vector& x, mfem::Vector& y) const { general_ams_->Mult(x, y); }

private:
   GeneralAMS * general_ams_;

   Solver * smoother_;
   ParDiscreteLinearOperator * pa_grad_;
   OperatorPtr G_;
   ParDiscreteLinearOperator * pa_interp_;
   OperatorPtr Pi_;

   mfem::Solver * Gspacesolver_;
   mfem::Solver * Pispacesolver_;

   mfem::ParFiniteElementSpace * h1_fespace_;
   mfem::ParFiniteElementSpace * h1_fespace_d_;
};

} // namespace mfem

#endif
