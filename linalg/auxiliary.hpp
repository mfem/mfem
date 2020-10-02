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

// forward declarations
class Coefficient;
class ParMesh;
class ParBilinearForm;
class ParDiscreteLinearOperator;

/** @brief Auxiliary space solvers for MatrixFreeAMS preconditioner

    Given an operator A and a transfer G, this will create a solver
    that approximates (G^T A G)^{-1}. Used for two different
    auxiliary spaces in the AMS cycle. */
class MatrixFreeAuxiliarySpace : public Solver
{
public:
   /** @brief Pi space constructor

       In the AMS framework this auxiliary space has two coefficients.

       @param alpha_coeff    coefficient on curl-curl term (1 if null)
       @param beta_coeff     coefficient on mass term (1 if null)
       @param cg_iterations  number of CG iterations used to invert
                             auxiliary system, choosing 0 means to
                             use a single V-cycle
   */
   MatrixFreeAuxiliarySpace(
      ParMesh& mesh_lor, Coefficient* alpha_coeff,
      Coefficient* beta_coeff, Array<int>& ess_bdr,
      Operator& curlcurl_oper, Operator& pi,
      int cg_iterations = 0);

   /** @brief G space constructor

       This has one coefficient in the AMS framework.

       @param beta_coeff     coefficient on mass term (1 if null)
       @param cg_iterations  number of CG iterations used to invert
                             auxiliary system, choosing 0 means to
                             use a single V-cycle
   */
   MatrixFreeAuxiliarySpace(
      ParMesh& mesh_lor,
      Coefficient* beta_coeff, Array<int>& ess_bdr,
      Operator& curlcurl_oper, Operator& g,
      int cg_iterations = 1);

   ~MatrixFreeAuxiliarySpace();

   void Mult(const Vector& x, Vector& y) const;

   void SetOperator(const Operator& op) {}

private:
   void SetupBoomerAMG(int system_dimension);
   void SetupVCycle();

   /// inner_cg_iterations > 99 applies an exact solve here
   void SetupCG(Operator& curlcurl_oper, Operator& conn,
                int inner_cg_iterations, bool very_verbose=false);

   Array<int> ess_tdof_list_;
   HypreParMatrix * aspacematrix_;
   Solver * aspacepc_;
   Operator* matfree_;
   CGSolver* cg_;
   Operator* aspacewrapper_;

   mutable int inner_aux_iterations_;
};

/** @brief Perform AMS cycle with generic Operator objects.

    Most users should use MatrixFreeAMS, which wraps this. */
class GeneralAMS : public Solver
{
public:
   /** @brief Constructor.

       Most of these arguments just need a Mult() operation,
       but pi and g also require MultTranspose() */
   GeneralAMS(const Operator& A,
              const Operator& pi,
              const Operator& g,
              const Operator& pispacesolver,
              const Operator& gspacesolver,
              const Operator& smoother,
              const Array<int>& ess_tdof_list);
   virtual ~GeneralAMS();

   /// in principle this should set A_ = op;
   void SetOperator(const Operator &op) {}

   virtual void Mult(const Vector& x, Vector& y) const;

private:
   const Operator& A_;
   const Operator& pi_;
   const Operator& g_;
   const Operator& pispacesolver_;
   const Operator& gspacesolver_;
   const Operator& smoother_;
   const Array<int> ess_tdof_list_;

   mutable StopWatch chrono_;

   mutable double residual_time_;
   mutable double smooth_time_;
   mutable double gspacesolver_time_;
   mutable double pispacesolver_time_;

   void FormResidual(const Vector& rhs, const Vector& x,
                     Vector& residual) const;
};

/** @brief An auxiliary Maxwell solver for a high-order curl-curl
    system without high-order assembly.

    The auxiliary space solves are done using a low-order refined approach,
    but all the interpolation operators, residuals, etc. are done in a
    matrix-free manner. */
class MatrixFreeAMS : public Solver
{
public:
   /// ess_bdr is the boundary attributes that are essential (not the dofs, the attributes)
   MatrixFreeAMS(ParBilinearForm& aform, Operator& oper,
                 ParFiniteElementSpace& nd_fespace, Coefficient* alpha_coeff,
                 Coefficient* beta_coeff, Array<int>& ess_bdr,
                 int inner_pi_iterations = 0, int inner_g_iterations = 1);
   ~MatrixFreeAMS();

   void SetOperator(const Operator &op) {}

   void Mult(const Vector& x, Vector& y) const { general_ams_->Mult(x, y); }

private:
   GeneralAMS * general_ams_;

   Solver * smoother_;
   ParDiscreteLinearOperator * pa_grad_;
   OperatorPtr G_;
   ParDiscreteLinearOperator * pa_interp_;
   OperatorPtr Pi_;

   Solver * Gspacesolver_;
   Solver * Pispacesolver_;

   ParFiniteElementSpace * h1_fespace_;
   ParFiniteElementSpace * h1_fespace_d_;
};

} // namespace mfem

#endif
