// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
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

#ifdef MFEM_USE_MPI

#include "solvers.hpp"

namespace mfem
{

// forward declarations
class Coefficient;
class MatrixCoefficient;
class ParMesh;
class ParBilinearForm;
class ParDiscreteLinearOperator;

/** @brief Auxiliary space solvers for MatrixFreeAMS preconditioner

    Given an operator A and a transfer G, create a solver that approximates
    (G^T A G)^{-1}. Used for two different auxiliary spaces in the AMS cycle.

    The produced solver is based on a low-order refined discretization for the
    high-order H1 problem. */
class MatrixFreeAuxiliarySpace : public Solver
{
public:
   /** @brief Pi space constructor

       In the AMS framework this auxiliary space has two coefficients.

       @param mesh_lor       Low-order refined auxiliary mesh
       @param alpha_coeff    Coefficient on curl-curl term (1 if null)
       @param beta_coeff     Coefficient on mass term (1 if null)
       @param beta_mcoeff    Matrix coefficient on mass term
       @param ess_bdr        Attributes for essential boundaries
       @param curlcurl_oper  High-order operator for the system
       @param pi             Identity interpolation operator
       @param useAmgX_       Use AmgX instead of hypre for auxiliary solves
       @param cg_iterations  Number of CG iterations used to invert auxiliary
                             system, choosing 0 means to use a single V-cycle
   */
   MatrixFreeAuxiliarySpace(
      ParMesh& mesh_lor, Coefficient* alpha_coeff,
      Coefficient* beta_coeff, MatrixCoefficient* beta_mcoeff,
      Array<int>& ess_bdr, Operator& curlcurl_oper, Operator& pi,
#ifdef MFEM_USE_AMGX
      bool useAmgX_,
#endif
      int cg_iterations = 0);

   /** @brief G space constructor

       This has one coefficient in the AMS framework.

       @param mesh_lor       Low-order refined auxiliary mesh
       @param beta_coeff     Coefficient on mass term (1 if null)
       @param beta_mcoeff    Matrix coefficient on mass term
       @param ess_bdr        Attributes for essential boundaries
       @param curlcurl_oper  High-order operator for the system
       @param g              Gradient interpolation operator
       @param useAmgX_       Use AmgX instead of hypre for auxiliary solves
       @param cg_iterations  Number of CG iterations used to invert auxiliary
                             system, choosing 0 means to use a single V-cycle
   */
   MatrixFreeAuxiliarySpace(
      ParMesh& mesh_lor, Coefficient* beta_coeff,
      MatrixCoefficient* beta_mcoeff, Array<int>& ess_bdr,
      Operator& curlcurl_oper, Operator& g,
#ifdef MFEM_USE_AMGX
      bool useAmgX_,
#endif
      int cg_iterations = 1);

   ~MatrixFreeAuxiliarySpace();

   void Mult(const Vector& x, Vector& y) const;

   void SetOperator(const Operator& op) {}

private:
   /** @brief Helper routine for constructors.

       @param system_dimension is passed to HypreBoomerAMG::SetSystemsOptions
   */
   void SetupAMG(int system_dimension);
   void SetupVCycle();

   /// inner_cg_iterations > 99 applies an exact solve here
   void SetupCG(Operator& curlcurl_oper, Operator& conn,
                int inner_cg_iterations);

   MPI_Comm comm;
   Array<int> ess_tdof_list;
   HypreParMatrix * lor_matrix;
   Solver * lor_pc;
   Operator* matfree;
   CGSolver* cg;
   Operator* aspacewrapper;
#ifdef MFEM_USE_AMGX
   const bool useAmgX;
#endif
   mutable int inner_aux_iterations;
};


/** @brief Perform AMS cycle with generic Operator objects.

    Most users should use MatrixFreeAMS, which wraps this. */
class GeneralAMS : public Solver
{
public:
   /** @brief Constructor.

       Most of these arguments just need a Mult() operation, but pi and g also
       require MultTranspose() */
   GeneralAMS(const Operator& curlcurl_op_,
              const Operator& pi_,
              const Operator& gradient_,
              const Operator& pispacesolver_,
              const Operator& gspacesolver_,
              const Operator& smoother_,
              const Array<int>& ess_tdof_list_);
   virtual ~GeneralAMS();

   /// in principle this should set A_ = op;
   void SetOperator(const Operator &op) override {}

   void Mult(const Vector& x, Vector& y) const override;

private:
   const Operator& curlcurl_op;
   const Operator& pi;
   const Operator& gradient;
   const Operator& pispacesolver;
   const Operator& gspacesolver;
   const Operator& smoother;
   const Array<int> ess_tdof_list;

   void FormResidual(const Vector& rhs, const Vector& x,
                     Vector& residual) const;
};


/** @brief An auxiliary Maxwell solver for a high-order curl-curl system without
    high-order assembly.

    The auxiliary space solves are done using a low-order refined approach, but
    all the interpolation operators, residuals, etc. are done in a matrix-free
    manner.

    See Barker and Kolev, Matrix-free preconditioning for high-order H(curl)
    discretizations (https://doi.org/10.1002/nla.2348) */
class MatrixFreeAMS : public Solver
{
public:
   /** @brief Construct matrix-free AMS preconditioner

       @param aform        BilinearForm for curl-curl problem, generally will
                           have a CurlCurlIntegrator and possibly a
                           VectorFEMassIntegrator.
       @param oper         Operator to precondition.
       @param nd_fespace   Underlying Nedelec finite element space.
       @param alpha_coeff  Coefficient on curl-curl term in Maxwell problem
                           (can be null, in which case constant 1 is assumed)
       @param beta_coeff   (scalar) coefficient on mass term in Maxwell problem
       @param beta_mcoeff  (matrix) coefficient on mass term
       @param ess_bdr      Boundary *attributes* that are marked essential. In
                           contrast to other MFEM cases, these are *attributes*
                           not dofs, because we need to apply these boundary
                           conditions to different bilinear forms.
       @param useAmgX      Use AmgX (instead of hypre) for LOR problems
       @param inner_pi_its Number of CG iterations on auxiliary pi space,
                           may need more for difficult coefficients
       @param inner_g_its  Number of CG iterations on auxiliary g space,
                           may need more for difficult coefficients
       @param nd_smoother  Optional user-provided smoother for Nedelec space,
                           this object takes ownership and will delete.
   */
   MatrixFreeAMS(ParBilinearForm& aform, Operator& oper,
                 ParFiniteElementSpace& nd_fespace, Coefficient* alpha_coeff,
                 Coefficient* beta_coeff, MatrixCoefficient* beta_mcoeff,
                 Array<int>& ess_bdr,
#ifdef MFEM_USE_AMGX
                 bool useAmgX = false,
#endif
                 int inner_pi_its = 0, int inner_g_its = 1,
                 Solver* nd_smoother = NULL);

   ~MatrixFreeAMS();

   void SetOperator(const Operator &op) {}

   void Mult(const Vector& x, Vector& y) const { general_ams->Mult(x, y); }

private:
   GeneralAMS * general_ams;

   Solver * smoother;
   ParDiscreteLinearOperator * pa_grad;
   OperatorPtr Gradient;
   ParDiscreteLinearOperator * pa_interp;
   OperatorPtr Pi;

   Solver * Gspacesolver;
   Solver * Pispacesolver;

   ParFiniteElementSpace * h1_fespace;
   ParFiniteElementSpace * h1_fespace_d;
};

} // namespace mfem

#endif // MFEM_USE_MPI

#endif
