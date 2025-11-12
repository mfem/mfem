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

#ifndef MFEM_HYBRIDIZATION
#define MFEM_HYBRIDIZATION

#include "../config/config.hpp"
#include "fespace.hpp"
#include "bilininteg.hpp"
#include <memory>

namespace mfem
{

/** @brief Auxiliary class Hybridization, used to implement BilinearForm
    hybridization.

    Hybridization can be viewed as a technique for solving linear systems
    obtained through finite element assembly. The assembled matrix $ A $ can
    be written as:
        $$ A = P^T \hat{A} P, $$
    where $ P $ is the matrix mapping the conforming finite element space to
    the purely local finite element space without any inter-element constraints
    imposed, and $ \hat{A} $ is the block-diagonal matrix of all element
    matrices.

    We assume that:
    - $ \hat{A} $ is invertible,
    - $ P $ has a left inverse $ R $, such that $ R P = I $,
    - a constraint matrix $ C $ can be constructed, such that
      $ \operatorname{Ker}(C) = \operatorname{Im}(P) $.

    Under these conditions, the linear system $ A x = b $ can be solved
    using the following procedure:
    - solve for $ \lambda $ in the linear system:
          $$ (C \hat{A}^{-1} C^T) \lambda = C \hat{A}^{-1} R^T b $$
    - compute $ x = R \hat{A}^{-1} (R^T b - C^T \lambda) $

    Hybridization is advantageous when the matrix
    $ H = (C \hat{A}^{-1} C^T) $ of the hybridized system is either smaller
    than the original system, or is simpler to invert with a known method.

    In some cases, e.g. high-order elements, the matrix $ C $ can be written
    as
        $$ C = \begin{pmatrix} 0 & C_b \end{pmatrix}, $$
    and then the hybridized matrix $ H $ can be assembled using the identity
        $$ H = C_b S_b^{-1} C_b^T, $$
    where $ S_b $ is the Schur complement of $ \hat{A} $ with respect to
    the same decomposition as the columns of $ C $:
        $$ S_b = \hat{A}_b - \hat{A}_{bf} \hat{A}_{f}^{-1} \hat{A}_{fb}. $$

    Hybridization can also be viewed as a discretization method for imposing
    (weak) continuity constraints between neighboring elements. */
class Hybridization
{
   friend class HybridizationExtension;
protected:
   FiniteElementSpace &fes; ///< The finite element space.
   FiniteElementSpace &c_fes; ///< The constraint finite element space.
   /// Extension for device execution.
   std::unique_ptr<class HybridizationExtension> ext;
   /// The constraint integrator.
   std::unique_ptr<BilinearFormIntegrator> c_bfi;
   /// The constraint boundary face integrators
   std::vector<BilinearFormIntegrator*> boundary_constraint_integs;
   /// Boundary markers for constraint face integrators
   std::vector<Array<int>*> boundary_constraint_integs_marker;
   /// Indicates if the boundary_constraint_integs integrators are owned externally
   bool extern_bdr_constr_integs{false};

   /// The constraint matrix.
   std::unique_ptr<SparseMatrix> Ct;
   /// The Schur complement system for the Lagrange multiplier.
   std::unique_ptr<SparseMatrix> H;

   Array<int> hat_offsets, hat_dofs_marker;
   Array<int> Af_offsets, Af_f_offsets;
   Array<real_t> Af_data;
   Array<int> Af_ipiv;

#ifdef MFEM_USE_MPI
   std::unique_ptr<HypreParMatrix> pC, P_pc; // for parallel non-conforming meshes
   OperatorHandle pH;
#endif

   /// Construct the constraint matrix.
   void ConstructC();

   /// Returns the local indices of the i-dofs and b-dofs of element @a el.
   void GetIBDofs(int el, Array<int> &i_dofs, Array<int> &b_dofs) const;

   /// Returns global indices of the b-dofs of element @a el.
   void GetBDofs(int el, int &num_idofs, Array<int> &b_dofs) const;

   /// Construct the Schur complement system.
   void ComputeH();

   // Compute depending on mode:
   // - mode 0: bf = Af^{-1} Rf^t b, where
   //           the non-"boundary" part of bf is set to 0;
   // - mode 1: bf = Af^{-1} ( Rf^t b - Cf^t lambda ), where
   //           the "essential" part of bf is set to 0.
   // Input: size(b)      =   fes->GetConformingVSize()
   //        size(lambda) = c_fes->GetConformingVSize()
   void MultAfInv(const Vector &b, const Vector &lambda, Vector &bf,
                  int mode) const;

public:
   /// Constructor.
   Hybridization(FiniteElementSpace *fespace, FiniteElementSpace *c_fespace);

   /// Destructor.
   ~Hybridization();

   /// Turns on device execution.
   void EnableDeviceExecution();

   /// @brief Set the integrator that will be used to construct the constraint
   /// matrix C.
   ///
   /// The Hybridization object assumes ownership of the integrator, i.e. it
   /// will delete the integrator when destroyed.
   void SetConstraintIntegrator(BilinearFormIntegrator *c_integ)
   { c_bfi.reset(c_integ); }

   /** Add the boundary face integrator that will be used to construct the
       constraint matrix C. The Hybridization object assumes ownership of the
       integrator, i.e. it will delete the integrator when destroyed. */
   void AddBdrConstraintIntegrator(BilinearFormIntegrator *c_integ)
   {
      boundary_constraint_integs.push_back(c_integ);
      boundary_constraint_integs_marker.push_back(nullptr);
   }
   void AddBdrConstraintIntegrator(BilinearFormIntegrator *c_integ,
                                   Array<int> &bdr_marker)
   {
      boundary_constraint_integs.push_back(c_integ);
      boundary_constraint_integs_marker.push_back(&bdr_marker);
   }

   /// Access all integrators added with AddBdrConstraintIntegrator().
   BilinearFormIntegrator& GetBdrConstraintIntegrator(int i)
   { return *boundary_constraint_integs[i]; }

   /// Access all boundary markers added with AddBdrConstraintIntegrator().
   /** If no marker was specified when the integrator was added, the
       corresponding pointer (to Array<int>) will be NULL. */
   Array<int>* GetBdrConstraintIntegratorMarker(int i)
   { return boundary_constraint_integs_marker[i]; }

   /// Indicate that boundary constraint integrators are not owned
   void UseExternalBdrConstraintIntegrators() { extern_bdr_constr_integs = true; }

   /// Prepare the Hybridization object for assembly.
   void Init(const Array<int> &ess_tdof_list);

   /// Assemble the element matrix A into the hybridized system matrix.
   void AssembleMatrix(int el, const DenseMatrix &A);

   /// Assemble all of the element matrices given in the form of a DenseTensor.
   void AssembleElementMatrices(const class DenseTensor &el_mats);

   /// Assemble the boundary element matrix A into the hybridized system matrix.
   void AssembleBdrMatrix(int bdr_el, const DenseMatrix &A);

   /// Finalize the construction of the hybridized matrix.
   void Finalize();

   /// Return the serial hybridized matrix.
   SparseMatrix &GetMatrix() { return *H; }

#ifdef MFEM_USE_MPI
   /// Return the parallel hybridized matrix.
   HypreParMatrix &GetParallelMatrix() { return *pH.Is<HypreParMatrix>(); }

   /// @brief Return the parallel hybridized matrix in the format specified by
   /// SetOperatorType().
   void GetParallelMatrix(OperatorHandle &H_h) const { H_h = pH; }

   /// Set the operator type id for the parallel hybridized matrix/operator.
   void SetOperatorType(Operator::Type tid) { pH.SetType(tid); }
#endif

   /// @brief Perform the reduction of the given right-hand side @a b to a
   /// right-hand side vector @a b_r for the hybridized system.
   void ReduceRHS(const Vector &b, Vector &b_r) const;

   /// @brief Reconstruct the solution of the original system @a sol from
   /// solution of the hybridized system @a sol_r and the original right-hand
   /// side @a b.
   ///
   /// It is assumed that the vector sol has the correct essential boundary
   /// conditions.
   void ComputeSolution(const Vector &b, const Vector &sol_r,
                        Vector &sol) const;

   /// @brief Destroy the current hybridization matrix while preserving the
   /// computed constraint matrix and the set of essential true dofs.
   ///
   /// After Reset(), a new hybridized matrix can be assembled via
   /// AssembleMatrix() and Finalize(). The Mesh and FiniteElementSpace objects
   /// are assumed to be unmodified. If that is not the case, a new
   /// Hybridization object must be created.
   void Reset();
};

}

#endif
