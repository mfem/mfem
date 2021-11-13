// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_WEAKFORMULATION
#define MFEM_WEAKFORMULATION

#include "../config/config.hpp"
#include "../linalg/linalg.hpp"

namespace mfem
{

/** @brief Class representing the whole weak formulation. (Convinient for DPG or
    Normal Equations) */
// First implementation for 1 Domain FE space and 1 trace FE space
// Example DPG primal Poisson
class NormalEquationsWeakFormulation
{

protected:
   Mesh * mesh = nullptr;
   int height, width;
   int nblocks;
   Array<int> dof_offsets;
   Array<int> tdof_offsets;

   /// Sparse matrix \f$ M \f$ to be associated with the bilinear form. Owned.
   SparseMatrix *mat = nullptr;

   /// Vector to be associated with the linear form
   Vector * y = nullptr;

   /** @brief Sparse Matrix \f$ M_e \f$ used to store the eliminations
        from the b.c.  Owned.
       \f$ M + M_e = M_{original} \f$ */
   SparseMatrix *mat_e = nullptr;

   // Domain FE space
   FiniteElementSpace * fes = nullptr;

   // Trace FE Space
   FiniteElementSpace * trace_fes = nullptr;

   Array<FiniteElementSpace * > fespaces;

   FiniteElementCollection * test_fecol; // FE Collection for broken test spaces

   /// Set of Domain Integrators to be applied. Forming matrix B
   BilinearFormIntegrator * domain_bf_integ = nullptr;

   /// Trace integrators. Forming Matrix Bhat
   BilinearFormIntegrator * trace_integ = nullptr;

   /// Set of Test Space (broken) Integrators to be applied. Forming matrix G
   BilinearFormIntegrator * test_integ = nullptr;

   DenseMatrix elemmat;
   Array<int>  vdofs;

   DenseTensor *element_matrices; ///< Owned.

   /// Set of Domain Integrators to be applied.
   LinearFormIntegrator * domain_lf_integs = nullptr;

   BlockMatrix * P = nullptr; // Block Prolongation
   BlockMatrix * R = nullptr; // Block Restriction

   mfem::Operator::DiagonalPolicy diag_policy;

   void Init();

   // Allocate appropriate SparseMatrix and assign it to mat
   void AllocMat();

   void ConformingAssemble();

   void BuildProlongation();

private:

public:

   /// Creates bilinear form associated with FE spaces @a *fespaces.
   NormalEquationsWeakFormulation(FiniteElementSpace * fes_,
                                  FiniteElementSpace * trace_fes_,
                                  FiniteElementCollection * fecol_)
      : fes(fes_), trace_fes(trace_fes_), test_fecol(fecol_)
   {
      fespaces.SetSize(2);
      fespaces[0] = fes;
      fespaces[1] = trace_fes;

      Init();
   }

   // Get the size of the bilinear form of the NormalEquationsWeakFormulation
   int Size() const { return height; }

   // Pre-allocate the internal SparseMatrix before assembly.
   void AllocateMatrix() { if (mat == NULL) { AllocMat(); } }

   ///  Finalizes the matrix initialization.
   void Finalize(int skip_zeros = 1);

   /// Returns a reference to the sparse matrix:  \f$ M \f$
   SparseMatrix &SpMat()
   {
      MFEM_VERIFY(mat, "mat is NULL and can't be dereferenced");
      return *mat;
   }

   /// Returns a reference to the sparse matrix of eliminated b.c.: \f$ M_e \f$
   SparseMatrix &SpMatElim()
   {
      MFEM_VERIFY(mat_e, "mat_e is NULL and can't be dereferenced");
      return *mat_e;
   }

   /// Adds new Domain BF Integrator. Assumes ownership of @a bfi.
   void SetDomainBFIntegrator(BilinearFormIntegrator *bfi);

   /// Adds new Domain Test BF Integrator. Assumes ownership of @a bfi.
   void SetTestIntegrator(BilinearFormIntegrator *bfi);

   /// Adds new Trace Element Integrator. Assumes ownership of @a bfi.
   void SetTraceElementBFIntegrator(BilinearFormIntegrator *bfi);

   /// Adds new Domain LF Integrator. Assumes ownership of @a bfi.
   void SetDomainLFIntegrator(LinearFormIntegrator *bfi);


   /// Assembles the form i.e. sums over all domain integrators.
   void Assemble(int skip_zeros = 1);


   // virtual void FormLinearSystem(const Array<int> &ess_tdof_list, Vector &x,
   //                               Vector &b, OperatorHandle &A, Vector &X,
   //                               Vector &B, int copy_interior = 0);

   // template <typename OpType>
   // void FormLinearSystem(const Array<int> &ess_tdof_list, Vector &x, Vector &b,
   //                       OpType &A, Vector &X, Vector &B,
   //                       int copy_interior = 0)
   // {
   //    OperatorHandle Ah;
   //    FormLinearSystem(ess_tdof_list, x, b, Ah, X, B, copy_interior);
   //    OpType *A_ptr = Ah.Is<OpType>();
   //    MFEM_VERIFY(A_ptr, "invalid OpType used");
   //    A.MakeRef(*A_ptr);
   // }

   // virtual void FormSystemMatrix(const Array<int> &ess_tdof_list,
   //                               OperatorHandle &A);

   /// Form the linear system matrix A, see FormLinearSystem() for details.
   /** Version of the method FormSystemMatrix() where the system matrix is
       returned in the variable @a A, of type OpType, holding a *reference* to
       the system matrix (created with the method OpType::MakeRef()). The
       reference will be invalidated when SetOperatorType(), Update(), or the
       destructor is called. */
   // template <typename OpType>
   // void FormSystemMatrix(const Array<int> &ess_tdof_list, OpType &A)
   // {
   //    OperatorHandle Ah;
   //    FormSystemMatrix(ess_tdof_list, Ah);
   //    OpType *A_ptr = Ah.Is<OpType>();
   //    MFEM_VERIFY(A_ptr, "invalid OpType used");
   //    A.MakeRef(*A_ptr);
   // }

   // virtual void RecoverFEMSolution(const Vector &X, const Vector &b, Vector &x);


   // void ComputeElementMatrices();
   // void ComputeElementVectors();

   /// Free the memory used by the element matrices.
   // void FreeElementMatrices()
   // { delete element_matrices; element_matrices = NULL; }

   /// Compute the element matrix of the given element
   /** The element matrix is computed by calling the domain integrators
       or the one stored internally by a prior call of ComputeElementMatrices()
       is returned when available.
   */
   // void ComputeElementMatrix(int i, DenseMatrix &elmat);

   // void ComputeElementVector(int i, Vector &elvector);

   /// Eliminate essential boundary DOFs from the system.
   /** The array @a bdr_attr_is_ess marks boundary attributes that constitute
       the essential part of the boundary. By default, the diagonal at the
       essential DOFs is set to 1.0. This behavior is controlled by the argument
       @a dpolicy. */
   // void EliminateEssentialBC(const Array<int> &bdr_attr_is_ess,
   //                           const Vector &sol, Vector &rhs,
   //                           Operator::DiagonalPolicy dpolicy = Operator::DIAG_ONE);

   /// Eliminate essential boundary DOFs from the system matrix.
   // void EliminateEssentialBC(const Array<int> &bdr_attr_is_ess,
   //   Operator::DiagonalPolicy dpolicy = Operator::DIAG_ONE);
   /// Perform elimination and set the diagonal entry to the given value
   // void EliminateEssentialBCDiag(const Array<int> &bdr_attr_is_ess,
   // double value);

   /// Eliminate the given @a vdofs.
   /** NOTE: here, @a vdofs is a list of DOFs from all the fespaces
       In this case the eliminations are applied to the internal \f$ M \f$
       and @a rhs without storing the elimination matrix \f$ M_e \f$. */
   // void EliminateVDofs(const Array<int> &vdofs, const Vector &sol, Vector &rhs,
   //   Operator::DiagonalPolicy dpolicy = Operator::DIAG_ONE);

   /// Eliminate the given @a vdofs (all the fespaces), storing the eliminated part internally in \f$ M_e \f$.
   /** This method works in conjunction with EliminateVDofsInRHS() and allows
       elimination of boundary conditions in multiple right-hand sides. In this
       method, @a vdofs is a list of DOFs. */
   // void EliminateVDofs(const Array<int> &vdofs,
   //   Operator::DiagonalPolicy dpolicy = Operator::DIAG_ONE);

   /** @brief Similar to
       EliminateVDofs(const Array<int> &, const Vector &, Vector &, DiagonalPolicy)
       but here @a ess_dofs is a marker (boolean) array on all vector-dofs
       (@a ess_dofs[i] < 0 is true). */
   // void EliminateEssentialBCFromDofs(const Array<int> &ess_dofs, const Vector &sol,
   //  Vector &rhs,
   //  Operator::DiagonalPolicy dpolicy = Operator::DIAG_ONE);

   /** @brief Similar to EliminateVDofs(const Array<int> &, DiagonalPolicy) but
       here @a ess_dofs is a marker (boolean) array on all vector-dofs
       (@a ess_dofs[i] < 0 is true). */
   // void EliminateEssentialBCFromDofs(const Array<int> &ess_dofs,
   //  Operator::DiagonalPolicy dpolicy = Operator::DIAG_ONE);
   /// Perform elimination and set the diagonal entry to the given value
   // void EliminateEssentialBCFromDofsDiag(const Array<int> &ess_dofs,
   //   double value);

   /** @brief Use the stored eliminated part of the matrix (see
       EliminateVDofs(const Array<int> &, DiagonalPolicy)) to modify the r.h.s.
       @a b; @a vdofs is a list of DOFs (non-directional, i.e. >= 0). */
   // void EliminateVDofsInRHS(const Array<int> &vdofs, const Vector &x,
   //  Vector &b);


   /// Sets diagonal policy used upon construction of the linear system.
   /** Policies include:

       - DIAG_ZERO (Set the diagonal values to zero)
       - DIAG_ONE  (Set the diagonal values to one)
       - DIAG_KEEP (Keep the diagonal values)
   */
   // void SetDiagonalPolicy(Operator::DiagonalPolicy policy)
   // {
   // diag_policy = policy;
   // }

   /// Destroys bilinear form.
   ~NormalEquationsWeakFormulation();

};

} // namespace mfem

#endif
