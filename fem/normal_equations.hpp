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

#ifndef MFEM_NORMALEQUATIONS
#define MFEM_NORMALEQUATIONS

#include "../config/config.hpp"
#include "../linalg/linalg.hpp"

namespace mfem
{

/** @brief Class representing the whole weak formulation. (Convinient for DPG or
    Normal Equations) */
// First implementation for 1 Domain FE space and 1 trace FE space
// Example DPG primal Poisson
class NormalEquations
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

   // Domain FE spaces
   Array<FiniteElementSpace * > domain_fes;

   // Trace FE Spaces
   Array<FiniteElementSpace * > trace_fes;

   // All FE Spaces
   Array<FiniteElementSpace * > fespaces;

   // Test FE Collections (Broken)
   Array<FiniteElementCollection *> test_fecols;

   /// Set of Domain Integrators to be applied. Forming matrix B
   Array2D<Array<BilinearFormIntegrator * > * > domain_integs;

   /// Trace integrators. Forming Matrix Bhat
   Array2D<Array<BilinearFormIntegrator * > * > trace_integs;

   /// Set of Test Space (broken) Integrators to be applied. Forming matrix G
   Array2D<Array<BilinearFormIntegrator * > * > test_integs;

   DenseMatrix elemmat;
   Array<int>  vdofs;

   DenseTensor *element_matrices; ///< Owned.

   /// Set of Domain Integrators to be applied.
   Array<Array<LinearFormIntegrator * > * > domain_lf_integs;

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
   NormalEquations(Array<FiniteElementSpace* > & fes_,
                   Array<FiniteElementSpace* > & trace_fes_,
                   Array<FiniteElementCollection *> & fecol_)
      : domain_fes(fes_), trace_fes(trace_fes_), test_fecols(fecol_)
   {
      nblocks = domain_fes.Size() + trace_fes.Size();
      mesh = domain_fes[0]->GetMesh();
      fespaces.Append(domain_fes);
      fespaces.Append(trace_fes);
      Init();
   }

   // Get the size of the bilinear form of the NormalEquations
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
   // void SetDomainBFIntegrator(BilinearFormIntegrator *bfi);
   void AddDomainBFIntegrator(BilinearFormIntegrator *bfi, int trial_fes,
                              int test_fes);

   /// Adds new Domain Test BF Integrator. Assumes ownership of @a bfi.
   void AddTestIntegrator(BilinearFormIntegrator *bfi, int test_fes0,
                          int test_fes1);

   /// Adds new Trace Element Integrator. Assumes ownership of @a bfi.
   void AddTraceElementBFIntegrator(BilinearFormIntegrator *bfi, int trial_fes,
                                    int test_fes);

   /// Adds new Domain LF Integrator. Assumes ownership of @a bfi.
   void AddDomainLFIntegrator(LinearFormIntegrator *bfi, int test_fes);


   /// Assembles the form i.e. sums over all domain integrators.
   void Assemble(int skip_zeros = 1);


   virtual void FormLinearSystem(const Array<int> &ess_tdof_list, Vector &x,
                                 OperatorHandle &A, Vector &X,
                                 Vector &B, int copy_interior = 0);

   template <typename OpType>
   void FormLinearSystem(const Array<int> &ess_tdof_list, Vector &x,
                         OpType &A, Vector &X, Vector &B,
                         int copy_interior = 0)
   {
      OperatorHandle Ah;
      FormLinearSystem(ess_tdof_list, x, Ah, X, B, copy_interior);
      OpType *A_ptr = Ah.Is<OpType>();
      MFEM_VERIFY(A_ptr, "invalid OpType used");
      A.MakeRef(*A_ptr);
   }

   virtual void FormSystemMatrix(const Array<int> &ess_tdof_list,
                                 OperatorHandle &A);

   template <typename OpType>
   void FormSystemMatrix(const Array<int> &ess_tdof_list, OpType &A)
   {
      OperatorHandle Ah;
      FormSystemMatrix(ess_tdof_list, Ah);
      OpType *A_ptr = Ah.Is<OpType>();
      MFEM_VERIFY(A_ptr, "invalid OpType used");
      A.MakeRef(*A_ptr);
   }

   void EliminateVDofs(const Array<int> &vdofs,
                       Operator::DiagonalPolicy dpolicy = Operator::DIAG_ONE);

   void EliminateVDofsInRHS(const Array<int> &vdofs, const Vector &x, Vector &b);


   void RecoverFEMSolution(const Vector &X,Vector &x);

   /// Sets diagonal policy used upon construction of the linear system.
   /** Policies include:

       - DIAG_ZERO (Set the diagonal values to zero)
       - DIAG_ONE  (Set the diagonal values to one)
       - DIAG_KEEP (Keep the diagonal values)
   */
   void SetDiagonalPolicy(Operator::DiagonalPolicy policy)
   {
      diag_policy = policy;
   }

   /// Destroys bilinear form.
   ~NormalEquations();

};

} // namespace mfem

#endif
