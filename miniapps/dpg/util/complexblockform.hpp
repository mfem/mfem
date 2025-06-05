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

#ifndef MFEM_COMPLEX_BLOCKFORM
#define MFEM_COMPLEX_BLOCKFORM

#include "mfem.hpp"
#include "complexstaticcond.hpp"

namespace mfem
{

class ComplexBlockForm
{

protected:

   bool initialized = false;

   Mesh * mesh = nullptr;
   int height, width;
   int nblocks;
   Array<int> dof_offsets;
   Array<int> tdof_offsets;

   /// Block matrix $ M $ to be associated with the real/imag Block bilinear form. Owned.
   BlockMatrix *mat_r = nullptr;
   BlockMatrix *mat_i = nullptr;
   ComplexOperator * mat = nullptr;

   /** @brief Block Matrix $ M_e $ used to store the eliminations
        from the b.c.  Owned.
       $ M + M_e = M_{original} $ */
   BlockMatrix *mat_e_r = nullptr;
   BlockMatrix *mat_e_i = nullptr;

   /// FE spaces
   Array<FiniteElementSpace * > fes;

   /// Set of Trial Integrators to be applied for matrix A
   Array2D<Array<BilinearFormIntegrator * > * > integs_r;
   Array2D<Array<BilinearFormIntegrator * > * > integs_i;

   /// Block Prolongation
   BlockMatrix * P = nullptr;
   /// Block Restriction
   BlockMatrix * R = nullptr;

   mfem::Operator::DiagonalPolicy diag_policy;

   void Init();
   void ReleaseInitMemory();

   // Allocate appropriate SparseMatrix and assign it to mat
   void AllocMat();

   void ConformingAssemble();

   void ComputeOffsets();

   virtual void BuildProlongation();

private:

public:

   ComplexBlockForm()
   {
      height = 0;
      width = 0;
   }

   /// Creates bilinear form associated with FE spaces @a fes_.
   ComplexBlockForm(Array<FiniteElementSpace* > & fes_)
   {
      SetSpaces(fes_);
   }

   void SetSpaces(Array<FiniteElementSpace* > & fes_)
   {
      fes = fes_;
      nblocks = fes.Size();
      mesh = fes[0]->GetMesh();
      Init();
   }

   // Get the size of the bilinear form of the ComplexBlockForm
   int Size() const { return height; }

   // Pre-allocate the internal real and imag BlockMatrix before assembly.
   void AllocateMatrix() { if (mat_r == nullptr) { AllocMat(); } }

   ///  Finalizes the matrix initialization.
   void Finalize(int skip_zeros = 1);

   /// Returns a reference to the BlockMatrix:  $ M_r $
   BlockMatrix &BlockMat_r()
   {
      MFEM_VERIFY(mat_r, "mat_r is NULL and can't be dereferenced");
      return *mat_r;
   }
   /// Returns a reference to the BlockMatrix:  $ M_i $
   BlockMatrix &BlockMat_i()
   {
      MFEM_VERIFY(mat_i, "mat_i is NULL and can't be dereferenced");
      return *mat_i;
   }

   /// Returns a reference to the BlockMatrix of eliminated b.c.: $ M_e_r $
   BlockMatrix &BlockMatElim_r()
   {
      MFEM_VERIFY(mat_e_r, "mat_e is NULL and can't be dereferenced");
      return *mat_e_r;
   }

   /// Returns a reference to the BlockMatrix of eliminated b.c.: $ M_e_i $
   BlockMatrix &BlockMatElim_i()
   {
      MFEM_VERIFY(mat_e_i, "mat_e is NULL and can't be dereferenced");
      return *mat_e_i;
   }

   /** Adds new Trial Integrator. Assumes ownership of @a bfi_r and @a bfi_i.
       @a n and @a m correspond to the trial FESpace and test FEColl
       respectively */
   void AddDomainIntegrator(BilinearFormIntegrator *bfi_r,
                            BilinearFormIntegrator *bfi_i,
                            int n, int m);

   /// Assembles the form i.e. sums over all integrators.
   void Assemble(int skip_zeros = 1);

   virtual void FormLinearSystem(const Array<int> &ess_tdof_list,
                                 Vector &x, Vector &b, OperatorHandle & A,
                                 Vector &X, Vector &B, int copy_interior = 0);

   template <typename OpType>
   void FormLinearSystem(const Array<int> &ess_tdof_list,
                         Vector &x, Vector &b, OpType &A,
                         Vector &X, Vector &B, int copy_interior = 0)
   {
      OperatorHandle Ah;
      FormLinearSystem(ess_tdof_list, x, b, Ah, X, B, copy_interior);
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

   void EliminateVDofsInRHS(const Array<int> &vdofs,
                            const Vector &x_r, const Vector & x_i,
                            Vector &b_r, Vector & b_i);

   virtual void RecoverFEMSolution(const Vector &X, Vector &x);

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

   virtual void Update();

   /// Destroys bilinear form.
   virtual ~ComplexBlockForm();

};

} // namespace mfem

#endif
