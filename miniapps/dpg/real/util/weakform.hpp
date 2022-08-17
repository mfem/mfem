// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_DPGWEAKFORM
#define MFEM_DPGWEAKFORM

#include "../../../../config/config.hpp"
#include "../../../../linalg/linalg.hpp"
#include "../../../../fem/fem.hpp"
#include "blockstaticcond.hpp"

namespace mfem
{

/// @brief Class representing the DPG weak formulation. 
class DPGWeakForm
{

protected:

   BlockStaticCondensation *static_cond; ///< Owned.

   bool initialized = false;

   Mesh * mesh = nullptr;
   int height, width;
   int nblocks;
   Array<int> dof_offsets;
   Array<int> tdof_offsets;

   /// Block matrix \f$ M \f$ to be associated with the Block bilinear form. Owned.
   BlockMatrix *mat = nullptr;

   /// BlockVector to be associated with the Block linear form
   BlockVector * y = nullptr;

   /** @brief Block Matrix \f$ M_e \f$ used to store the eliminations
        from the b.c.  Owned.
       \f$ M + M_e = M_{original} \f$ */
   BlockMatrix *mat_e = nullptr;

   // Trial FE spaces
   Array<FiniteElementSpace * > trial_fes;

   // Flags to determine if a FiniteElementSpace is Trace
   Array<int> IsTraceFes;

   // Test FE Collections (Broken)
   Array<FiniteElementCollection *> test_fecols;
   Array<int> test_fecols_vdims;

   /// Set of Trial Integrators to be applied for matrix B
   Array2D<Array<BilinearFormIntegrator * > * > trial_integs;

   /// Set of Test Space (broken) Integrators to be applied for matrix G
   Array2D<Array<BilinearFormIntegrator * > * > test_integs;

   /// Set of Liniear Froem Integrators to be applied.
   Array<Array<LinearFormIntegrator * > * > lfis;

   BlockMatrix * P = nullptr; // Block Prolongation
   BlockMatrix * R = nullptr; // Block Restriction

   mfem::Operator::DiagonalPolicy diag_policy;

   void Init();
   void ReleaseInitMemory();

   // Allocate appropriate SparseMatrix and assign it to mat
   void AllocMat();

   void ConformingAssemble();

   void ComputeOffsets();

   virtual void BuildProlongation();

   bool store_matrices = false;

   // Store the matrix L^-1 B  and Vector L^-1 l
   // where G = L L^t
   Array<DenseMatrix * > Bmat;
   Array<Vector * > fvec;
   Vector residuals;


private:

public:

   /// Creates bilinear form associated with FE spaces @a *fespaces.
   DPGWeakForm()
   {
      height = 0.;
      width = 0;
   }

   DPGWeakForm(Array<FiniteElementSpace* > & fes_,
                   Array<FiniteElementCollection *> & fecol_)
   {
      SetSpaces(fes_,fecol_);
   }

   void SetTestFECollVdim(int test_fec, int vdim)
   {
      test_fecols_vdims[test_fec] = vdim;
   }



   void SetSpaces(Array<FiniteElementSpace* > & fes_,
                  Array<FiniteElementCollection *> & fecol_)
   {
      trial_fes = fes_;
      test_fecols = fecol_;
      test_fecols_vdims.SetSize(test_fecols.Size());
      test_fecols_vdims = 1;
      nblocks = trial_fes.Size();
      mesh = trial_fes[0]->GetMesh();

      IsTraceFes.SetSize(nblocks);
      // Initialize with False
      IsTraceFes = false;
      for (int i = 0; i < nblocks; i++)
      {
         IsTraceFes[i] =
            (dynamic_cast<const H1_Trace_FECollection*>(trial_fes[i]->FEColl()) ||
             dynamic_cast<const ND_Trace_FECollection*>(trial_fes[i]->FEColl()) ||
             dynamic_cast<const RT_Trace_FECollection*>(trial_fes[i]->FEColl()));
      }
      Init();
   }

   // Get the size of the bilinear form of the DPGWeakForm
   int Size() const { return height; }

   // Pre-allocate the internal SparseMatrix before assembly.
   void AllocateMatrix() { if (mat == NULL) { AllocMat(); } }

   ///  Finalizes the matrix initialization.
   void Finalize(int skip_zeros = 1);

   /// Returns a reference to the sparse matrix:  \f$ M \f$
   BlockMatrix &BlockMat()
   {
      MFEM_VERIFY(mat, "mat is NULL and can't be dereferenced");
      return *mat;
   }

   /// Returns a reference to the sparse matrix of eliminated b.c.: \f$ M_e \f$
   BlockMatrix &BlockMatElim()
   {
      MFEM_VERIFY(mat_e, "mat_e is NULL and can't be dereferenced");
      return *mat_e;
   }

   /// Adds new Trial Integrator. Assumes ownership of @a bfi.
   void AddTrialIntegrator(BilinearFormIntegrator *bfi, int trial_fes,
                           int test_fes);

   /// Adds new Test Integrator. Assumes ownership of @a bfi.
   void AddTestIntegrator(BilinearFormIntegrator *bfi, int test_fes0,
                          int test_fes1);

   /// Adds new Domain LF Integrator. Assumes ownership of @a bfi.
   void AddDomainLFIntegrator(LinearFormIntegrator *bfi, int test_fes);

   /// Assembles the form i.e. sums over all integrators.
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

   virtual void RecoverFEMSolution(const Vector &X,Vector &x);

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

   void StoreMatrices(bool store_matrices_ = true)
   {
      store_matrices = store_matrices_;
      if (Bmat.Size() == 0)
      {
         Bmat.SetSize(mesh->GetNE());
         fvec.SetSize(mesh->GetNE());
         for (int i =0; i<mesh->GetNE(); i++)
         {
            Bmat[i] = nullptr;
            fvec[i] = nullptr;
         }
      }
   }

   void EnableStaticCondensation();

   Vector & ComputeResidual(const BlockVector & x);

   /// Destroys bilinear form.
   virtual ~DPGWeakForm();

};

} // namespace mfem

#endif
