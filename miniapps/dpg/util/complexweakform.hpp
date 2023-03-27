// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_COMPLEX_DPGWEAKFORM
#define MFEM_COMPLEX_DPGWEAKFORM

#include "mfem.hpp"
#include "complexstaticcond.hpp"

namespace mfem
{

/** @brief Class representing the DPG weak formulation for complex valued systems
    (see the class DPGWeakForm). */
class ComplexDPGWeakForm
{

protected:

   ComplexBlockStaticCondensation *static_cond; ///< Owned.

   bool initialized = false;

   Mesh * mesh = nullptr;
   int height, width;
   int nblocks;
   Array<int> dof_offsets;
   Array<int> tdof_offsets;

   /// Block matrix \f$ M \f$ to be associated with the real/imag Block bilinear form. Owned.
   BlockMatrix *mat_r = nullptr;
   BlockMatrix *mat_i = nullptr;
   ComplexOperator * mat = nullptr;

   /// BlockVectors to be associated with the real/imag Block linear form
   BlockVector * y_r = nullptr;
   BlockVector * y_i = nullptr;
   Vector * y = nullptr;

   /** @brief Block Matrix \f$ M_e \f$ used to store the eliminations
        from the b.c.  Owned.
       \f$ M + M_e = M_{original} \f$ */
   BlockMatrix *mat_e_r = nullptr;
   BlockMatrix *mat_e_i = nullptr;

   /// Trial FE spaces
   Array<FiniteElementSpace * > trial_fes;

   /// Flags to determine if a FiniteElementSpace is Trace
   Array<int> IsTraceFes;

   /// Test FE Collections (Broken)
   Array<FiniteElementCollection *> test_fecols;
   Array<int> test_fecols_vdims;

   /// Set of Trial Integrators to be applied for matrix B
   Array2D<Array<BilinearFormIntegrator * > * > trial_integs_r;
   Array2D<Array<BilinearFormIntegrator * > * > trial_integs_i;

   /// Set of Test Space (broken) Integrators to be applied for matrix G
   Array2D<Array<BilinearFormIntegrator * > * > test_integs_r;
   Array2D<Array<BilinearFormIntegrator * > * > test_integs_i;

   /// Set of Liniear Froem Integrators to be applied.
   Array<Array<LinearFormIntegrator * > * > lfis_r;
   Array<Array<LinearFormIntegrator * > * > lfis_i;

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

   bool store_matrices = false;

   /** Store the matrix L^-1 B  and Vector L^-1 l
       where G = L L^t */
   Array<ComplexDenseMatrix * > Bmat;
   Array<Vector * > fvec;
   Vector residuals;

private:

public:

   ComplexDPGWeakForm()
   {
      height = 0;
      width = 0;
   }

   /// Creates bilinear form associated with FE spaces @a fes_.
   ComplexDPGWeakForm(Array<FiniteElementSpace* > & fes_,
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

   // Get the size of the bilinear form of the ComplexDPGWeakForm
   int Size() const { return height; }

   // Pre-allocate the internal real and imag BlockMatrix before assembly.
   void AllocateMatrix() { if (mat_r == nullptr) { AllocMat(); } }

   ///  Finalizes the matrix initialization.
   void Finalize(int skip_zeros = 1);

   /// Returns a reference to the BlockMatrix:  \f$ M_r \f$
   BlockMatrix &BlockMat_r()
   {
      MFEM_VERIFY(mat_r, "mat_r is NULL and can't be dereferenced");
      return *mat_r;
   }
   /// Returns a reference to the BlockMatrix:  \f$ M_i \f$
   BlockMatrix &BlockMat_i()
   {
      MFEM_VERIFY(mat_i, "mat_i is NULL and can't be dereferenced");
      return *mat_i;
   }

   /// Returns a reference to the BlockMatrix of eliminated b.c.: \f$ M_e_r \f$
   BlockMatrix &BlockMatElim_r()
   {
      MFEM_VERIFY(mat_e_r, "mat_e is NULL and can't be dereferenced");
      return *mat_e_r;
   }

   /// Returns a reference to the BlockMatrix of eliminated b.c.: \f$ M_e_i \f$
   BlockMatrix &BlockMatElim_i()
   {
      MFEM_VERIFY(mat_e_i, "mat_e is NULL and can't be dereferenced");
      return *mat_e_i;
   }

   /** Adds new Trial Integrator. Assumes ownership of @a bfi_r and @a bfi_i.
       @a n and @a m correspond to the trial FESpace and test FEColl
       respectively */
   void AddTrialIntegrator(BilinearFormIntegrator *bfi_r,
                           BilinearFormIntegrator *bfi_i,
                           int n, int m);

   /// Adds new Test Integrator. Assumes ownership of @a bfi_r and @a bfi_i.
   void AddTestIntegrator(BilinearFormIntegrator *bfi_r,
                          BilinearFormIntegrator *bfi_i,
                          int n, int m);

   /// Adds new Domain LF Integrator. Assumes ownership of @a lfi_r and lfi_i.
   void AddDomainLFIntegrator(LinearFormIntegrator *lfi_r,
                              LinearFormIntegrator *lfi_i,
                              int n);

   /// Assembles the form i.e. sums over all integrators.
   void Assemble(int skip_zeros = 1);

   virtual void FormLinearSystem(const Array<int> &ess_tdof_list,
                                 Vector &x, OperatorHandle & A,
                                 Vector &X, Vector &B,
                                 int copy_interior = 0);

   template <typename OpType>
   void FormLinearSystem(const Array<int> &ess_tdof_list,
                         Vector &x, OpType &A,
                         Vector &X, Vector &B,
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

   void EliminateVDofsInRHS(const Array<int> &vdofs,
                            const Vector &x_r, const Vector & x_i,
                            Vector &b_r, Vector & b_i);

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

   Vector & ComputeResidual(const Vector & x);

   /// Destroys bilinear form.
   virtual ~ComplexDPGWeakForm();

};

} // namespace mfem

#endif
