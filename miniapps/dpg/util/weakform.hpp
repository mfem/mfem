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

#ifndef MFEM_DPGWEAKFORM
#define MFEM_DPGWEAKFORM

#include "mfem.hpp"
#include "blockstaticcond.hpp"

namespace mfem
{

/** @brief Class representing the DPG weak formulation.
    Given the variational formulation
                  a(u,v) = b(v), (or A u = b, where <Au,v> = a(u,v))
    this class forms the DPG linear system
                        A^T G^-1 A u  = A^T G^-1 b
    This system results from the minimum residual formulation
                        u = argmin_w ||G^-1(b - Aw)||.
    Here G is a symmetic positive definite matrix resulting from the discretization of
    the Riesz operator on the test space. Since the test space is broken
    (discontinuous),  G is defined and inverted element-wise and the assembly
    of the global system is performed in the same manner as the standard FEM method.
    Note that DPGWeakForm can handle multiple Finite Element spaces.*/
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

   /// Block vector \f$ y \f$ to be associated with the Block linear form
   BlockVector * y = nullptr;

   /** @brief Block Matrix \f$ M_e \f$ used to store the eliminations
        from the b.c.  Owned.
       \f$ M + M_e = M_{original} \f$ */
   BlockMatrix *mat_e = nullptr;

   /// Trial FE spaces
   Array<FiniteElementSpace * > trial_fes;

   /// Flags to determine if a FiniteElementSpace is Trace
   Array<int> IsTraceFes;

   /// Test FE Collections (Broken)
   Array<FiniteElementCollection *> test_fecols;
   Array<int> test_fecols_vdims;

   /// Set of Trial Integrators to be applied for matrix B.
   Array2D<Array<BilinearFormIntegrator * > * > trial_integs;

   /// Set of Test Space (broken) Integrators to be applied for matrix G
   Array2D<Array<BilinearFormIntegrator * > * > test_integs;

   /// Set of Linear Form Integrators to be applied.
   Array<Array<LinearFormIntegrator * > * > lfis;

   /// Block Prolongation
   BlockMatrix * P = nullptr;
   /// Block Restriction
   BlockMatrix * R = nullptr;

   mfem::Operator::DiagonalPolicy diag_policy;

   void Init();
   void ReleaseInitMemory();

   /// Allocate appropriate BlockMatrix and assign it to mat
   void AllocMat();

   void ConformingAssemble();

   void ComputeOffsets();

   virtual void BuildProlongation();

   bool store_matrices = false;

   /** Store the matrix L^-1 B  and Vector L^-1 l
       where G = L L^t */
   Array<DenseMatrix * > Bmat;
   Array<Vector * > fvec;
   Vector residuals;

public:
   /// Default constructor. User must call SetSpaces to setup the FE spaces
   DPGWeakForm()
   {
      height = 0;
      width = 0;
   }

   /// Creates bilinear form associated with FE spaces @a fes_.
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
      for (int i = 0; i < nblocks; i++)
      {
         IsTraceFes[i] =
            (dynamic_cast<const H1_Trace_FECollection*>(trial_fes[i]->FEColl()) ||
             dynamic_cast<const ND_Trace_FECollection*>(trial_fes[i]->FEColl()) ||
             dynamic_cast<const RT_Trace_FECollection*>(trial_fes[i]->FEColl()));
      }
      Init();
   }

   /// Get the size of the bilinear form of the DPGWeakForm
   int Size() const { return height; }

   /// Pre-allocate the internal BlockMatrix before assembly.
   void AllocateMatrix() { if (mat == nullptr) { AllocMat(); } }

   ///  Finalizes the matrix initialization.
   void Finalize(int skip_zeros = 1);

   /// Returns a reference to the BlockMatrix:  \f$ M \f$
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
   void AddTrialIntegrator(BilinearFormIntegrator *bfi, int n, int m);

   /// Adds new Test Integrator. Assumes ownership of @a bfi.
   void AddTestIntegrator(BilinearFormIntegrator *bfi, int n, int m);

   /// Adds new Domain LF Integrator. Assumes ownership of @a bfi.
   void AddDomainLFIntegrator(LinearFormIntegrator *lfi, int n);

   /// Assembles the form i.e. sums over all integrators.
   void Assemble(int skip_zeros = 1);

   /** @brief Form the linear system A X = B, corresponding to this DPG weak
       form */
   /** This method applies any necessary transformations to the linear system
       such as: eliminating boundary conditions; applying conforming constraints
       for non-conforming AMR; static condensation;

       The GridFunction-size vector @a x must contain the essential b.c. The
       DPGWeakForm must be assembled.

       The vector @a X is initialized with a suitable initial guess: the essential
       entries of @a X are set to the corresponding b.c. and all other entries
       are set to zero (@a copy_interior == 0) or copied from @a x
       (@a copy_interior != 0).

       After solving the linear system, the finite element solution @a x can be
       recovered by calling RecoverFEMSolution() (with the same vectors @a X,
       and @a x).

       NOTE: If there are no transformations, @a X simply reuses the data of
             @a x. */
   virtual void FormLinearSystem(const Array<int> &ess_tdof_list, Vector &x,
                                 OperatorHandle &A, Vector &X,
                                 Vector &B, int copy_interior = 0);

   /** @brief Form the linear system A X = B, corresponding to this DPG weak form
       Version of the method FormLinearSystem() where the system matrix is
       returned in the variable @a A, of type OpType, holding a *reference* to
       the system matrix (created with the method OpType::MakeRef()). */
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

   /// Form the linear system matrix @a A, see FormLinearSystem() for details.
   virtual void FormSystemMatrix(const Array<int> &ess_tdof_list,
                                 OperatorHandle &A);

   /// Form the linear system matrix A, see FormLinearSystem() for details.
   /** Version of the method FormSystemMatrix() where the system matrix is
       returned in the variable @a A, of type OpType, holding a *reference* to
       the system matrix (created with the method OpType::MakeRef()). */
   template <typename OpType>
   void FormSystemMatrix(const Array<int> &ess_tdof_list, OpType &A)
   {
      OperatorHandle Ah;
      FormSystemMatrix(ess_tdof_list, Ah);
      OpType *A_ptr = Ah.Is<OpType>();
      MFEM_VERIFY(A_ptr, "invalid OpType used");
      A.MakeRef(*A_ptr);
   }

   /// Eliminate the given @a vdofs, storing the eliminated part internally in \f$ M_e \f$.
   /** This method works in conjunction with EliminateVDofsInRHS() and allows
       elimination of boundary conditions in multiple right-hand sides. In this
       method, @a vdofs is a list of DOFs. */
   void EliminateVDofs(const Array<int> &vdofs,
                       Operator::DiagonalPolicy dpolicy = Operator::DIAG_ONE);

   /** @brief Use the stored eliminated part of the matrix (see
       EliminateVDofs(const Array<int> &, DiagonalPolicy)) to modify the r.h.s.
       @a b; @a vdofs is a list of DOFs. */
   void EliminateVDofsInRHS(const Array<int> &vdofs, const Vector &x, Vector &b);

   /// Recover the solution of a linear system formed with FormLinearSystem().
   /** Call this method after solving a linear system constructed using the
       FormLinearSystem() method to recover the solution as a GridFunction-size
       vector in @a x. Use the same arguments as in the FormLinearSystem() call.
   */
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

   /// Update the DPGWeakForm after mesh modifications (AMR)
   virtual void Update();

   /// Store internal element matrices used for computation of residual after solve
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

   /// Compute DPG residual based error estimator
   Vector & ComputeResidual(const BlockVector & x);

   virtual ~DPGWeakForm();

};

} // namespace mfem

#endif
