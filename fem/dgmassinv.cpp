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

#include "dgmassinv.hpp"
#include "bilinearform.hpp"
#include "dgmassinv_kernels.hpp"

namespace mfem
{

struct DGMassInvKernels { DGMassInvKernels(); };

DGMassInverse::DGMassInverse(const FiniteElementSpace &fes_orig,
                             Coefficient *coeff,
                             const IntegrationRule *ir,
                             int btype)
   : Solver(fes_orig.GetTrueVSize()),
     fec(fes_orig.GetMaxElementOrder(),
         fes_orig.GetMesh()->Dimension(),
         btype,
         fes_orig.GetTypicalFE()->GetMapType()),
     fes(fes_orig.GetMesh(), &fec)
{
   static DGMassInvKernels kernels;

   MFEM_VERIFY(fes.IsDGSpace(), "Space must be DG.");
   MFEM_VERIFY(!fes.IsVariableOrder(), "Variable orders not supported.");

   const int btype_orig =
      static_cast<const L2_FECollection*>(fes_orig.FEColl())->GetBasisType();

   if (btype_orig == btype)
   {
      // No change of basis required
      d2q = nullptr;
   }
   else
   {
      // original basis to solver basis
      const auto mode = DofToQuad::TENSOR;
      const FiniteElement &fe_orig = *fes_orig.GetTypicalFE();
      const FiniteElement &fe = *fes.GetTypicalFE();
      d2q = &fe_orig.GetDofToQuad(fe.GetNodes(), mode);

      const int n = d2q->ndof;
      Array<real_t> B_inv = d2q->B; // deep copy
      Array<int> ipiv(n);
      // solver basis to original
      LUFactors lu(B_inv.HostReadWrite(), ipiv.HostWrite());
      lu.Factor(n);
      B_.SetSize(n*n);
      lu.GetInverseMatrix(n, B_.HostWrite());
      Bt_.SetSize(n*n);
      DenseMatrix B_matrix(B_.HostReadWrite(), n, n);
      DenseMatrix Bt_matrix(Bt_.HostWrite(), n, n);
      Bt_matrix.Transpose(B_matrix);
   }

   if (coeff) { m = new MassIntegrator(*coeff, ir); }
   else { m = new MassIntegrator(ir); }

   diag_inv.SetSize(height);
   // Workspace vectors used for CG
   r_.SetSize(height);
   d_.SetSize(height);
   z_.SetSize(height);
   // Only need transformed RHS if basis is different
   if (btype_orig != btype) { b2_.SetSize(height); }

   M.reset(new BilinearForm(&fes));
   M->AddDomainIntegrator(m); // M assumes ownership of m
   M->SetAssemblyLevel(AssemblyLevel::PARTIAL);

   // Assemble the bilinear form and its diagonal (for preconditioning).
   Update();
}

DGMassInverse::DGMassInverse(const FiniteElementSpace &fes_, Coefficient &coeff,
                             int btype)
   : DGMassInverse(fes_, &coeff, nullptr, btype) { }

DGMassInverse::DGMassInverse(const FiniteElementSpace &fes_, Coefficient &coeff,
                             const IntegrationRule &ir, int btype)
   : DGMassInverse(fes_, &coeff, &ir, btype) { }

DGMassInverse::DGMassInverse(const FiniteElementSpace &fes_,
                             const IntegrationRule &ir, int btype)
   : DGMassInverse(fes_, nullptr, &ir, btype) { }

DGMassInverse::DGMassInverse(const FiniteElementSpace &fes_, int btype)
   : DGMassInverse(fes_, nullptr, nullptr, btype) { }

void DGMassInverse::SetOperator(const Operator &op)
{
   MFEM_ABORT("SetOperator not supported with DGMassInverse.")
}

void DGMassInverse::SetRelTol(const real_t rel_tol_) { rel_tol = rel_tol_; }

void DGMassInverse::SetAbsTol(const real_t abs_tol_) { abs_tol = abs_tol_; }

void DGMassInverse::SetMaxIter(const int max_iter_) { max_iter = max_iter_; }

void DGMassInverse::Update()
{
   M->Assemble();
   M->AssembleDiagonal(diag_inv);
   diag_inv.Reciprocal();
}

DGMassInverse::~DGMassInverse() = default;

void DGMassInverse::Mult(const Vector &Mu, Vector &u) const
{
   // Dispatch to templated version based on dim, d1d, and q1d.
   const int dim = fes.GetMesh()->Dimension();
   const int d1d = m->dofs1D;
   const int q1d = m->quad1D;

   CGKernels::Run(dim, d1d, q1d, *this, Mu, u);
}

DGMassInvKernels::DGMassInvKernels()
{
   using k = DGMassInverse::CGKernels;
   // 2D
   k::Specialization<2,1,1>::Add();
   k::Specialization<2,2,2>::Add();
   k::Specialization<2,3,3>::Add();
   k::Specialization<2,3,5>::Add();
   k::Specialization<2,4,4>::Add();
   k::Specialization<2,4,6>::Add();
   k::Specialization<2,5,5>::Add();
   k::Specialization<2,5,7>::Add();
   k::Specialization<2,6,6>::Add();
   k::Specialization<2,6,8>::Add();
   // 3D
   k::Specialization<3,2,2>::Add();
   k::Specialization<3,2,3>::Add();
   k::Specialization<3,3,3>::Add();
   k::Specialization<3,3,4>::Add();
   k::Specialization<3,3,5>::Add();
   k::Specialization<3,4,4>::Add();
   k::Specialization<3,4,5>::Add();
   k::Specialization<3,4,6>::Add();
   k::Specialization<3,4,8>::Add();
   k::Specialization<3,5,5>::Add();
   k::Specialization<3,5,6>::Add();
   k::Specialization<3,5,7>::Add();
   k::Specialization<3,5,8>::Add();
   k::Specialization<3,6,6>::Add();
   k::Specialization<3,6,7>::Add();
}

} // namespace mfem
