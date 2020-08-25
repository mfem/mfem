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

#include "mfem.hpp"
#include "ceed_algebraic.hpp"

namespace mfem
{
namespace navier
{

UnconstrainedMFEMCeedOperator::UnconstrainedMFEMCeedOperator(CeedOperator oper) :
   oper_(oper)
{
   int ierr = 0;
   Ceed ceed;
   ierr += CeedOperatorGetCeed(oper, &ceed);
   CeedElemRestriction er;
   ierr += CeedOperatorGetActiveElemRestriction(oper, &er);
   int s;
   ierr += CeedElemRestrictionGetLVectorSize(er, &s);
   height = width = s;
   ierr += CeedVectorCreate(ceed, height, &v_);
   ierr += CeedVectorCreate(ceed, width, &u_);
   MFEM_ASSERT(ierr == 0, "CEED error");
}

UnconstrainedMFEMCeedOperator::~UnconstrainedMFEMCeedOperator()
{
   int ierr = 0;
   ierr += CeedVectorDestroy(&v_);
   ierr += CeedVectorDestroy(&u_);
   MFEM_ASSERT(ierr == 0, "CEED error");
}

void UnconstrainedMFEMCeedOperator::Mult(const mfem::Vector& x, mfem::Vector& y) const
{
   int ierr = 0;

   ierr += CeedVectorSetArray(u_, CEED_MEM_HOST, CEED_USE_POINTER, x.GetData());
   ierr += CeedVectorSetArray(v_, CEED_MEM_HOST, CEED_USE_POINTER, y.GetData());

   ierr += CeedOperatorApply(oper_, u_, v_, CEED_REQUEST_IMMEDIATE);
   ierr += CeedVectorSyncArray(v_, CEED_MEM_HOST);

   MFEM_ASSERT(ierr == 0, "CEED error");
}

MFEMCeedJacobi::MFEMCeedJacobi(Ceed ceed,
                               int size,
                               CeedVector diagonal,
                               const mfem::Array<int>& ess_tdof_list,
                               double scale)
   :
   mfem::Operator(size, size),
   ess_tdof_list_(ess_tdof_list) {
   int ierr = 0;
   ierr += CeedVectorCreate(ceed, height, &v_);
   ierr += CeedVectorCreate(ceed, width, &u_);
   ierr += CeedVectorCreate(ceed, size, &inv_diag_);
   const CeedScalar *diag_data;
   CeedScalar *inv_diag_data;
   ierr += CeedVectorGetArrayRead(diagonal, CEED_MEM_HOST, &diag_data);
   ierr += CeedVectorGetArray(inv_diag_, CEED_MEM_HOST, &inv_diag_data);
   for (int i = 0; i < size; ++i)
   {
      MFEM_ASSERT(diag_data[i] > 0.0, "Not positive definite!");
      inv_diag_data[i] = scale / diag_data[i];
   }
   ierr += CeedVectorRestoreArray(inv_diag_, &inv_diag_data);
   ierr += CeedVectorRestoreArrayRead(diagonal, &diag_data);

   MFEM_ASSERT(ierr == 0, "CEED error");
}

MFEMCeedJacobi::~MFEMCeedJacobi() {
  CeedVectorDestroy(&v_);
  CeedVectorDestroy(&u_);
  CeedVectorDestroy(&inv_diag_);
}

void MFEMCeedJacobi::Mult(const mfem::Vector& x, mfem::Vector& y) const {
  int ierr = 0;

  // TODO: following line should be done in CEED / on GPU?
  y = x;

  ierr += CeedVectorSetArray(u_, CEED_MEM_HOST, CEED_USE_POINTER, x.GetData());
  ierr += CeedVectorSetArray(v_, CEED_MEM_HOST, CEED_USE_POINTER, y.GetData());

  ierr += CeedVectorPointwiseMult(v_, inv_diag_);

  ierr += CeedVectorSyncArray(v_, CEED_MEM_HOST);

  MFEM_ASSERT(ierr == 0, "CEED error");
}

void MFEMCeedJacobi::MultTranspose(const mfem::Vector& x, mfem::Vector& y) const
{
   Mult(x, y);
}

MFEMCeedVCycle::MFEMCeedVCycle(const mfem::Operator& fine_operator,
               const mfem::Solver& coarse_solver,
               const mfem::Operator& fine_smoother,
               const mfem::Operator& interp) :
  fine_operator_(fine_operator),
  coarse_solver_(coarse_solver),
  fine_smoother_(fine_smoother),
  interp_(interp)
{
   MFEM_VERIFY(fine_operator_.Height() == interp_.Height(), "Sizes don't match!");
   MFEM_VERIFY(coarse_solver_.Height() == interp_.Width(), "Sizes don't match!");

   residual_.SetSize(fine_operator_.Height());
   correction_.SetSize(fine_operator_.Height());
   coarse_residual_.SetSize(coarse_solver_.Height());
   coarse_correction_.SetSize(coarse_solver_.Height());
}

void MFEMCeedVCycle::FormResidual(const mfem::Vector& b,
                                  const mfem::Vector& x,
                                  mfem::Vector& r) const
{
   fine_operator_.Mult(x, r);
   r *= -1.0;
   r += b;
}

void MFEMCeedVCycle::Mult(const mfem::Vector& b, mfem::Vector& x) const
{
   x = 0.0;
   fine_smoother_.Mult(b, correction_);
   x += correction_;

   FormResidual(b, x, residual_);
   interp_.MultTranspose(residual_, coarse_residual_);
   coarse_correction_ = 0.0;
   coarse_solver_.Mult(coarse_residual_, coarse_correction_);
   interp_.Mult(coarse_correction_, correction_);
   x += correction_;

   FormResidual(b, x, residual_);
   fine_smoother_.Mult(residual_, correction_);
   x += correction_;
}

int MFEMCeedInterpolation::Initialize(
  Ceed ceed, CeedBasis basisctof,
  CeedElemRestriction erestrictu_coarse, CeedElemRestriction erestrictu_fine)
{
   int ierr = 0;

   ierr = CeedInterpolationCreate(ceed, basisctof, erestrictu_coarse,
                                  erestrictu_fine, &ceed_interp_); CeedChk(ierr);

   ierr = CeedVectorCreate(ceed, height, &v_); CeedChk(ierr);
   ierr = CeedVectorCreate(ceed, width, &u_); CeedChk(ierr);

   return 0;
}

MFEMCeedInterpolation::MFEMCeedInterpolation(
   Ceed ceed, CeedBasis basisctof,
   CeedElemRestriction erestrictu_coarse,
   CeedElemRestriction erestrictu_fine)
{
   int lo_nldofs, ho_nldofs;
   CeedElemRestrictionGetLVectorSize(erestrictu_coarse, &lo_nldofs);
   CeedElemRestrictionGetLVectorSize(erestrictu_fine, &ho_nldofs);
   height = ho_nldofs;
   width = lo_nldofs;
   owns_basis_ = false;
   Initialize(ceed, basisctof, erestrictu_coarse, erestrictu_fine);
}
  

MFEMCeedInterpolation::MFEMCeedInterpolation(
   Ceed ceed,
   mfem::FiniteElementSpace& lo_fespace,
   mfem::FiniteElementSpace& ho_fespace,
   CeedElemRestriction erestrictu_coarse,
   CeedElemRestriction erestrictu_fine)
   :
   mfem::Operator(ho_fespace.GetNDofs(), lo_fespace.GetNDofs())
{
   const int dim = ho_fespace.GetMesh()->Dimension();
   const int order = ho_fespace.GetOrder(0);
   const int low_order = lo_fespace.GetOrder(0);
   const int bp3_ncompu = 1;

   // P coarse and P fine (P is number of nodal points = degree + 1)
   CeedInt Pc = low_order + 1;
   CeedInt Pf = order + 1;

   // Basis
   // TODO: would like to use CeedBasisCreateTensorH1 (general)
   // without Lagrange assumption
   CeedBasis basisctof;
   CeedBasisCreateTensorH1Lagrange(ceed, dim, bp3_ncompu, Pc, Pf,
                                   CEED_GAUSS_LOBATTO, &basisctof);
   owns_basis_ = true;
   Initialize(ceed, basisctof, erestrictu_coarse, erestrictu_fine);
   basisctof_ = basisctof;
}

MFEMCeedInterpolation::~MFEMCeedInterpolation()
{
   CeedVectorDestroy(&v_);
   CeedVectorDestroy(&u_);
   if (owns_basis_)
   {
      CeedBasisDestroy(&basisctof_);
   }
   CeedInterpolationDestroy(&ceed_interp_);
}

void MFEMCeedInterpolation::Mult(const mfem::Vector& x, mfem::Vector& y) const
{
   int ierr = 0;

   ierr += CeedVectorSetArray(u_, CEED_MEM_HOST, CEED_USE_POINTER, x.GetData());
   ierr += CeedVectorSetArray(v_, CEED_MEM_HOST, CEED_USE_POINTER, y.GetData());

   ierr += CeedInterpolationInterpolate(ceed_interp_, u_, v_);

   ierr += CeedVectorSyncArray(v_, CEED_MEM_HOST);

   MFEM_ASSERT(ierr == 0, "CEED error");
}

void MFEMCeedInterpolation::MultTranspose(const mfem::Vector& x,
                                          mfem::Vector& y) const
{
   int ierr = 0;

   ierr += CeedVectorSetArray(v_, CEED_MEM_HOST, CEED_USE_POINTER, x.GetData());
   ierr += CeedVectorSetArray(u_, CEED_MEM_HOST, CEED_USE_POINTER, y.GetData());

   ierr += CeedInterpolationRestrict(ceed_interp_, v_, u_);

   ierr += CeedVectorSyncArray(u_, CEED_MEM_HOST);

   MFEM_ASSERT(ierr == 0, "CEED error");
}

void CoarsenEssentialDofs(const mfem::Operator& mfem_interp,
                          const mfem::Array<int>& ho_ess_tdof_list,
                          mfem::Array<int>& alg_lo_ess_tdof_list)
{
   mfem::Vector ho_boundary_ones(mfem_interp.Height());
   ho_boundary_ones = 0.0;
   for (int k : ho_ess_tdof_list)
   {
      ho_boundary_ones(k) = 1.0;
   }
   mfem::Vector lo_boundary_ones(mfem_interp.Width());
   mfem_interp.MultTranspose(ho_boundary_ones, lo_boundary_ones);
   for (int i = 0; i < lo_boundary_ones.Size(); ++i)
   {
      if (lo_boundary_ones(i) > 0.9)
      {
         alg_lo_ess_tdof_list.Append(i);
      }
   }
}

/// probably don't need both order and dim
/// (probably don't need either; infer order from size of B1d,
/// infer dim from that and er->elemsize)
CeedMultigridLevel::CeedMultigridLevel(CeedOperator oper,
                                       const mfem::Array<int>& ho_ess_tdof_list,
                                       int order_reduction)
   :
   oper_(oper)
{
   const double jacobi_scale = 0.65; // TODO: separate construction?
   Ceed ceed;
   CeedOperatorGetCeed(oper, &ceed);
   CeedATPMGBundle(oper, order_reduction, &coarse_basis_, &basisctof_,
                   &lo_er_, &coarse_oper_);

   // this is a local diagonal, in the sense of l-vector
   CeedVector diagceed;
   int length;
   CeedOperatorGetSize(oper, &length);
   CeedVectorCreate(ceed, length, &diagceed);
   CeedVectorSetValue(diagceed, 0.0);
   CeedOperatorLinearAssembleDiagonal(oper, diagceed, CEED_REQUEST_IMMEDIATE);
   nobc_smoother_ = new MFEMCeedJacobi(ceed, length, diagceed,
                                       ho_ess_tdof_list, jacobi_scale);
   nobc_smoother_->FormSystemOperator(ho_ess_tdof_list, smoother_);
   CeedVectorDestroy(&diagceed);

   CeedOperatorGetActiveElemRestriction(oper, &ho_er_);
   mfem_interp_ = new MFEMCeedInterpolation(ceed, basisctof_, lo_er_, ho_er_);

   CoarsenEssentialDofs(*mfem_interp_, ho_ess_tdof_list, lo_ess_tdof_list_);
}

CeedMultigridLevel::~CeedMultigridLevel()
{
   CeedOperatorDestroy(&coarse_oper_);
   CeedBasisDestroy(&coarse_basis_);
   CeedBasisDestroy(&basisctof_);
   CeedElemRestrictionDestroy(&lo_er_);

   delete nobc_smoother_;
   delete smoother_;
   delete mfem_interp_;
}

CeedMultigridVCycle::CeedMultigridVCycle(
   const CeedMultigridLevel& level,
   const mfem::Operator& fine_operator,
   const mfem::Solver& coarse_solver)
   :
   mfem::Solver(fine_operator.Height()),
   cycle_(fine_operator, coarse_solver, *level.smoother_, *level.mfem_interp_)
{
}

void CeedMultigridVCycle::Mult(const mfem::Vector& x, mfem::Vector& y) const
{
   cycle_.Mult(x, y);
}

/*
CeedCGWithAMG::CeedCGWithAMG(CeedOperator oper,
                             mfem::Array<int>& ess_tdof_list,
                             int sparse_solver_type,
                             bool use_amgx)
{
   mfem_ceed_ = new MFEMCeedOperator(oper, ess_tdof_list);
   height = width = mfem_ceed_->Height();

   CeedOperatorFullAssemble(oper, &mat_assembled_);

   // // todo: interface for this, will eventually matter!
   // int sparsified_nnz = mat_assembled->NumNonZeroElems();
   // std::cout << "Coarse operator NNZ: " << sparsified_nnz << std::endl;
   // std::cout << "Estimated complexity: "
   //    << (double) (ho_estimated_nnz + sparsified_nnz) /
   // (double) ho_estimated_nnz << std::endl;
   for (int i = 0; i < ess_tdof_list.Size(); ++i) {
      mat_assembled_->EliminateRowCol(ess_tdof_list[i], mfem::Matrix::DIAG_ONE);
   }
   innercg_.SetOperator(*mfem_ceed_);
  
#ifdef CEED_USE_AMGX
   if (use_amgx) {
      NvidiaAMGX * amgx = new NvidiaAMGX();
      const bool amgx_verbose = false;
      amgx->ConfigureAsPreconditioner(amgx_verbose);
      amgx->SetOperator(*mat_assembled_);
      hypre_assembled_ = NULL;
      inner_prec_ = amgx;
   } else
#endif
   {
      hypre_assembled_ = SerialHypreMatrix(*mat_assembled_);
      mfem::HypreBoomerAMG * amg = new mfem::HypreBoomerAMG(*hypre_assembled_);
      amg->SetPrintLevel(0);
      inner_prec_ = amg;
   }
   innercg_.SetPreconditioner(*inner_prec_);
   innercg_.SetPrintLevel(-1);
   innercg_.SetMaxIter(500);
   innercg_.SetRelTol(1.e-16);

   if (sparse_solver_type == 0)
   {
      solver_ = &innercg_;
   }
   else
   {
      solver_ = inner_prec_;
   }
}

CeedCGWithAMG::~CeedCGWithAMG()
{
   delete mfem_ceed_;

   delete mat_assembled_;
   delete hypre_assembled_;
   delete inner_prec_;
}
*/

CeedPlainCG::CeedPlainCG(CeedOperator oper,
                         mfem::Array<int>& ess_tdof_list)
{
   mfem_ceed_ = new MFEMCeedOperator(oper, ess_tdof_list);
   height = width = mfem_ceed_->Height();

   innercg_.SetOperator(*mfem_ceed_);
   innercg_.SetPrintLevel(-1);
   innercg_.SetMaxIter(500);
   innercg_.SetRelTol(1.e-16);
}

CeedPlainCG::~CeedPlainCG()
{
   delete mfem_ceed_;
}

} // namespace navier
} // namespace mfem
