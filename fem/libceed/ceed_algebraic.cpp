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

#include "ceed_algebraic.hpp"

#ifdef MFEM_USE_CEED
#include "../../fem/bilinearform.hpp"

namespace mfem
{

/// copy/paste hack
int CeedHackReallocArray(size_t n, size_t unit, void *p) {
  *(void **)p = realloc(*(void **)p, n*unit);
  if (n && unit && !*(void **)p)
    // LCOV_EXCL_START
    return CeedError(NULL, 1, "realloc failed to allocate %zd members of size "
                     "%zd\n", n, unit);
  // LCOV_EXCL_STOP

  return 0;
}

#define CeedHackRealloc(n, p) CeedHackReallocArray((n), sizeof(**(p)), p)

/// copy/paste hack
int CeedHackFree(void *p) {
  free(*(void **)p);
  *(void **)p = NULL;
  return 0;
}

/**
   Many parts of this are stolen from CeedOperatorAssembleLinearDiagonal_Ref()

   todo: think of ways to make this faster when we know a sparsity structure
   (ie, for low-order refined or algebraic sparsification)
*/
int CeedOperatorFullAssemble(CeedOperator op,
                             mfem::SparseMatrix ** mat) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);

  // Assemble QFunction
  CeedQFunction qf;
  ierr = CeedOperatorGetQFunction(op, &qf); CeedChk(ierr);
  CeedInt numinputfields, numoutputfields;
  ierr= CeedQFunctionGetNumArgs(qf, &numinputfields, &numoutputfields);
  CeedChk(ierr);
  CeedVector assembledqf;
  CeedElemRestriction rstr_q;
  ierr = CeedOperatorLinearAssembleQFunction(
    op, &assembledqf, &rstr_q, CEED_REQUEST_IMMEDIATE); CeedChk(ierr);

  CeedInt qflength;
  ierr = CeedVectorGetLength(assembledqf, &qflength); CeedChk(ierr);

  CeedOperatorField * input_fields;
  CeedOperatorField * output_fields;
  ierr = CeedOperatorGetFields(op, &input_fields, &output_fields); CeedChk(ierr);

  // Determine active input basis
  CeedQFunctionField *qffields;
  ierr = CeedQFunctionGetFields(qf, &qffields, NULL); CeedChk(ierr);
  CeedInt numemodein = 0, ncomp, dim = 1;
  CeedEvalMode *emodein = NULL;
  CeedBasis basisin = NULL;
  CeedElemRestriction rstrin = NULL;
  for (CeedInt i=0; i<numinputfields; i++) {
    CeedVector vec;
    ierr = CeedOperatorFieldGetVector(input_fields[i], &vec); CeedChk(ierr);
    if (vec == CEED_VECTOR_ACTIVE) {
      ierr = CeedOperatorFieldGetBasis(input_fields[i], &basisin);
      CeedChk(ierr);
      ierr = CeedBasisGetNumComponents(basisin, &ncomp); CeedChk(ierr);
      ierr = CeedBasisGetDimension(basisin, &dim); CeedChk(ierr);
      ierr = CeedOperatorFieldGetElemRestriction(input_fields[i], &rstrin);
      CeedChk(ierr);
      CeedEvalMode emode;
      ierr = CeedQFunctionFieldGetEvalMode(qffields[i], &emode);
      CeedChk(ierr);
      switch (emode) {
      case CEED_EVAL_NONE:
      case CEED_EVAL_INTERP:
        ierr = CeedHackRealloc(numemodein + 1, &emodein); CeedChk(ierr);
        emodein[numemodein] = emode;
        numemodein += 1;
        break;
      case CEED_EVAL_GRAD:
        ierr = CeedHackRealloc(numemodein + dim, &emodein); CeedChk(ierr);
        for (CeedInt d=0; d<dim; d++)
          emodein[numemodein+d] = emode;
        numemodein += dim;
        break;
      case CEED_EVAL_WEIGHT:
      case CEED_EVAL_DIV:
      case CEED_EVAL_CURL:
        break; // Caught by QF Assembly
      }
    }
  }

  // Determine active output basis
  ierr = CeedQFunctionGetFields(qf, NULL, &qffields); CeedChk(ierr);
  CeedInt numemodeout = 0;
  CeedEvalMode *emodeout = NULL;
  CeedBasis basisout = NULL;
  CeedElemRestriction rstrout = NULL;
  for (CeedInt i=0; i<numoutputfields; i++) {
    CeedVector vec;
    ierr = CeedOperatorFieldGetVector(output_fields[i], &vec); CeedChk(ierr);
    if (vec == CEED_VECTOR_ACTIVE) {
      ierr = CeedOperatorFieldGetBasis(output_fields[i], &basisout);
      CeedChk(ierr);
      ierr = CeedOperatorFieldGetElemRestriction(output_fields[i], &rstrout);
      CeedChk(ierr);
      CeedChk(ierr);
      CeedEvalMode emode;
      ierr = CeedQFunctionFieldGetEvalMode(qffields[i], &emode);
      CeedChk(ierr);
      switch (emode) {
      case CEED_EVAL_NONE:
      case CEED_EVAL_INTERP:
        ierr = CeedHackRealloc(numemodeout + 1, &emodeout); CeedChk(ierr);
        emodeout[numemodeout] = emode;
        numemodeout += 1;
        break;
      case CEED_EVAL_GRAD:
        ierr = CeedHackRealloc(numemodeout + dim, &emodeout); CeedChk(ierr);
        for (CeedInt d=0; d<dim; d++)
          emodeout[numemodeout+d] = emode;
        numemodeout += dim;
        break;
      case CEED_EVAL_WEIGHT:
      case CEED_EVAL_DIV:
      case CEED_EVAL_CURL:
        break; // Caught by QF Assembly
      }
    }
  }

  CeedInt nnodes, nelem, elemsize, nqpts;
  ierr = CeedElemRestrictionGetNumElements(rstrin, &nelem); CeedChk(ierr);
  ierr = CeedElemRestrictionGetElementSize(rstrin, &elemsize); CeedChk(ierr);
  ierr = CeedElemRestrictionGetLVectorSize(rstrin, &nnodes); CeedChk(ierr);
  ierr = CeedBasisGetNumQuadraturePoints(basisin, &nqpts); CeedChk(ierr);

  // Determine elem_dof relation
  CeedVector index_vec;
  ierr = CeedVectorCreate(ceed, nnodes, &index_vec); CeedChk(ierr);
  CeedScalar * array;
  ierr = CeedVectorGetArray(index_vec, CEED_MEM_HOST, &array); CeedChk(ierr);
  for (CeedInt i = 0; i < nnodes; ++i) {
    array[i] = i;
  }
  ierr = CeedVectorRestoreArray(index_vec, &array); CeedChk(ierr);
  CeedVector elem_dof;
  ierr = CeedVectorCreate(ceed, nelem * elemsize, &elem_dof); CeedChk(ierr);
  ierr = CeedVectorSetValue(elem_dof, 0.0); CeedChk(ierr);
  CeedElemRestrictionApply(rstrin, CEED_NOTRANSPOSE, index_vec,
                           elem_dof, CEED_REQUEST_IMMEDIATE); CeedChk(ierr);
  const CeedScalar * elem_dof_a;
  ierr = CeedVectorGetArrayRead(elem_dof, CEED_MEM_HOST, &elem_dof_a);
  CeedChk(ierr);
  ierr = CeedVectorDestroy(&index_vec); CeedChk(ierr);

  /// loop over elements and put in mfem::SparseMatrix
  mfem::SparseMatrix * out = new mfem::SparseMatrix(nnodes, nnodes);
  const CeedScalar *interpin, *gradin;
  ierr = CeedBasisGetInterp(basisin, &interpin); CeedChk(ierr);
  ierr = CeedBasisGetGrad(basisin, &gradin); CeedChk(ierr);

  const CeedScalar * assembledqfarray;
  ierr = CeedVectorGetArrayRead(assembledqf, CEED_MEM_HOST, &assembledqfarray);
  CeedChk(ierr);

  CeedInt layout[3];
  ierr = CeedElemRestrictionGetELayout(rstr_q, &layout); CeedChk(ierr);
  ierr = CeedElemRestrictionDestroy(&rstr_q); CeedChk(ierr);

  // numinputfields is 2 in both 2D and 3D...
  // elemsize and nqpts are total, not 1D
  const int skip_zeros = 0; // enforce structurally symmetric for later elimination
  MFEM_ASSERT(numemodein == numemodeout, "My undestanding fails in this case.");
  for (int e = 0; e < nelem; ++e) {
    /// get mfem::Array<int> for use in SparseMatrix::AddSubMatrix()
    mfem::Array<int> rows(elemsize);
    for (int i = 0; i < elemsize; ++i) {
      rows[i] = elem_dof_a[e * elemsize + i];
    }

    // form element matrix itself
    mfem::DenseMatrix Bmat(nqpts * numemodein, elemsize);
    Bmat = 0.0;
    mfem::DenseMatrix Dmat(nqpts * numemodeout,
                           nqpts * numemodein);
    Dmat = 0.0;
    mfem::DenseMatrix elem_mat(elemsize, elemsize);
    elem_mat = 0.0;
    for (int q = 0; q < nqpts; ++q) {
      for (int n = 0; n < elemsize; ++n) {
        CeedInt din = -1;
        for (int ein = 0; ein < numemodein; ++ein) {
          if (emodein[ein] == CEED_EVAL_GRAD) {
            din += 1;
          }
          if (emodein[ein] == CEED_EVAL_INTERP) {
            Bmat(numemodein * q + ein, n) += interpin[q * elemsize + n];
          } else if (emodein[ein] == CEED_EVAL_GRAD) {
            Bmat(numemodein * q + ein, n) += gradin[(din*nqpts+q) * elemsize + n];
          } else {
            MFEM_ASSERT(false, "Not implemented!");
          }
        }
      }
      for (int ei = 0; ei < numemodein; ++ei) {
        for (int ej = 0; ej < numemodein; ++ej) {
          const int comp = ei * numemodein + ej;
          const int index = q*layout[0] + comp*layout[1] + e*layout[2];
          Dmat(numemodein * q + ei, numemodein * q + ej) +=
            assembledqfarray[index];
        }
      }
    }
    mfem::DenseMatrix BTD(Bmat.Width(), Dmat.Width());
    mfem::MultAtB(Bmat, Dmat, BTD);
    mfem::Mult(BTD, Bmat, elem_mat);

    /// put element matrix in sparsemat
    out->AddSubMatrix(rows, rows, elem_mat, skip_zeros);
  }

  ierr = CeedVectorRestoreArrayRead(elem_dof, &elem_dof_a); CeedChk(ierr);
  ierr = CeedVectorDestroy(&elem_dof); CeedChk(ierr);
  ierr = CeedVectorRestoreArrayRead(assembledqf, &assembledqfarray); CeedChk(ierr);
  ierr = CeedVectorDestroy(&assembledqf); CeedChk(ierr);
  ierr = CeedHackFree(&emodein); CeedChk(ierr);
  ierr = CeedHackFree(&emodeout); CeedChk(ierr);

  out->Finalize(skip_zeros);
  *mat = out;

  return 0;
}

/// convenience function, ugly hack
mfem::HypreParMatrix* SerialHypreMatrix(mfem::SparseMatrix& mat)
{
   HYPRE_Int row_starts[3];
   row_starts[0] = 0;
   row_starts[1] = mat.Height();
   row_starts[2] = mat.Height();
   mfem::HypreParMatrix * out = new mfem::HypreParMatrix(
      MPI_COMM_WORLD, mat.Height(), row_starts, &mat);
   out->CopyRowStarts();
   out->CopyColStarts();

   /// 3 gives MFEM full ownership of i, j, data
   // out->SetOwnerFlags(3, out->OwnsOffd(), out->OwnsColMap());
   // mat.LoseData();

   return out;
}

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

   for (int i = 0; i < ess_tdof_list.Size(); ++i)
   {
      mat_assembled_->EliminateRowCol(ess_tdof_list[i], mfem::Matrix::DIAG_ONE);
   }
   innercg_.SetOperator(*mfem_ceed_);
  
#ifdef CEED_USE_AMGX
   if (use_amgx)
   {
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
                         mfem::Array<int>& ess_tdof_list,
                         int max_iter)
{
   mfem_ceed_ = new MFEMCeedOperator(oper, ess_tdof_list);
   height = width = mfem_ceed_->Height();

   innercg_.SetOperator(*mfem_ceed_);
   innercg_.SetPrintLevel(-1);
   innercg_.SetMaxIter(max_iter);
   innercg_.SetRelTol(0.0);
   innercg_.SetAbsTol(0.0);
}

CeedPlainCG::~CeedPlainCG()
{
   delete mfem_ceed_;
}

AlgebraicCeedSolver::AlgebraicCeedSolver(Operator& fine_mfem_op,
                                         BilinearForm& form, Array<int>& ess_dofs)
{
   int order = form.FESpace()->GetOrder(0);
   num_levels = 0;
   int current_order = order;
   while (current_order > 0)
   {
      num_levels++;
      current_order = current_order / 2;
   }

   auto *bffis = form.GetDBFI();
   MFEM_VERIFY(bffis->Size() == 1, "Only implemented for one integrator!");
   DiffusionIntegrator * dintegrator =
      dynamic_cast<DiffusionIntegrator*>((*bffis)[0]);
   MFEM_VERIFY(dintegrator, "Not a diffusion integrator!");
   CeedOperator current_op = dintegrator->GetCeedData()->oper;

   operators = new Operator*[num_levels];
   operators[0] = &fine_mfem_op;
   levels = new CeedMultigridLevel*[num_levels - 1];
   mfem::Array<int> * current_ess_dofs = &ess_dofs;
   current_order = order;
   for (int i = 0; i < num_levels - 1; ++i)
   {
      const int order_reduction = current_order - (current_order / 2);
      current_order = current_order / 2;
      levels[i] = new CeedMultigridLevel(current_op, *current_ess_dofs, order_reduction);
      current_op = levels[i]->GetCoarseCeed();
      current_ess_dofs = &levels[i]->GetCoarseEssentialDofList();
      operators[i + 1] = new MFEMCeedOperator(current_op, *current_ess_dofs);
   }
   mfem::Solver * coarsest_solver;
   CeedMultigridLevel * coarsest = levels[num_levels - 2];

   /*
   if (ceed_amg)
   {
      // bool use_amgx = (ceed_spec[1] == 'g'); // TODO very crude
      bool use_amgx = false;
      const int sparse_solver_type = 1; // single v-cycle
      coarsest_solver = new CeedCGWithAMG(coarsest->GetCoarseCeed(),
                                          coarsest->GetCoarseEssentialDofList(),
                                          sparse_solver_type,
                                          use_amgx);
   } else {
   */
   int coarse_cg_iterations = 10; // even less might be good
   coarsest_solver = new CeedPlainCG(coarsest->GetCoarseCeed(),
                                     coarsest->GetCoarseEssentialDofList(),
                                     coarse_cg_iterations);

   // loop up from coarsest to build V-cycle solvers
   solvers = new Solver*[num_levels];
   solvers[num_levels - 1] = coarsest_solver;
   for (int i = 0; i < num_levels - 1; ++i)
   {
      int index = num_levels - 2 - i;
      solvers[index] = new CeedMultigridVCycle(*levels[index], *operators[index],
                                               *solvers[index + 1]);
   }
}

AlgebraicCeedSolver::~AlgebraicCeedSolver()
{
   for (int i = 0; i < num_levels - 1; ++i)
   {
      delete solvers[i];
      delete operators[i + 1];
      delete levels[i];
   }
   delete solvers[num_levels - 1];
   delete [] solvers;
   delete [] operators;
   delete [] levels;
}

void AlgebraicCeedSolver::Mult(const Vector& x, Vector& y) const
{
   solvers[0]->Mult(x, y);
}

} // namespace mfem

#endif // MFEM_USE_CEED
