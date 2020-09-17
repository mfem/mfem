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

#include "ceed_algebraic_navier.hpp"
#include "mfem.hpp"

namespace mfem
{
namespace navier
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

CeedPlainCG::CeedPlainCG(CeedOperator oper,
                         mfem::Array<int>& ess_tdof_list,
                         int max_iter)
{
   mfem_ceed_ = new MFEMCeedOperator(oper, ess_tdof_list);
   height = width = mfem_ceed_->Height();

   innercg_.SetOperator(*mfem_ceed_);
   innercg_.SetPrintLevel(-1);
   innercg_.SetMaxIter(max_iter);
   innercg_.SetRelTol(1.e-16);
}

CeedPlainCG::~CeedPlainCG()
{
   delete mfem_ceed_;
}

} // namespace navier
} // namespace mfem
