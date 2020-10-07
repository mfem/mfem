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
#include "../fem/bilinearform.hpp"
#include "../fem/fespace.hpp"
#include "../fem/libceed/ceedsolvers-atpmg.h"
#include "../fem/libceed/ceedsolvers-interpolation.h"

#include "../fem/pfespace.hpp"

namespace mfem
{

/// copy/paste hack used in CeedOperatorFullAssemble
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

/// copy/paste hack used in CeedOperatorFullAssemble
int CeedHackFree(void *p) {
  free(*(void **)p);
  *(void **)p = NULL;
  return 0;
}

/**
   Wrap CeedInterpolation object in an mfem::Operator
*/
class MFEMCeedInterpolation : public mfem::Operator
{
public:
   MFEMCeedInterpolation(Ceed ceed,
                         mfem::FiniteElementSpace& lo_fespace,
                         mfem::FiniteElementSpace& ho_fespace,
                         CeedElemRestriction erestrictu_coarse,
                         CeedElemRestriction erestrictu_fine);

   MFEMCeedInterpolation(
      Ceed ceed, CeedBasis basisctof,
      CeedElemRestriction erestrictu_coarse,
      CeedElemRestriction erestrictu_fine);

   ~MFEMCeedInterpolation();

   virtual void Mult(const mfem::Vector& x, mfem::Vector& y) const;

   virtual void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const;

   using Operator::SetupRAP;

private:
   int Initialize(Ceed ceed, CeedBasis basisctof,
                  CeedElemRestriction erestrictu_coarse,
                  CeedElemRestriction erestrictu_fine);

   CeedBasis basisctof_;
   CeedVector u_, v_;

   CeedInterpolation ceed_interp_;

   bool owns_basis_;
};

// forward declaration
class MFEMCeedVCycle;

#ifdef MFEM_USE_MPI
   using GroupComm = GroupCommunicator;
#else
   using GroupComm = void;
#endif

/**
   This takes a CeedOperator with essential dofs
   and produces a coarser / lower-order operator, an interpolation
   operator between fine/coarse levels, and a list of coarse
   essential dofs.
*/
class CeedMultigridLevel
{
public:
   /// The constructor builds the coarse *operator*, a smoother
   /// for the fine level, and an interpolation between them.
   /// It does *not* build a coarse *solver*.
   /// (smoother construction should also be separate?)
   CeedMultigridLevel(CeedOperator oper,
                      const mfem::Array<int>& ess_dofs,
                      int order_reduction,
                      Mesh &mesh,
                      GroupComm *gc_fine,
                      const Operator *P_fine,
                      const Operator *R_fine);
   ~CeedMultigridLevel();

   /// return coarse operator as CeedOperator (no boundary conditions)
   CeedOperator GetCoarseCeed() { return coarse_oper_; }

   mfem::Array<int>& GetCoarseEssentialDofList() { return lo_ess_tdof_list_; }

   Operator *GetProlongation() const { return P; }

   HypreParMatrix *GetProlongationHypreParMatrix() const { return P_hypre; }

   const Operator *GetFineProlongation() const { return P_fine_; }

   Operator *GetRestriction() const { return R; }

   GroupComm *GetGroupComm() const { return gc; }

   Array<int> &GetDofOffsets() { return dof_offsets; }

   friend class MFEMCeedVCycle;

private:
   CeedElemRestriction ho_er_; // not owned

   CeedOperator oper_; // not owned
   CeedOperator coarse_oper_;
   CeedBasis * coarse_basis_;
   CeedBasis * basisctof_;
   CeedElemRestriction * lo_er_;

   MFEMCeedInterpolation * mfem_interp_;
   Operator *mfem_interp_rap_;

   Array<HYPRE_Int> dof_offsets, tdof_offsets, tdof_nb_offsets;

   const mfem::Array<int>& ho_ess_tdof_list_;
   mfem::Array<int> lo_ess_tdof_list_;
   int numsub_;

   GroupComm *gc;
   Operator *P, *R;
   HypreParMatrix *P_hypre;
   const Operator *P_fine_;
   TransposeOperator *R_fine_tr;
};

/**
   Just wrap a Ceed operator in the mfem::Operator interface

   This has no boundary conditions, I expect "users" (as if I had
   any) to use MFEMCeedOperator (which defaults to this if you don't
   give it essential dofs)
*/
class UnconstrainedMFEMCeedOperator : public mfem::Operator
{
public:
   UnconstrainedMFEMCeedOperator(CeedOperator oper);
   ~UnconstrainedMFEMCeedOperator();

   virtual void Mult(const mfem::Vector& x, mfem::Vector& y) const;
   using Operator::SetupRAP;
private:
   CeedOperator oper_;
   CeedVector u_, v_;   // mutable?
};

class MFEMCeedOperator : public mfem::Operator
{
public:
   MFEMCeedOperator(CeedOperator oper, mfem::Array<int>& ess_tdofs, const Operator *P)
      :
      unconstrained_op_(oper)
   {
      Operator *rap = unconstrained_op_.SetupRAP(P, P);
      height = width = rap->Height();
      bool own_rap = rap != &unconstrained_op_;
      constrained_op_ = new ConstrainedOperator(rap, ess_tdofs, own_rap);
   }

   MFEMCeedOperator(CeedOperator oper, const Operator *P)
      :
      unconstrained_op_(oper)
   {
      mfem::Array<int> empty;
      Operator *rap = unconstrained_op_.SetupRAP(P, P);
      height = width = rap->Height();
      bool own_rap = rap != &unconstrained_op_;
      constrained_op_ = new ConstrainedOperator(rap, empty, own_rap);
   }

   ~MFEMCeedOperator()
   {
      delete constrained_op_;
   }

   void Mult(const mfem::Vector& x, mfem::Vector& y) const
   {
      constrained_op_->Mult(x, y);
   }

private:
   UnconstrainedMFEMCeedOperator unconstrained_op_;
   ConstrainedOperator *constrained_op_;
};

class MFEMCeedVCycle : public mfem::Solver
{
public:
   MFEMCeedVCycle(const CeedMultigridLevel& level,
                  const mfem::Operator& fine_operator,
                  const mfem::Solver& coarse_solver);
   ~MFEMCeedVCycle();

   void Mult(const mfem::Vector& x, mfem::Vector& y) const;
   void SetOperator(const Operator &op) { }

private:
   void FormResidual(const mfem::Vector& b,
                     const mfem::Vector& x,
                     mfem::Vector& r) const;

   const mfem::Operator& fine_operator_;
   const mfem::Solver& coarse_solver_;
   const mfem::Operator* fine_smoother_;
   const mfem::Operator& interp_;

   /// work vectors (too many of them, can be economized)
   mutable mfem::Vector residual_;
   mutable mfem::Vector correction_;
   mutable mfem::Vector coarse_residual_;
   mutable mfem::Vector coarse_correction_;
};

#ifdef MFEM_USE_MPI

class CeedCGWithAMG : public mfem::Solver
{
public:
   CeedCGWithAMG(CeedMultigridLevel &level,
                 int sparse_solver_type,
                 bool use_amgx);

   ~CeedCGWithAMG();

   void SetOperator(const mfem::Operator& op) { }
   void Mult(const mfem::Vector& x, mfem::Vector& y) const
   {
      solver_->Mult(x, y);
   }

private:
   mfem::CGSolver innercg_;

   MFEMCeedOperator * mfem_ceed_;

   mfem::SparseMatrix * mat_assembled_;
   mfem::HypreParMatrix * hypre_assembled_;
   mfem::Solver * inner_prec_;
   mfem::Solver * solver_;
};

#endif

/**
   Do a fixed number of CG iterations on the coarsest level.

   (this object is probably unnecessary, in particular why do
   we need to interface with a CeedOperator instead of an mfem::Operator?)
*/
class CeedPlainCG : public mfem::Solver
{
public:
   CeedPlainCG(CeedOperator oper,
               mfem::Array<int>& ess_tdof_list,
               const Operator *P,
               int max_its=10);

   ~CeedPlainCG();

   void SetOperator(const mfem::Operator& op) { }
   void Mult(const mfem::Vector& x, mfem::Vector& y) const
   {
      innercg_.Mult(x, y);
   }

private:
   mfem::CGSolver innercg_;

   MFEMCeedOperator * mfem_ceed_;
};

int CeedSingleOperatorFullAssemble(CeedOperator op, mfem::SparseMatrix * out)
{
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
   for (CeedInt i=0; i<numinputfields; i++)
   {
      CeedVector vec;
      ierr = CeedOperatorFieldGetVector(input_fields[i], &vec); CeedChk(ierr);
      if (vec == CEED_VECTOR_ACTIVE)
      {
         ierr = CeedOperatorFieldGetBasis(input_fields[i], &basisin);
         CeedChk(ierr);
         ierr = CeedBasisGetNumComponents(basisin, &ncomp); CeedChk(ierr);
         ierr = CeedBasisGetDimension(basisin, &dim); CeedChk(ierr);
         ierr = CeedOperatorFieldGetElemRestriction(input_fields[i], &rstrin);
         CeedChk(ierr);
         CeedEvalMode emode;
         ierr = CeedQFunctionFieldGetEvalMode(qffields[i], &emode);
         CeedChk(ierr);
         switch (emode)
         {
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
   for (CeedInt i=0; i<numoutputfields; i++)
   {
      CeedVector vec;
      ierr = CeedOperatorFieldGetVector(output_fields[i], &vec); CeedChk(ierr);
      if (vec == CEED_VECTOR_ACTIVE)
      {
         ierr = CeedOperatorFieldGetBasis(output_fields[i], &basisout);
         CeedChk(ierr);
         ierr = CeedOperatorFieldGetElemRestriction(output_fields[i], &rstrout);
         CeedChk(ierr);
         CeedChk(ierr);
         CeedEvalMode emode;
         ierr = CeedQFunctionFieldGetEvalMode(qffields[i], &emode);
         CeedChk(ierr);
         switch (emode)
         {
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
   for (CeedInt i = 0; i < nnodes; ++i)
   {
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
   // mfem::SparseMatrix * out = new mfem::SparseMatrix(nnodes, nnodes);
   MFEM_ASSERT(out->Height() == nnodes, "Sizes don't match!");
   MFEM_ASSERT(out->Width() == nnodes, "Sizes don't match!");
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
   for (int e = 0; e < nelem; ++e)
   {
      /// get mfem::Array<int> for use in SparseMatrix::AddSubMatrix()
      mfem::Array<int> rows(elemsize);
      for (int i = 0; i < elemsize; ++i)
      {
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
      for (int q = 0; q < nqpts; ++q)
      {
         for (int n = 0; n < elemsize; ++n)
         {
            CeedInt din = -1;
            for (int ein = 0; ein < numemodein; ++ein)
            {
               if (emodein[ein] == CEED_EVAL_GRAD)
               {
                  din += 1;
               }
               if (emodein[ein] == CEED_EVAL_INTERP)
               {
                  Bmat(numemodein * q + ein, n) += interpin[q * elemsize + n];
               }
               else if (emodein[ein] == CEED_EVAL_GRAD)
               {
                  Bmat(numemodein * q + ein, n) += gradin[(din*nqpts+q) * elemsize + n];
               }
               else
               {
                  MFEM_ASSERT(false, "Not implemented!");
               }
            }
         }
         for (int ei = 0; ei < numemodein; ++ei)
         {
            for (int ej = 0; ej < numemodein; ++ej)
            {
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

   return 0;
}

/**
   todo: think of ways to make this faster when we know a sparsity structure (?)
   (ie, for low-order refined or algebraic sparsification?)
*/
int CeedOperatorFullAssemble(CeedOperator op,
                             mfem::SparseMatrix ** mat)
{
   int ierr;

   CeedElemRestriction er;
   ierr = CeedOperatorGetActiveElemRestriction(op, &er); CeedChk(ierr);
   CeedInt nnodes;
   ierr = CeedElemRestrictionGetLVectorSize(er, &nnodes); CeedChk(ierr);

   mfem::SparseMatrix * out = new mfem::SparseMatrix(nnodes, nnodes);

   bool isComposite;
   ierr = CeedOperatorIsComposite(op, &isComposite); CeedChk(ierr);
   if (isComposite)
   {
      CeedInt numsub;
      CeedOperator *subops;
      CeedOperatorGetNumSub(op, &numsub);
      ierr = CeedOperatorGetSubList(op, &subops); CeedChk(ierr);
      for (int i = 0; i < numsub; ++i)
      {
         ierr = CeedSingleOperatorFullAssemble(subops[i], out); CeedChk(ierr);
      }
   }
   else
   {
      ierr = CeedSingleOperatorFullAssemble(op, out); CeedChk(ierr);
   }
   const int skip_zeros = 0; // enforce structurally symmetric for later elimination
   out->Finalize(skip_zeros);
   *mat = out;

   return 0;
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

   const CeedScalar *x_ptr;
   CeedScalar *y_ptr;
   CeedMemType mem;
   CeedGetPreferredMemType(internal::ceed, &mem);
   if ( Device::Allows(Backend::CUDA) && mem==CEED_MEM_DEVICE )
   {
      x_ptr = x.Read();
      y_ptr = y.ReadWrite();
   }
   else
   {
      x_ptr = x.HostRead();
      y_ptr = y.HostReadWrite();
      mem = CEED_MEM_HOST;
   }

   ierr += CeedVectorSetArray(u_, mem, CEED_USE_POINTER, const_cast<CeedScalar*>(x_ptr));
   ierr += CeedVectorSetArray(v_, mem, CEED_USE_POINTER, y_ptr);

   ierr += CeedOperatorApply(oper_, u_, v_, CEED_REQUEST_IMMEDIATE);

   ierr += CeedVectorTakeArray(u_, mem, const_cast<CeedScalar**>(&x_ptr));
   ierr += CeedVectorTakeArray(v_, mem, &y_ptr);

   MFEM_ASSERT(ierr == 0, "CEED error");
}

struct IdentitySolver : Solver
{
   IdentitySolver(int n) : Solver(n) { }
   void Mult(const Vector &b, Vector &x) const { x = b; }
   void SetOperator(const Operator &op) { }
};

mfem::Solver * BuildSmootherFromCeed(Operator * mfem_op, CeedOperator ceed_op,
                                     const Array<int>& ess_tdofs,
                                     const Operator *P,
                                     bool chebyshev)
{
   // this is a local diagonal, in the sense of l-vector
   CeedVector diagceed;
   CeedInt length;
   Ceed ceed;
   CeedOperatorGetCeed(ceed_op, &ceed);
   CeedOperatorGetSize(ceed_op, &length);
   CeedVectorCreate(ceed, length, &diagceed);
   CeedVectorSetValue(diagceed, 0.0);
   CeedOperatorLinearAssembleDiagonal(ceed_op, diagceed, CEED_REQUEST_IMMEDIATE);
   const CeedScalar * diagvals;
   CeedMemType mem;
   CeedGetPreferredMemType(ceed, &mem);
   if ( Device::Allows(Backend::CUDA) && mem==CEED_MEM_DEVICE )
   {
      // intentional no-op
   }
   else
   {
      mem = CEED_MEM_HOST;
   }
   CeedVectorGetArrayRead(diagceed, mem, &diagvals);
   mfem::Vector local_diag(const_cast<CeedScalar*>(diagvals), length);
   mfem::Vector mfem_diag;
   if (P)
   {
      mfem_diag.SetSize(P->Width());
      P->MultTranspose(local_diag, mfem_diag);
   }
   else
   {
      mfem_diag.SetDataAndSize(local_diag, local_diag.Size());
   }
   mfem::Solver * out = NULL;
   if (chebyshev)
   {
      const int cheb_order = 3;
      out = new OperatorChebyshevSmoother(mfem_op, mfem_diag, ess_tdofs, cheb_order);
   }
   else
   {
      const double jacobi_scale = 0.65;
      out = new OperatorJacobiSmoother(mfem_diag, ess_tdofs, jacobi_scale);
   }
   CeedVectorRestoreArrayRead(diagceed, &diagvals);
   CeedVectorDestroy(&diagceed);
   return out;
}

MFEMCeedVCycle::MFEMCeedVCycle(
   const CeedMultigridLevel& level,
   const mfem::Operator& fine_operator,
   const mfem::Solver& coarse_solver)
   :
   mfem::Solver(fine_operator.Height()),
   fine_operator_(fine_operator),
   coarse_solver_(coarse_solver),
   interp_(*level.mfem_interp_rap_)
{
   MFEM_VERIFY(fine_operator_.Height() == interp_.Height(), "Sizes don't match!");
   MFEM_VERIFY(coarse_solver_.Height() == interp_.Width(), "Sizes don't match!");

   residual_.SetSize(fine_operator_.Height());
   correction_.SetSize(fine_operator_.Height());
   coarse_residual_.SetSize(coarse_solver_.Height());
   coarse_correction_.SetSize(coarse_solver_.Height());

   fine_smoother_ = BuildSmootherFromCeed(const_cast<Operator*>(&fine_operator),
                                          level.oper_,
                                          level.ho_ess_tdof_list_,
                                          level.GetFineProlongation(),
                                          true);
}

MFEMCeedVCycle::~MFEMCeedVCycle()
{
   delete fine_smoother_;
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
   fine_smoother_->Mult(b, correction_);
   x += correction_;

   FormResidual(b, x, residual_);
   interp_.MultTranspose(residual_, coarse_residual_);
   coarse_correction_ = 0.0;
   coarse_solver_.Mult(coarse_residual_, coarse_correction_);
   interp_.Mult(coarse_correction_, correction_);
   x += correction_;
   FormResidual(b, x, residual_);
   fine_smoother_->Mult(residual_, correction_);
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

   const CeedScalar *x_ptr;
   CeedScalar *y_ptr;
   CeedMemType mem;
   CeedGetPreferredMemType(internal::ceed, &mem);
   if ( Device::Allows(Backend::CUDA) && mem==CEED_MEM_DEVICE )
   {
      x_ptr = x.Read();
      y_ptr = y.ReadWrite();
   }
   else
   {
      x_ptr = x.HostRead();
      y_ptr = y.HostReadWrite();
      mem = CEED_MEM_HOST;
   }

   ierr += CeedVectorSetArray(u_, mem, CEED_USE_POINTER, const_cast<CeedScalar*>(x_ptr));
   ierr += CeedVectorSetArray(v_, mem, CEED_USE_POINTER, y_ptr);

   ierr += CeedInterpolationInterpolate(ceed_interp_, u_, v_);

   ierr += CeedVectorTakeArray(u_, mem, const_cast<CeedScalar**>(&x_ptr));
   ierr += CeedVectorTakeArray(v_, mem, &y_ptr);

   MFEM_ASSERT(ierr == 0, "CEED error");
}

void MFEMCeedInterpolation::MultTranspose(const mfem::Vector& x,
                                          mfem::Vector& y) const
{
   int ierr = 0;

   const CeedScalar *x_ptr;
   CeedScalar *y_ptr;
   CeedMemType mem;
   CeedGetPreferredMemType(internal::ceed, &mem);
   if ( Device::Allows(Backend::CUDA) && mem==CEED_MEM_DEVICE )
   {
      x_ptr = x.Read();
      y_ptr = y.ReadWrite();
   }
   else
   {
      x_ptr = x.HostRead();
      y_ptr = y.HostReadWrite();
      mem = CEED_MEM_HOST;
   }

   ierr += CeedVectorSetArray(v_, mem, CEED_USE_POINTER, const_cast<CeedScalar*>(x_ptr));
   ierr += CeedVectorSetArray(u_, mem, CEED_USE_POINTER, y_ptr);

   ierr += CeedInterpolationRestrict(ceed_interp_, v_, u_);

   ierr += CeedVectorTakeArray(v_, mem, const_cast<CeedScalar**>(&x_ptr));
   ierr += CeedVectorTakeArray(u_, mem, &y_ptr);

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
   auto lobo = lo_boundary_ones.HostRead();
   for (int i = 0; i < lo_boundary_ones.Size(); ++i)
   {
      if (lobo[i] > 0.9)
      {
         alg_lo_ess_tdof_list.Append(i);
      }
   }
}

CeedMultigridLevel::CeedMultigridLevel(CeedOperator oper,
                                       const mfem::Array<int>& ho_ess_tdof_list,
                                       int order_reduction,
                                       Mesh &mesh,
                                       GroupComm *gc_fine,
                                       const Operator *P_fine,
                                       const Operator *R_fine)
   :
   oper_(oper),
   ho_ess_tdof_list_(ho_ess_tdof_list),
   gc(NULL),
   P(NULL),
   R(NULL),
   P_fine_(P_fine)
{
   Ceed ceed;
   CeedOperatorGetCeed(oper, &ceed);

   bool isComposite;
   CeedOperatorIsComposite(oper, &isComposite);
   if (isComposite)
   {
      CeedOperatorGetNumSub(oper, &numsub_);
   }
   else
   {
      numsub_ = 1;
   }
   coarse_basis_ = new CeedBasis[numsub_];
   basisctof_ = new CeedBasis[numsub_];
   lo_er_ = new CeedElemRestriction[numsub_];

   CeedInt *dof_map;
   if (isComposite)
   {
      CeedOperator *subops;
      CeedOperatorGetSubList(oper, &subops);
      CeedCompositeOperatorCreate(ceed, &coarse_oper_);
      for (int i = 0; i < numsub_; ++i)
      {
         CeedOperator subcoarse;
         CeedInt *dof_map_tmp;
         CeedATPMGBundle(subops[i], order_reduction, &coarse_basis_[i], &basisctof_[i],
                         &lo_er_[i], &subcoarse, dof_map_tmp);
         if (i == 0) { dof_map = dof_map_tmp; }
         else { free(dof_map_tmp); }
         CeedCompositeOperatorAddSub(coarse_oper_, subcoarse);
         CeedOperatorDestroy(&subcoarse); // give ownership to composite operator
      }
      CeedOperatorGetActiveElemRestriction(subops[0], &ho_er_);
   }
   else
   {
      CeedATPMGBundle(oper, order_reduction, &coarse_basis_[0], &basisctof_[0],
                      &lo_er_[0], &coarse_oper_, dof_map);
      CeedOperatorGetActiveElemRestriction(oper, &ho_er_);
   }

#ifdef MFEM_USE_MPI
   if (gc_fine)
   {
      int lsize;
      CeedElemRestrictionGetLVectorSize(lo_er_[0], &lsize);
      const Table &group_ldof_fine = gc_fine->GroupLDofTable();

      Array<int> ldof_group(lsize);
      ldof_group = 0;

      GroupTopology &group_topo = gc_fine->GetGroupTopology();
      gc = new GroupCommunicator(group_topo);
      Table &group_ldof = gc->GroupLDofTable();
      group_ldof.MakeI(group_ldof_fine.Size());
      for (int g=1; g<group_ldof_fine.Size(); ++g)
      {
         int nldof_fine_g = group_ldof_fine.RowSize(g);
         const int *ldof_fine_g = group_ldof_fine.GetRow(g);
         for (int i=0; i<nldof_fine_g; ++i)
         {
            int icoarse = dof_map[ldof_fine_g[i]];
            if (icoarse >= 0)
            {
               group_ldof.AddAColumnInRow(g);
               ldof_group[icoarse] = g;
            }
         }
      }
      group_ldof.MakeJ();
      for (int g=1; g<group_ldof_fine.Size(); ++g)
      {
         int nldof_fine_g = group_ldof_fine.RowSize(g);
         const int *ldof_fine_g = group_ldof_fine.GetRow(g);
         for (int i=0; i<nldof_fine_g; ++i)
         {
            int icoarse = dof_map[ldof_fine_g[i]];
            if (icoarse >= 0)
            {
               group_ldof.AddConnection(g, icoarse);
            }
         }
      }
      group_ldof.ShiftUpI();
      gc->Finalize();
      Array<int> ldof_ltdof(lsize);
      ldof_ltdof = -2;
      int ltsize = 0;
      for (int i=0; i<lsize; ++i)
      {
         int g = ldof_group[i];
         if (group_topo.IAmMaster(g))
         {
            ldof_ltdof[i] = ltsize;
            ++ltsize;
         }
      }
      gc->SetLTDofTable(ldof_ltdof);
      gc->Bcast(ldof_ltdof);

      SparseMatrix *R_mat = new SparseMatrix(ltsize, lsize);
      for (int j=0; j<lsize; ++j)
      {
         if (group_topo.IAmMaster(ldof_group[j]))
         {
            int i = ldof_ltdof[j];
            R_mat->Set(i,j,1.0);
         }
      }
      R_mat->Finalize();
      R = R_mat;

      P = new ConformingProlongationOperator(lsize, *gc);

      // Want also to represent P as a HypreParMatrix
      // (only need this at the coarsest level, should have an option
      // to turn this off)

      // This is a lot of duplicated code from ParFiniteElementSpace.
      // We can either accept the code duplication, or perhaps create a derived
      // class that privately inherits from ParFiniteElementSpace that
      // represents the coarse space.

      // In the mean time, the functionality of
      // Build_Dof_TrueDof_Matrix, GetLocalTDofNumber, GetGlobalTDofNumber, etc.
      // is reproduced below.
      ParMesh *pmesh = dynamic_cast<ParMesh*>(&mesh);
      MFEM_VERIFY(pmesh != NULL, "");
      Array<HYPRE_Int> *offsets[2] = {&dof_offsets, &tdof_offsets};
      HYPRE_Int loc_sizes[2] = {lsize, ltsize};
      pmesh->GenerateOffsets(2, loc_sizes, offsets);

      MPI_Comm comm = pmesh->GetComm();

      if (HYPRE_AssumedPartitionCheck())
      {
         // communicate the neighbor offsets in tdof_nb_offsets
         int nsize = group_topo.GetNumNeighbors()-1;
         MPI_Request *requests = new MPI_Request[2*nsize];
         MPI_Status  *statuses = new MPI_Status[2*nsize];
         tdof_nb_offsets.SetSize(nsize+1);
         tdof_nb_offsets[0] = tdof_offsets[0];

         // send and receive neighbors' local tdof offsets
         int request_counter = 0;
         for (int i = 1; i <= nsize; i++)
         {
            MPI_Irecv(&tdof_nb_offsets[i], 1, HYPRE_MPI_INT,
                     group_topo.GetNeighborRank(i), 5365, comm,
                     &requests[request_counter++]);
         }
         for (int i = 1; i <= nsize; i++)
         {
            MPI_Isend(&tdof_nb_offsets[0], 1, HYPRE_MPI_INT,
                     group_topo.GetNeighborRank(i), 5365, comm,
                     &requests[request_counter++]);
         }
         MPI_Waitall(request_counter, requests, statuses);

         delete [] statuses;
         delete [] requests;
      }

      HYPRE_Int *i_diag = Memory<HYPRE_Int>(lsize+1);
      HYPRE_Int *j_diag = Memory<HYPRE_Int>(ltsize);
      int diag_counter;

      HYPRE_Int *i_offd = Memory<HYPRE_Int>(lsize+1);
      HYPRE_Int *j_offd = Memory<HYPRE_Int>(lsize-ltsize);
      int offd_counter;

      HYPRE_Int *cmap   = Memory<HYPRE_Int>(lsize-ltsize);

      HYPRE_Int *col_starts = tdof_offsets;
      HYPRE_Int *row_starts = dof_offsets;

      Array<Pair<HYPRE_Int, int> > cmap_j_offd(lsize-ltsize);

      i_diag[0] = i_offd[0] = 0;
      diag_counter = offd_counter = 0;
      for (int i_ldof = 0; i_ldof < lsize; i_ldof++)
      {
         int g = ldof_group[i_ldof];
         int i_ltdof = ldof_ltdof[i_ldof];
         if (group_topo.IAmMaster(g))
         {
            j_diag[diag_counter++] = i_ltdof;
         }
         else
         {
            HYPRE_Int global_tdof_number;
            int g = ldof_group[i_ldof];
            if (HYPRE_AssumedPartitionCheck())
            {
               global_tdof_number
                  = i_ltdof + tdof_nb_offsets[group_topo.GetGroupMaster(g)];
            }
            else
            {
               global_tdof_number
                  = i_ltdof + tdof_offsets[group_topo.GetGroupMasterRank(g)];
            }

            cmap_j_offd[offd_counter].one = global_tdof_number;
            cmap_j_offd[offd_counter].two = offd_counter;
            offd_counter++;
         }
         i_diag[i_ldof+1] = diag_counter;
         i_offd[i_ldof+1] = offd_counter;
      }

      SortPairs<HYPRE_Int, int>(cmap_j_offd, offd_counter);

      for (int i = 0; i < offd_counter; i++)
      {
         cmap[i] = cmap_j_offd[i].one;
         j_offd[cmap_j_offd[i].two] = i;
      }

      P_hypre = new HypreParMatrix(
         comm, pmesh->GetMyRank(), pmesh->GetNRanks(),
         row_starts, col_starts,
         i_diag, j_diag, i_offd, j_offd,
         cmap, offd_counter
      );
   }
#endif

   if (R_fine)
   {
      const SparseMatrix *R_fine_mat = dynamic_cast<const SparseMatrix *>(R_fine);
      MFEM_ASSERT(R_fine_mat, "");
      R_fine_mat->BuildTranspose();
      R_fine_tr = new TransposeOperator(R_fine);
   }
   else
   {
      R_fine_tr = NULL;
   }

   mfem_interp_ = new MFEMCeedInterpolation(ceed, basisctof_[0], lo_er_[0], ho_er_);
   // Why does SetupRAP reverse the argument order???
   mfem_interp_rap_ = mfem_interp_->SetupRAP(P, R_fine_tr);
   CoarsenEssentialDofs(*mfem_interp_rap_, ho_ess_tdof_list, lo_ess_tdof_list_);

   free(dof_map);
}

CeedMultigridLevel::~CeedMultigridLevel()
{
   CeedOperatorDestroy(&coarse_oper_);

   for (int i = 0; i < numsub_; ++i)
   {
      CeedBasisDestroy(&coarse_basis_[i]);
      CeedBasisDestroy(&basisctof_[i]);
      CeedElemRestrictionDestroy(&lo_er_[i]);
   }
   delete [] coarse_basis_;
   delete [] basisctof_;
   delete [] lo_er_;

   delete mfem_interp_;

   delete P;
   delete R;
   delete R_fine_tr;
#ifdef MFEM_USE_MPI
   delete gc;
#endif
}

#ifdef MFEM_USE_MPI

CeedCGWithAMG::CeedCGWithAMG(CeedMultigridLevel &level,
                             int sparse_solver_type,
                             bool use_amgx)
{
   CeedOperator oper = level.GetCoarseCeed();
   mfem::Array<int>& ess_tdof_list = level.GetCoarseEssentialDofList();
   const Operator *P = level.GetProlongation();
   const HypreParMatrix *P_hypre = level.GetProlongationHypreParMatrix();

   mfem_ceed_ = new MFEMCeedOperator(oper, ess_tdof_list, P);
   height = width = mfem_ceed_->Height();

   CeedOperatorFullAssemble(oper, &mat_assembled_);

   innercg_.SetOperator(*mfem_ceed_);

// Disable AMGX for now...
#if 0
#ifdef CEED_USE_AMGX
   if (use_amgx)
   {
      NvidiaAMGX * amgx = new NvidiaAMGX();
      const bool amgx_verbose = false;
      amgx->ConfigureAsPreconditioner(amgx_verbose);
      amgx->SetOperator(*mat_assembled_);
      hypre_assembled_ = NULL;
      inner_prec_ = amgx;
   }
#endif
#endif

   HypreParMatrix *hypre_local = new HypreParMatrix(
      P_hypre->GetComm(), P_hypre->GetGlobalNumRows(), level.GetDofOffsets(),
      mat_assembled_
   );

   hypre_assembled_ = RAP(hypre_local, P_hypre);
   HypreParMatrix *hypre_e = hypre_assembled_->EliminateRowsCols(ess_tdof_list);
   delete hypre_e;

   HypreBoomerAMG * amg = new HypreBoomerAMG(*hypre_assembled_);
   amg->SetPrintLevel(0);
   inner_prec_ = amg;

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

#endif // MFEM_USE_MPI, so it builds without hypre

CeedPlainCG::CeedPlainCG(CeedOperator oper,
                         mfem::Array<int>& ess_tdof_list,
                         const Operator *P,
                         int max_iter)
{
   mfem_ceed_ = new MFEMCeedOperator(oper, ess_tdof_list, P);
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
                                         BilinearForm& form, Array<int>& ess_dofs,
                                         bool use_amg)
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
   int num_integrators = bffis->Size();
   CeedCompositeOperatorCreate(internal::ceed, &fine_composite_op);
   for (int i = 0; i < num_integrators; ++i)
   {
      bool casted = false;
      DiffusionIntegrator * dintegrator = dynamic_cast<DiffusionIntegrator*>((*bffis)[i]);
      if (dintegrator)
      {
         CeedCompositeOperatorAddSub(fine_composite_op, dintegrator->GetCeedData()->oper);
         casted = true;
      }
      MassIntegrator * mintegrator = dynamic_cast<MassIntegrator*>((*bffis)[i]);
      if (mintegrator)
      {
         MFEM_ASSERT(!casted, "Integrator already used (programmer error)!");
         CeedCompositeOperatorAddSub(fine_composite_op, mintegrator->GetCeedData()->oper);
         casted = true;
      }
      VectorDiffusionIntegrator * vdintegrator = dynamic_cast<VectorDiffusionIntegrator*>((*bffis)[i]);
      if (vdintegrator)
      {
         MFEM_ASSERT(!casted, "Integrator already used (programmer error)!");
         CeedCompositeOperatorAddSub(fine_composite_op, vdintegrator->GetCeedData()->oper);
         casted = true;
      }
      VectorMassIntegrator * vmintegrator = dynamic_cast<VectorMassIntegrator*>((*bffis)[i]);
      if (vmintegrator)
      {
         MFEM_ASSERT(!casted, "Integrator already used (programmer error)!");
         CeedCompositeOperatorAddSub(fine_composite_op, vmintegrator->GetCeedData()->oper);
         casted = true;
      }
      MFEM_VERIFY(casted, "Integrator not supported in AlgebraicCeedSolver!");
   }
   CeedOperator current_op = fine_composite_op;

   FiniteElementSpace &fes = *form.FESpace();
   Mesh &mesh = *fes.GetMesh();
   GroupComm *gc = NULL;
#ifdef MFEM_USE_MPI
   ParFiniteElementSpace *pfes = dynamic_cast<ParFiniteElementSpace*>(&fes);
   if (pfes)
   {
      gc = &pfes->GroupComm();
   }
#endif
   const Operator *R = fes.GetRestrictionMatrix();
   const Operator *P = fes.GetProlongationMatrix();
   operators = new Operator*[num_levels];
   operators[0] = &fine_mfem_op;
   levels = new CeedMultigridLevel*[num_levels - 1];
   mfem::Array<int> * current_ess_dofs = &ess_dofs;
   current_order = order;
   for (int i = 0; i < num_levels - 1; ++i)
   {
      const int order_reduction = current_order - (current_order / 2);
      current_order = current_order / 2;
      levels[i] = new CeedMultigridLevel(current_op, *current_ess_dofs, order_reduction, mesh, gc, P, R);
      current_op = levels[i]->GetCoarseCeed();
      current_ess_dofs = &levels[i]->GetCoarseEssentialDofList();
      P = levels[i]->GetProlongation();
      R = levels[i]->GetRestriction();
      gc = levels[i]->GetGroupComm();
      operators[i + 1] = new MFEMCeedOperator(current_op, *current_ess_dofs, levels[i]->GetProlongation());
   }
   mfem::Solver * coarsest_solver;
   CeedMultigridLevel * coarsest = NULL;
   if (num_levels > 1)
   {
      coarsest = levels[num_levels - 2];
   }

   if (num_levels > 1)
   {
      if (Device::Allows(Backend::CUDA) || !use_amg)
      {
         coarsest_solver = BuildSmootherFromCeed(NULL, coarsest->GetCoarseCeed(),
                                                 coarsest->GetCoarseEssentialDofList(),
                                                 coarsest->GetProlongation(),
                                                 false);
      }
      else
      {
#ifdef MFEM_USE_MPI
         bool use_amgx = false;
         const int sparse_solver_type = 1; // single v-cycle
         coarsest_solver = new CeedCGWithAMG(*coarsest,
                                             sparse_solver_type,
                                             use_amgx);
#else
         int coarse_cg_iterations = 10;
         coarsest_solver = new CeedPlainCG(coarsest->GetCoarseCeed(),
                                           coarsest->GetCoarseEssentialDofList(),
                                           coarse_cg_iterations);
#endif
      }
   }
   else
   {
      // TODO... interface for AMG doesn't work well in this case
      coarsest_solver = BuildSmootherFromCeed(&fine_mfem_op, fine_composite_op, ess_dofs, P, false);
   }

   // loop up from coarsest to build V-cycle solvers
   solvers = new Solver*[num_levels];
   solvers[num_levels - 1] = coarsest_solver;
   for (int i = 0; i < num_levels - 1; ++i)
   {
      int index = num_levels - 2 - i;
      solvers[index] = new MFEMCeedVCycle(*levels[index], *operators[index],
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

   CeedOperatorDestroy(&fine_composite_op);
}

void AlgebraicCeedSolver::Mult(const Vector& x, Vector& y) const
{
   solvers[0]->Mult(x, y);
}

} // namespace mfem

#endif // MFEM_USE_CEED
