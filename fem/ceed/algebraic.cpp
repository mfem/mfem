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

#include "algebraic.hpp"

#include "../bilinearform.hpp"
#include "../fespace.hpp"
#include "solvers-atpmg.hpp"
#include "solvers-qcoarsen.hpp"
#include "../pfespace.hpp"
#include "../../general/forall.hpp"

namespace mfem
{

namespace ceed
{

#ifdef MFEM_USE_CEED

int CeedOperatorFullAssemble(CeedOperator op, SparseMatrix **mat)
{
   int ierr;

   CeedInt nentries;
   CeedInt *rows, *cols;
   CeedVector values;
   const CeedScalar *vals;

   ierr = CeedOperatorLinearAssembleSymbolic(op, &nentries, &rows, &cols);
   CeedChk(ierr);
   ierr = CeedVectorCreate(internal::ceed, nentries, &values); CeedChk(ierr);
   ierr = CeedOperatorLinearAssemble(op, values); CeedChk(ierr);

   ierr = CeedVectorGetArrayRead(values, CEED_MEM_HOST, &vals); CeedChk(ierr);
   CeedElemRestriction er;
   ierr = CeedOperatorGetActiveElemRestriction(op, &er); CeedChk(ierr);
   CeedInt nnodes;
   ierr = CeedElemRestrictionGetLVectorSize(er, &nnodes); CeedChk(ierr);
   SparseMatrix *out = new SparseMatrix(nnodes, nnodes);
   for (int k = 0; k < nentries; ++k)
   {
      out->Add(rows[k], cols[k], vals[k]);
   }
   ierr = CeedVectorRestoreArrayRead(values, &vals); CeedChk(ierr);
   const int skip_zeros = 0;
   out->Finalize(skip_zeros);
   ierr = CeedVectorDestroy(&values); CeedChk(ierr);
   *mat = out;

   return 0;
}

/** Wraps a CeedOperator in an mfem::Operator, with essential boundary
    conditions and a prolongation operator for parallel application. */
class ConstrainedOperator : public mfem::Operator
{
public:
   /// This object takes ownership of oper and will delete it
   ConstrainedOperator(CeedOperator oper, const Array<int> &ess_tdofs_,
                       const mfem::Operator *P_);
   ConstrainedOperator(CeedOperator oper, const mfem::Operator *P_);
   ~ConstrainedOperator();
   void Mult(const Vector& x, Vector& y) const;
   CeedOperator GetCeedOperator() const;
   const Array<int> &GetEssentialTrueDofs() const;
   const mfem::Operator *GetProlongation() const;
private:
   Array<int> ess_tdofs;
   const mfem::Operator *P;
   ceed::Operator *unconstrained_op;
   mfem::ConstrainedOperator *constrained_op;
};

ConstrainedOperator::ConstrainedOperator(
   CeedOperator oper,
   const Array<int> &ess_tdofs_,
   const mfem::Operator *P_)
   : ess_tdofs(ess_tdofs_), P(P_)
{
   unconstrained_op = new ceed::Operator(oper);
   mfem::Operator *rap = unconstrained_op->SetupRAP(P, P);
   height = width = rap->Height();
   bool own_rap = (rap != unconstrained_op);
   constrained_op = new mfem::ConstrainedOperator(rap, ess_tdofs, own_rap);
}

ConstrainedOperator::ConstrainedOperator(CeedOperator oper,
                                         const mfem::Operator *P_)
   : ConstrainedOperator(oper, Array<int>(), P_)
{ }

ConstrainedOperator::~ConstrainedOperator()
{
   delete constrained_op;
   delete unconstrained_op;
}

void ConstrainedOperator::Mult(const Vector& x, Vector& y) const
{
   constrained_op->Mult(x, y);
}

CeedOperator ConstrainedOperator::GetCeedOperator() const
{
   return unconstrained_op->GetCeedOperator();
}

const Array<int> &ConstrainedOperator::GetEssentialTrueDofs() const
{
   return ess_tdofs;
}

const mfem::Operator *ConstrainedOperator::GetProlongation() const
{
   return P;
}

/// assumes a square operator (you could do rectangular, you'd have
/// to find separate active input and output fields/restrictions)
int CeedOperatorGetSize(CeedOperator oper, CeedInt * size)
{
   int ierr;
   CeedElemRestriction er;
   ierr = CeedOperatorGetActiveElemRestriction(oper, &er); CeedChk(ierr);
   ierr = CeedElemRestrictionGetLVectorSize(er, size); CeedChk(ierr);
   return 0;
}

Solver *BuildSmootherFromCeed(ConstrainedOperator &op, bool chebyshev)
{
   int ierr;
   CeedOperator ceed_op = op.GetCeedOperator();
   const Array<int> &ess_tdofs = op.GetEssentialTrueDofs();
   const mfem::Operator *P = op.GetProlongation();
   // Assemble the a local diagonal, in the sense of L-vector
   CeedVector diagceed;
   CeedInt length;
   ierr = CeedOperatorGetSize(ceed_op, &length); PCeedChk(ierr);
   ierr = CeedVectorCreate(internal::ceed, length, &diagceed); PCeedChk(ierr);
   CeedMemType mem;
   ierr = CeedGetPreferredMemType(internal::ceed, &mem); PCeedChk(ierr);
   if (!Device::Allows(Backend::CUDA) || mem != CEED_MEM_DEVICE)
   {
      mem = CEED_MEM_HOST;
   }
   Vector local_diag(length);
   CeedScalar *ptr = (mem == CEED_MEM_HOST) ? local_diag.HostWrite() :
                     local_diag.Write(true);
   ierr = CeedVectorSetArray(diagceed, mem, CEED_USE_POINTER, ptr);
   PCeedChk(ierr);
   ierr = CeedOperatorLinearAssembleDiagonal(ceed_op, diagceed,
                                             CEED_REQUEST_IMMEDIATE);
   PCeedChk(ierr);
   ierr = CeedVectorTakeArray(diagceed, mem, NULL); PCeedChk(ierr);

   Vector t_diag;
   if (P)
   {
      t_diag.SetSize(P->Width());
      P->MultTranspose(local_diag, t_diag);
   }
   else
   {
      t_diag.NewMemoryAndSize(local_diag.GetMemory(), length, false);
   }
   Solver *out = NULL;
   if (chebyshev)
   {
      const int cheb_order = 3;
      out = new OperatorChebyshevSmoother(&op, t_diag, ess_tdofs, cheb_order);
   }
   else
   {
      const double jacobi_scale = 0.65;
      out = new OperatorJacobiSmoother(t_diag, ess_tdofs, jacobi_scale);
   }
   ierr = CeedVectorDestroy(&diagceed); PCeedChk(ierr);
   return out;
}

#ifdef MFEM_USE_MPI

/// Builds and applies assembled AMG to a CeedOperator
class AssembledAMG : public Solver
{
public:
   AssembledAMG(ConstrainedOperator &oper, HypreParMatrix *P, bool amgx=false)
   {
      MFEM_ASSERT(P != NULL, "Provided HypreParMatrix is invalid!");
      height = width = oper.Height();

      int ierr;
      const Array<int> ess_tdofs = oper.GetEssentialTrueDofs();

      ierr = CeedOperatorFullAssemble(oper.GetCeedOperator(), &mat_local);
      PCeedChk(ierr);

      {
         HypreParMatrix hypre_local(
            P->GetComm(), P->GetGlobalNumRows(), P->RowPart(), mat_local);
         op_assembled = RAP(&hypre_local, P);
      }
      HypreParMatrix *mat_e = op_assembled->EliminateRowsCols(ess_tdofs);
      delete mat_e;

#ifdef MFEM_USE_AMGX
      if (amgx)
      {
         bool amgx_verbose = false;
         amg = new AmgXSolver(op_assembled->GetComm(),
                              AmgXSolver::PRECONDITIONER, amgx_verbose);
         amg->SetOperator(*op_assembled);
      }
      else
#endif
      {
         HypreBoomerAMG * hypre_amg = new HypreBoomerAMG(*op_assembled);
         hypre_amg->SetPrintLevel(0);
         amg = hypre_amg;
      }

   }

   void SetOperator(const mfem::Operator &op) override
   { amg->SetOperator(op); }

   void Mult(const Vector &x, Vector &y) const override { amg->Mult(x, y); }
   ~AssembledAMG()
   {
      delete op_assembled;
      delete amg;
      delete mat_local;
   }
private:
   SparseMatrix *mat_local;
   HypreParMatrix *op_assembled;
   Solver *amg;
};

#endif // MFEM_USE_MPI

void CoarsenEssentialDofs(const mfem::Operator &interp,
                          const Array<int> &ho_ess_tdofs,
                          Array<int> &alg_lo_ess_tdofs)
{
   Vector ho_boundary_ones(interp.Height());
   ho_boundary_ones = 0.0;
   const int *ho_ess_tdofs_h = ho_ess_tdofs.HostRead();
   for (int i=0; i<ho_ess_tdofs.Size(); ++i)
   {
      ho_boundary_ones[ho_ess_tdofs_h[i]] = 1.0;
   }
   Vector lo_boundary_ones(interp.Width());
   interp.MultTranspose(ho_boundary_ones, lo_boundary_ones);
   auto lobo = lo_boundary_ones.HostRead();
   for (int i = 0; i < lo_boundary_ones.Size(); ++i)
   {
      if (lobo[i] > 0.9)
      {
         alg_lo_ess_tdofs.Append(i);
      }
   }
}

void AddToCompositeOperator(BilinearFormIntegrator *integ, CeedOperator op)
{
   if (integ->SupportsCeed())
   {
      CeedCompositeOperatorAddSub(op, integ->GetCeedOp().GetCeedOperator());
   }
   else
   {
      MFEM_ABORT("This integrator does not support Ceed!");
   }
}

CeedOperator CreateCeedCompositeOperatorFromBilinearForm(BilinearForm &form)
{
   int ierr;
   CeedOperator op;
   ierr = CeedCompositeOperatorCreate(internal::ceed, &op); PCeedChk(ierr);

   MFEM_VERIFY(form.GetBBFI()->Size() == 0,
               "Not implemented for this integrator!");
   MFEM_VERIFY(form.GetFBFI()->Size() == 0,
               "Not implemented for this integrator!");
   MFEM_VERIFY(form.GetBFBFI()->Size() == 0,
               "Not implemented for this integrator!");

   // Get the domain bilinear form integrators (DBFIs)
   Array<BilinearFormIntegrator*> *bffis = form.GetDBFI();
   for (int i = 0; i < bffis->Size(); ++i)
   {
      AddToCompositeOperator((*bffis)[i], op);
   }
   return op;
}

CeedOperator CoarsenCeedCompositeOperator(
   CeedOperator op,
   CeedElemRestriction er,
   CeedBasis c2f,
   int order_reduction,
   int qorder_reduction
)
{
   int ierr;
   bool isComposite;
   ierr = CeedOperatorIsComposite(op, &isComposite); PCeedChk(ierr);
   MFEM_ASSERT(isComposite, "");

   CeedOperator op_coarse;
   ierr = CeedCompositeOperatorCreate(internal::ceed,
                                      &op_coarse); PCeedChk(ierr);

   int nsub;
   ierr = CeedOperatorGetNumSub(op, &nsub); PCeedChk(ierr);
   CeedOperator *subops;
   ierr = CeedOperatorGetSubList(op, &subops); PCeedChk(ierr);
   for (int isub=0; isub<nsub; ++isub)
   {
      CeedOperator subop = subops[isub];
      CeedBasis basis_coarse, basis_c2f;
      CeedOperator subop_coarse, t_subop_coarse;
      ierr = CeedATPMGOperator(subop, order_reduction, er, &basis_coarse,
                               &basis_c2f, &t_subop_coarse); PCeedChk(ierr);
      if (qorder_reduction == 0)
      {
         subop_coarse = t_subop_coarse;
      }
      else
      {
         CeedVector qcoarsen_assembledqf;
         CeedQFunctionContext qcoarsen_context;
         ierr = CeedOperatorQCoarsen(t_subop_coarse, qorder_reduction,
                                     &subop_coarse, &qcoarsen_assembledqf,
                                     &qcoarsen_context, CEED_GAUSS,
                                     CEED_GAUSS); PCeedChk(ierr);
         // next line todo: delete inside previous function?
         ierr = CeedVectorDestroy(&qcoarsen_assembledqf); PCeedChk(ierr);
         ierr = CeedQFunctionContextDestroy(&qcoarsen_context); PCeedChk(ierr);
         ierr = CeedOperatorDestroy(&t_subop_coarse); PCeedChk(ierr);
      }
      // destructions below make sense because these objects are
      // refcounted by existing objects
      ierr = CeedBasisDestroy(&basis_coarse); PCeedChk(ierr);
      ierr = CeedBasisDestroy(&basis_c2f); PCeedChk(ierr);
      ierr = CeedCompositeOperatorAddSub(op_coarse, subop_coarse);
      PCeedChk(ierr);
      ierr = CeedOperatorDestroy(&subop_coarse); PCeedChk(ierr);
   }
   return op_coarse;
}

AlgebraicMultigrid::AlgebraicMultigrid(
   AlgebraicSpaceHierarchy &hierarchy,
   BilinearForm &form,
   const Array<int> &ess_tdofs
) : GeometricMultigrid(hierarchy)
{
   // Construct finest level
   ceed_operators.Prepend(CreateCeedCompositeOperatorFromBilinearForm(form));
   essentialTrueDofs.Prepend(new Array<int>);
   *essentialTrueDofs[0] = ess_tdofs;

   int current_order = hierarchy.GetFESpaceAtLevel(0).GetOrder(0);

   // Construct interpolation, operators, at all levels of hierarchy by coarsening
   while (current_order > 1)
   {
      // TODO: make order reduction decision based on the assembled qfunction
      // (ie, what comes out of CeedOperatorLinearAssembleQFunction,
      // which currently only gets called in CeedQFunctionQCoarsen...)
      const int order_reduction = current_order - (current_order/2);
      hierarchy.PrependPCoarsenedLevel(current_order, order_reduction);
      current_order = current_order / 2;

      AlgebraicCoarseSpace &space = hierarchy.GetAlgebraicCoarseSpace(0);
      ceed_operators.Prepend(CoarsenCeedCompositeOperator(
                                ceed_operators[0], space.GetCeedElemRestriction(),
                                space.GetCeedCoarseToFine(), space.GetOrderReduction(),
                                space.GetOrderReduction()));
      mfem::Operator *P = hierarchy.GetProlongationAtLevel(0);
      essentialTrueDofs.Prepend(new Array<int>);
      CoarsenEssentialDofs(*P, *essentialTrueDofs[1],
                           *essentialTrueDofs[0]);
   }

   int nlevels = fespaces.GetNumLevels();
   // Add the operators and smoothers to the hierarchy, from coarse to fine
   for (int ilevel=0; ilevel<nlevels; ++ilevel)
   {
      FiniteElementSpace &space = hierarchy.GetFESpaceAtLevel(ilevel);
      const mfem::Operator *P = space.GetProlongationMatrix();
      ConstrainedOperator *op = new ConstrainedOperator(
         ceed_operators[ilevel], *essentialTrueDofs[ilevel], P);
      Solver *smoother;
      if (ilevel != 0)
      {
         smoother = BuildSmootherFromCeed(*op, true);
      }
      else
      {
         bool assemble_matrix = false;
#ifdef MFEM_USE_MPI
#ifdef MFEM_USE_AMGX
         if (Device::Allows(Backend::CUDA)) { assemble_matrix = true; }
#else
         if (!Device::Allows(Backend::CUDA)) { assemble_matrix = true; }
#endif
         HypreParMatrix *P_mat = NULL;
         if (assemble_matrix)
         {
            if (nlevels == 1)
            {
               // Only one level -- no coarsening, finest level
               ParFiniteElementSpace *pfes
                  = dynamic_cast<ParFiniteElementSpace*>(&space);
               if (pfes) { P_mat = pfes->Dof_TrueDof_Matrix(); }
            }
            else
            {
               ParAlgebraicCoarseSpace *pspace
                  = dynamic_cast<ParAlgebraicCoarseSpace*>(&space);
               if (pspace) { P_mat = pspace->GetProlongationHypreParMatrix(); }
            }
         }
         if (P_mat)
         {
            smoother = new AssembledAMG(*op, P_mat, Device::Allows(Backend::CUDA));
         }
         else
#endif
         {
            smoother = BuildSmootherFromCeed(*op, true);
         }
      }
      AddLevel(op, smoother, true, true);
   }
}

AlgebraicMultigrid::~AlgebraicMultigrid()
{
}

int AlgebraicInterpolation::Initialize(
   Ceed ceed, CeedBasis basisctof,
   CeedElemRestriction erestrictu_coarse, CeedElemRestriction erestrictu_fine)
{
   int ierr = 0;

   int height, width;
   ierr = CeedElemRestrictionGetLVectorSize(erestrictu_coarse, &width);
   CeedChk(ierr);
   ierr = CeedElemRestrictionGetLVectorSize(erestrictu_fine, &height);
   CeedChk(ierr);

   // interpolation qfunction
   const int bp3_ncompu = 1;
   CeedQFunction l_qf_restrict, l_qf_prolong;
   ierr = CeedQFunctionCreateIdentity(ceed, bp3_ncompu, CEED_EVAL_NONE,
                                      CEED_EVAL_INTERP, &l_qf_restrict); CeedChk(ierr);
   ierr = CeedQFunctionCreateIdentity(ceed, bp3_ncompu, CEED_EVAL_INTERP,
                                      CEED_EVAL_NONE, &l_qf_prolong); CeedChk(ierr);

   qf_restrict = l_qf_restrict;
   qf_prolong = l_qf_prolong;

   CeedVector c_fine_multiplicity;
   ierr = CeedVectorCreate(ceed, height, &c_fine_multiplicity); CeedChk(ierr);
   ierr = CeedVectorSetValue(c_fine_multiplicity, 0.0); CeedChk(ierr);

   // Create the restriction operator
   // Restriction - Fine to coarse
   ierr = CeedOperatorCreate(ceed, qf_restrict, CEED_QFUNCTION_NONE,
                             CEED_QFUNCTION_NONE, &op_restrict); CeedChk(ierr);
   ierr = CeedOperatorSetField(op_restrict, "input", erestrictu_fine,
                               CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE); CeedChk(ierr);
   ierr = CeedOperatorSetField(op_restrict, "output", erestrictu_coarse,
                               basisctof, CEED_VECTOR_ACTIVE); CeedChk(ierr);

   // Interpolation - Coarse to fine
   // Create the prolongation operator
   ierr =  CeedOperatorCreate(ceed, qf_prolong, CEED_QFUNCTION_NONE,
                              CEED_QFUNCTION_NONE, &op_interp); CeedChk(ierr);
   ierr =  CeedOperatorSetField(op_interp, "input", erestrictu_coarse,
                                basisctof, CEED_VECTOR_ACTIVE); CeedChk(ierr);
   ierr = CeedOperatorSetField(op_interp, "output", erestrictu_fine,
                               CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE); CeedChk(ierr);

   ierr = CeedElemRestrictionGetMultiplicity(erestrictu_fine,
                                             c_fine_multiplicity); CeedChk(ierr);
   ierr = CeedVectorCreate(ceed, height, &fine_multiplicity_r); CeedChk(ierr);

   CeedScalar* fine_r_data;
   const CeedScalar* fine_data;
   ierr = CeedVectorGetArray(fine_multiplicity_r, CEED_MEM_HOST,
                             &fine_r_data); CeedChk(ierr);
   ierr = CeedVectorGetArrayRead(c_fine_multiplicity, CEED_MEM_HOST,
                                 &fine_data); CeedChk(ierr);
   for (int i = 0; i < height; ++i)
   {
      fine_r_data[i] = 1.0 / fine_data[i];
   }

   ierr = CeedVectorRestoreArray(fine_multiplicity_r, &fine_r_data); CeedChk(ierr);
   ierr = CeedVectorRestoreArrayRead(c_fine_multiplicity, &fine_data);
   CeedChk(ierr);
   ierr = CeedVectorDestroy(&c_fine_multiplicity); CeedChk(ierr);

   ierr = CeedVectorCreate(ceed, height, &fine_work); CeedChk(ierr);

   ierr = CeedVectorCreate(ceed, height, &v_); CeedChk(ierr);
   ierr = CeedVectorCreate(ceed, width, &u_); CeedChk(ierr);

   return 0;
}

int AlgebraicInterpolation::Finalize()
{
   int ierr;

   ierr = CeedQFunctionDestroy(&qf_restrict); CeedChk(ierr);
   ierr = CeedQFunctionDestroy(&qf_prolong); CeedChk(ierr);
   ierr = CeedOperatorDestroy(&op_interp); CeedChk(ierr);
   ierr = CeedOperatorDestroy(&op_restrict); CeedChk(ierr);
   ierr = CeedVectorDestroy(&fine_multiplicity_r); CeedChk(ierr);
   ierr = CeedVectorDestroy(&fine_work); CeedChk(ierr);

   return 0;
}

AlgebraicInterpolation::AlgebraicInterpolation(
   Ceed ceed, CeedBasis basisctof,
   CeedElemRestriction erestrictu_coarse,
   CeedElemRestriction erestrictu_fine)
{
   int ierr;
   int lo_nldofs, ho_nldofs;
   ierr = CeedElemRestrictionGetLVectorSize(erestrictu_coarse, &lo_nldofs);
   PCeedChk(ierr);
   ierr = CeedElemRestrictionGetLVectorSize(erestrictu_fine,
                                            &ho_nldofs); PCeedChk(ierr);
   height = ho_nldofs;
   width = lo_nldofs;
   owns_basis_ = false;
   Initialize(ceed, basisctof, erestrictu_coarse, erestrictu_fine);
}

AlgebraicInterpolation::~AlgebraicInterpolation()
{
   int ierr;
   ierr = CeedVectorDestroy(&v_); PCeedChk(ierr);
   ierr = CeedVectorDestroy(&u_); PCeedChk(ierr);
   if (owns_basis_)
   {
      ierr = CeedBasisDestroy(&basisctof_); PCeedChk(ierr);
   }
   Finalize();
}

/// a = a (pointwise*) b
/// @todo: using MPI_FORALL in this Ceed-like function is ugly
int CeedVectorPointwiseMult(CeedVector a, const CeedVector b)
{
   int ierr;
   Ceed ceed;
   CeedVectorGetCeed(a, &ceed);

   int length, length2;
   ierr = CeedVectorGetLength(a, &length); CeedChk(ierr);
   ierr = CeedVectorGetLength(b, &length2); CeedChk(ierr);
   if (length != length2)
   {
      return CeedError(ceed, 1, "Vector sizes don't match");
   }

   CeedMemType mem;
   if (Device::Allows(Backend::DEVICE_MASK))
   {
      mem = CEED_MEM_DEVICE;
   }
   else
   {
      mem = CEED_MEM_HOST;
   }
   CeedScalar *a_data;
   const CeedScalar *b_data;
   ierr = CeedVectorGetArray(a, mem, &a_data); CeedChk(ierr);
   ierr = CeedVectorGetArrayRead(b, mem, &b_data); CeedChk(ierr);
   MFEM_FORALL(i, length,
   {a_data[i] *= b_data[i];});

   ierr = CeedVectorRestoreArray(a, &a_data); CeedChk(ierr);
   ierr = CeedVectorRestoreArrayRead(b, &b_data); CeedChk(ierr);

   return 0;
}

void AlgebraicInterpolation::Mult(const mfem::Vector& x, mfem::Vector& y) const
{
   int ierr = 0;
   const CeedScalar *in_ptr;
   CeedScalar *out_ptr;
   CeedMemType mem;
   ierr = CeedGetPreferredMemType(internal::ceed, &mem); PCeedChk(ierr);
   if ( Device::Allows(Backend::DEVICE_MASK) && mem==CEED_MEM_DEVICE )
   {
      in_ptr = x.Read();
      out_ptr = y.ReadWrite();
   }
   else
   {
      in_ptr = x.HostRead();
      out_ptr = y.HostReadWrite();
      mem = CEED_MEM_HOST;
   }
   ierr = CeedVectorSetArray(u_, mem, CEED_USE_POINTER,
                             const_cast<CeedScalar*>(in_ptr)); PCeedChk(ierr);
   ierr = CeedVectorSetArray(v_, mem, CEED_USE_POINTER,
                             out_ptr); PCeedChk(ierr);

   ierr = CeedOperatorApply(op_interp, u_, v_,
                            CEED_REQUEST_IMMEDIATE); PCeedChk(ierr);
   ierr = CeedVectorPointwiseMult(v_, fine_multiplicity_r); PCeedChk(ierr);

   ierr = CeedVectorTakeArray(u_, mem, const_cast<CeedScalar**>(&in_ptr));
   PCeedChk(ierr);
   ierr = CeedVectorTakeArray(v_, mem, &out_ptr); PCeedChk(ierr);
}

void AlgebraicInterpolation::MultTranspose(const mfem::Vector& x,
                                           mfem::Vector& y) const
{
   int ierr = 0;
   CeedMemType mem;
   ierr = CeedGetPreferredMemType(internal::ceed, &mem); PCeedChk(ierr);
   const CeedScalar *in_ptr;
   CeedScalar *out_ptr;
   if ( Device::Allows(Backend::DEVICE_MASK) && mem==CEED_MEM_DEVICE )
   {
      in_ptr = x.Read();
      out_ptr = y.ReadWrite();
   }
   else
   {
      in_ptr = x.HostRead();
      out_ptr = y.HostReadWrite();
      mem = CEED_MEM_HOST;
   }
   ierr = CeedVectorSetArray(v_, mem, CEED_USE_POINTER,
                             const_cast<CeedScalar*>(in_ptr)); PCeedChk(ierr);
   ierr = CeedVectorSetArray(u_, mem, CEED_USE_POINTER,
                             out_ptr); PCeedChk(ierr);

   int length;
   ierr = CeedVectorGetLength(v_, &length); PCeedChk(ierr);

   const CeedScalar *multiplicitydata;
   CeedScalar *workdata;
   ierr = CeedVectorGetArrayRead(fine_multiplicity_r, mem,
                                 &multiplicitydata); PCeedChk(ierr);
   ierr = CeedVectorGetArray(fine_work, mem, &workdata); PCeedChk(ierr);
   MFEM_FORALL(i, length,
   {workdata[i] = in_ptr[i] * multiplicitydata[i];});
   ierr = CeedVectorRestoreArrayRead(fine_multiplicity_r,
                                     &multiplicitydata);
   ierr = CeedVectorRestoreArray(fine_work, &workdata); PCeedChk(ierr);

   ierr = CeedOperatorApply(op_restrict, fine_work, u_,
                            CEED_REQUEST_IMMEDIATE); PCeedChk(ierr);

   ierr = CeedVectorTakeArray(v_, mem, const_cast<CeedScalar**>(&in_ptr));
   PCeedChk(ierr);
   ierr = CeedVectorTakeArray(u_, mem, &out_ptr); PCeedChk(ierr);
}

AlgebraicSpaceHierarchy::AlgebraicSpaceHierarchy(FiniteElementSpace &fes)
{
   meshes.Prepend(fes.GetMesh());
   ownedMeshes.Prepend(false);
   fespaces.Prepend(&fes);
   ownedFES.Prepend(false);

   Ceed ceed = internal::ceed;
   InitTensorRestriction(fes, ceed, &fine_er);
}

void AlgebraicSpaceHierarchy::AddCoarseLevel(AlgebraicCoarseSpace* space,
                                             CeedElemRestriction er)
{
   MFEM_VERIFY(meshes.Size() >= 1, "At least one level must exist!");
   Mesh* finemesh = meshes[0];
   meshes.Prepend(finemesh); // every entry of meshes points to finest mesh
   ownedMeshes.Prepend(false);
   fespaces.Prepend(space);
   ownedFES.Prepend(true); // owns all but finest
   ceed_interpolations.Prepend(new AlgebraicInterpolation(
                                  internal::ceed,
                                  space->GetCeedCoarseToFine(),
                                  space->GetCeedElemRestriction(),
                                  er)
                              );
   const SparseMatrix *R = fespaces[1]->GetRestrictionMatrix();
   if (R)
   {
      R->BuildTranspose();
      R_tr.Prepend(new TransposeOperator(*R));
   }
   else
   {
      R_tr.Prepend(NULL);
   }
   prolongations.Prepend(ceed_interpolations[0]->SetupRAP(
                            space->GetProlongationMatrix(), R_tr[0]));
   ownedProlongations.Prepend(prolongations[0] != ceed_interpolations[0]);
}

// the ifdefs and dynamic casts are very ugly, but the interface is kinda nice?
void AlgebraicSpaceHierarchy::PrependPCoarsenedLevel(
   int current_order, int order_reduction)
{
   MFEM_VERIFY(fespaces.Size() >= 1, "At least one level must exist!");
   int dim = meshes[0]->Dimension();

   AlgebraicCoarseSpace *fine_alg_space =
      dynamic_cast<AlgebraicCoarseSpace*>(fespaces[0]);
   CeedElemRestriction current_er;
#ifdef MFEM_USE_MPI
   GroupCommunicator *gc = NULL;
#endif
   if (fine_alg_space)
   {
      current_er = fine_alg_space->GetCeedElemRestriction();
#ifdef MFEM_USE_MPI
      ParAlgebraicCoarseSpace *par_alg_space =
         dynamic_cast<ParAlgebraicCoarseSpace*>(fine_alg_space);
      if (par_alg_space) { gc = par_alg_space->GetGroupCommunicator(); }
#endif
   }
   else
   {
      current_er = fine_er;
#ifdef MFEM_USE_MPI
      ParFiniteElementSpace *pfes =
         dynamic_cast<ParFiniteElementSpace*>(fespaces[0]);
      if (pfes) { gc = &pfes->GroupComm(); }
#endif
   }
   AlgebraicCoarseSpace *space;
#ifdef MFEM_USE_MPI
   if (gc)
   {
      space = new ParAlgebraicCoarseSpace(
         *fespaces[0], current_er, current_order, dim, order_reduction, gc);
   }
   else
#endif
   {
      space = new AlgebraicCoarseSpace(
         *fespaces[0], current_er, current_order, dim, order_reduction);
   }
   AddCoarseLevel(space, current_er);
}

AlgebraicCoarseSpace::AlgebraicCoarseSpace(
   FiniteElementSpace &fine_fes,
   CeedElemRestriction fine_er,
   int order,
   int dim,
   int order_reduction_
) : order_reduction(order_reduction_)
{
   int ierr;
   order_reduction = order_reduction_;

   ierr = CeedATPMGElemRestriction(order, order_reduction, fine_er,
                                   &ceed_elem_restriction, dof_map);
   PCeedChk(ierr);
   ierr = CeedBasisATPMGCoarseToFine(internal::ceed, order+1, dim,
                                     order_reduction, &coarse_to_fine);
   PCeedChk(ierr);
   ierr = CeedElemRestrictionGetLVectorSize(ceed_elem_restriction, &ndofs);
   PCeedChk(ierr);
   mesh = fine_fes.GetMesh();
}

AlgebraicCoarseSpace::~AlgebraicCoarseSpace()
{
   int ierr;
   delete [] dof_map;
   ierr = CeedBasisDestroy(&coarse_to_fine); PCeedChk(ierr);
   ierr = CeedElemRestrictionDestroy(&ceed_elem_restriction); PCeedChk(ierr);
}

#ifdef MFEM_USE_MPI

ParAlgebraicCoarseSpace::ParAlgebraicCoarseSpace(
   FiniteElementSpace &fine_fes,
   CeedElemRestriction fine_er,
   int order,
   int dim,
   int order_reduction_,
   GroupCommunicator *gc_fine)
   : AlgebraicCoarseSpace(fine_fes, fine_er, order, dim, order_reduction_)
{
   int lsize;
   CeedElemRestrictionGetLVectorSize(ceed_elem_restriction, &lsize);
   const Table &group_ldof_fine = gc_fine->GroupLDofTable();

   ldof_group.SetSize(lsize);
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
   ldof_ltdof.SetSize(lsize);
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

   R_mat = new SparseMatrix(ltsize, lsize);
   for (int j=0; j<lsize; ++j)
   {
      if (group_topo.IAmMaster(ldof_group[j]))
      {
         int i = ldof_ltdof[j];
         R_mat->Set(i,j,1.0);
      }
   }
   R_mat->Finalize();

   if (Device::Allows(Backend::DEVICE_MASK))
   {
      P = new DeviceConformingProlongationOperator(*gc, R_mat);
   }
   else
   {
      P = new ConformingProlongationOperator(lsize, *gc);
   }
   P_mat = NULL;
}

HypreParMatrix *ParAlgebraicCoarseSpace::GetProlongationHypreParMatrix()
{
   if (P_mat) { return P_mat; }

   ParMesh *pmesh = dynamic_cast<ParMesh*>(mesh);
   MFEM_VERIFY(pmesh != NULL, "");
   Array<HYPRE_Int> dof_offsets, tdof_offsets, tdof_nb_offsets;
   Array<HYPRE_Int> *offsets[2] = {&dof_offsets, &tdof_offsets};
   int lsize = P->Height();
   int ltsize = P->Width();
   HYPRE_Int loc_sizes[2] = {lsize, ltsize};
   pmesh->GenerateOffsets(2, loc_sizes, offsets);

   MPI_Comm comm = pmesh->GetComm();

   const GroupTopology &group_topo = gc->GetGroupTopology();

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

   P_mat = new HypreParMatrix(
      comm, pmesh->GetMyRank(), pmesh->GetNRanks(),
      row_starts, col_starts,
      i_diag, j_diag, i_offd, j_offd,
      cmap, offd_counter
   );

   P_mat->CopyRowStarts();
   P_mat->CopyColStarts();

   return P_mat;
}

ParAlgebraicCoarseSpace::~ParAlgebraicCoarseSpace()
{
   delete P;
   delete R_mat;
   delete P_mat;
   delete gc;
}

#endif // MFEM_USE_MPI

#endif // MFEM_USE_CEED

AlgebraicSolver::AlgebraicSolver(BilinearForm &form,
                                 const Array<int>& ess_tdofs)
{
   MFEM_VERIFY(DeviceCanUseCeed(),
               "AlgebraicSolver requires a Ceed device");
   MFEM_VERIFY(form.GetAssemblyLevel() == AssemblyLevel::PARTIAL ||
               form.GetAssemblyLevel() == AssemblyLevel::NONE,
               "AlgebraicSolver requires partial assembly or fully matrix-free.");
   MFEM_VERIFY(UsesTensorBasis(*form.FESpace()),
               "AlgebraicSolver requires tensor product basis functions.");
#ifdef MFEM_USE_CEED
   fespaces = new AlgebraicSpaceHierarchy(*form.FESpace());
   multigrid = new AlgebraicMultigrid(*fespaces, form, ess_tdofs);
#else
   MFEM_ABORT("AlgebraicSolver requires Ceed support");
#endif
}

AlgebraicSolver::~AlgebraicSolver()
{
#ifdef MFEM_USE_CEED
   delete fespaces;
   delete multigrid;
#endif
}

void AlgebraicSolver::Mult(const Vector& x, Vector& y) const
{
#ifdef MFEM_USE_CEED
   multigrid->Mult(x, y);
#endif
}

void AlgebraicSolver::SetOperator(const mfem::Operator& op)
{
#ifdef MFEM_USE_CEED
   multigrid->SetOperator(op);
#endif
}

} // namespace ceed

} // namespace mfem
