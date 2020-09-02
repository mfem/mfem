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

namespace mfem
{

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
class CeedMultigridVCycle;

/**
   This takes a CeedOperator with essential dofs 
   and produces a coarser / lower-order operator, an interpolation
   operator between fine/coarse levels, and a smoother.

   todo: not clear the smoother belongs in this object
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
                      int order_reduction);
   ~CeedMultigridLevel();

   /// return coarse operator as CeedOperator (no boundary conditions)
   CeedOperator GetCoarseCeed() { return coarse_oper_; }

   mfem::Array<int>& GetCoarseEssentialDofList() { return lo_ess_tdof_list_; }

   friend class CeedMultigridVCycle;

private:
   CeedElemRestriction ho_er_; // not owned

   CeedOperator oper_; // not owned
   CeedOperator coarse_oper_;
   CeedBasis coarse_basis_;
   CeedBasis basisctof_;
   CeedElemRestriction lo_er_;

   mfem::Operator * smoother_;
   MFEMCeedInterpolation * mfem_interp_;

   mfem::Array<int> lo_ess_tdof_list_;
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
private:
   CeedOperator oper_;
   CeedVector u_, v_;   // mutable?
};

class MFEMCeedOperator : public mfem::Operator
{
public:
   MFEMCeedOperator(CeedOperator oper, mfem::Array<int>& ess_tdofs) 
      :
      unconstrained_op_(oper)
   {
      unconstrained_op_.FormSystemOperator(ess_tdofs, constrained_op_);
      height = width = unconstrained_op_.Height();
   }

   MFEMCeedOperator(CeedOperator oper)
      :
      unconstrained_op_(oper)
   {
      mfem::Array<int> empty;
      unconstrained_op_.FormSystemOperator(empty, constrained_op_);
      height = width = unconstrained_op_.Height();
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
   mfem::Operator * constrained_op_;
};

class MFEMCeedVCycle : public mfem::Solver
{
public:
   MFEMCeedVCycle(const mfem::Operator& fine_operator,
                  const mfem::Solver& coarse_solver,
                  const mfem::Operator& fine_smoother,
                  const mfem::Operator& interp);

   void Mult(const mfem::Vector& x, mfem::Vector& y) const;
   void SetOperator(const Operator &op) { }

private:
   void FormResidual(const mfem::Vector& b,
                     const mfem::Vector& x,
                     mfem::Vector& r) const;

   const mfem::Operator& fine_operator_;
   const mfem::Solver& coarse_solver_;
   const mfem::Operator& fine_smoother_;
   const mfem::Operator& interp_;

   /// work vectors (too many of them, can be economized)
   mutable mfem::Vector residual_;
   mutable mfem::Vector correction_;
   mutable mfem::Vector coarse_residual_;
   mutable mfem::Vector coarse_correction_;
};

/**
   The basic idea is that we loop from fine to coarse
   making CeedMultigridLevel objects, make a coarsest solver, and then
   loop back up to the fine level making CeedMultigridVCyle objects
*/
class CeedMultigridVCycle : public mfem::Solver
{
public:
   CeedMultigridVCycle(const CeedMultigridLevel& level,
                       const mfem::Operator& fine_operator,
                       const mfem::Solver& coarse_solver);

   void SetOperator(const mfem::Operator& op) {}
   void Mult(const mfem::Vector& x, mfem::Vector& y) const;

private:
   MFEMCeedVCycle cycle_;
};

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

CeedMultigridLevel::CeedMultigridLevel(CeedOperator oper,
                                       const mfem::Array<int>& ho_ess_tdof_list,
                                       int order_reduction)
   :
   oper_(oper)
{
   const double jacobi_scale = 0.65;
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
   const CeedScalar * diagvals;
   CeedVectorGetArrayRead(diagceed, CEED_MEM_HOST, &diagvals);
   mfem::Vector mfem_diag(const_cast<CeedScalar*>(diagvals), length);
   smoother_ = new OperatorJacobiSmoother(mfem_diag, ho_ess_tdof_list, jacobi_scale);
   // need an mfem::Operator to do Chebyshev, would be possible but needs a little work
   // smoother_ = new OperatorChebyshevSmoother(mfem_diag, ho_ess_tdof_list, cheb_order, MPI_COMM_WORLD);
   CeedVectorRestoreArrayRead(diagceed, &diagvals);
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
   CeedMultigridLevel * coarsest = NULL;
   if (num_levels > 1)
   {
      coarsest = levels[num_levels - 2];
   }

   int coarse_cg_iterations = 10; // even less might be good
   if (num_levels > 1)
   {
      coarsest_solver = new CeedPlainCG(coarsest->GetCoarseCeed(),
                                        coarsest->GetCoarseEssentialDofList(),
                                        coarse_cg_iterations);
   }
   else
   {
      coarsest_solver = new CeedPlainCG(current_op, *current_ess_dofs, coarse_cg_iterations);
   }

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
