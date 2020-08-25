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

#ifndef MFEM_CEED_ALGEBRAIC_HPP
#define MFEM_CEED_ALGEBRAIC_HPP

#include "mfem.hpp"
#include "ceedsolvers-utility.h"
#include "ceedsolvers-atpmg.h"
#include "ceedsolvers-interpolation.h"

namespace mfem
{
namespace navier
{

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

/// Do you want to wrap this in a ConstrainedOperator or do
/// you want to do the ess_tdof stuff here?
class MFEMCeedJacobi : public mfem::Operator
{
public:
   MFEMCeedJacobi(Ceed ceed,
                  int size,
                  CeedVector diagonal,
                  const mfem::Array<int>& ess_tdof_list,
                  double scale=1.0);
   ~MFEMCeedJacobi();

   virtual void Mult(const mfem::Vector& x, mfem::Vector& y) const;

   virtual void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const;

private:
   const mfem::Array<int>& ess_tdof_list_;
   CeedVector inv_diag_;
   CeedVector u_, v_;
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
   wrap CeedInterpolation object in an mfem::Operator
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

   Long term, this becomes more of a Ceed object and less of an
   MFEM object

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

   MFEMCeedJacobi * nobc_smoother_;
   mfem::Operator * smoother_;
   MFEMCeedInterpolation * mfem_interp_;

   mfem::Array<int> lo_ess_tdof_list_;
};


/**
   I think the basic idea is that we loop from fine to coarse
   making CeedMultigridLevel objects, make a coarsest solver, and then
   loop back up to the fine level making CeedMultigridVCyle objects?
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

class CeedCGWithAMG : public mfem::Solver
{
public:
   CeedCGWithAMG(CeedOperator oper,
                 mfem::Array<int>& ess_tdof_list,
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
   // mfem::HypreBoomerAMG * hypre_inner_prec_;
   mfem::Solver * inner_prec_;
   mfem::Solver * solver_;
};

class CeedPlainCG : public mfem::Solver
{
public:
   CeedPlainCG(CeedOperator oper,
               mfem::Array<int>& ess_tdof_list);

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


} // namespace navier
} // namespace mfem

#endif
