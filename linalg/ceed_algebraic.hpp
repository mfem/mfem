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

#include "../config/config.hpp"

#ifdef MFEM_USE_CEED
#include "operator.hpp"
#include "../fem/libceed/ceedsolvers-utility.h"

namespace mfem
{

// forward declarations (too many?)
class BilinearForm;
class MFEMCeedVCycle;
class MFEMCeedInterpolation;
class SparseMatrix;

int CeedOperatorFullAssemble(CeedOperator op, SparseMatrix ** mat);

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
   MFEMCeedOperator(CeedOperator oper, mfem::Array<int>& ess_tdofs) ;
   MFEMCeedOperator(CeedOperator oper);

   ~MFEMCeedOperator() { delete constrained_op_; }

   void Mult(const mfem::Vector& x, mfem::Vector& y) const;

private:
   UnconstrainedMFEMCeedOperator unconstrained_op_;
   mfem::Operator * constrained_op_;
};

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
                      int order_reduction);
   ~CeedMultigridLevel();

   /// return coarse operator as CeedOperator (no boundary conditions)
   CeedOperator GetCoarseCeed() { return coarse_oper_; }

   mfem::Array<int>& GetCoarseEssentialDofList() { return lo_ess_tdof_list_; }

   friend class MFEMCeedVCycle;

private:
   CeedElemRestriction ho_er_; // not owned

   CeedOperator oper_; // not owned
   CeedOperator coarse_oper_;
   CeedBasis * coarse_basis_;
   CeedBasis * basisctof_;
   CeedElemRestriction * lo_er_;

   MFEMCeedInterpolation * mfem_interp_;

   const mfem::Array<int>& ho_ess_tdof_list_;
   mfem::Array<int> lo_ess_tdof_list_;
   int numsub_;
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

class AlgebraicCeedSolver : public mfem::Solver
{
public:
   AlgebraicCeedSolver(Operator& fine_mfem_op, BilinearForm& form, 
                       Array<int>& ess_dofs, bool use_amg=false);
   ~AlgebraicCeedSolver();

   /// Note that this does not rebuild the hierarchy or smoothers,
   /// just changes the finest level operator for residual computations
   void SetOperator(const Operator& op) { operators[0] = const_cast<Operator*>(&op); }

   void Mult(const Vector& x, Vector& y) const;

private:
   int num_levels;
   Operator ** operators;
   CeedMultigridLevel ** levels;
   Solver ** solvers;
   CeedOperator fine_composite_op;
};

} // namespace mfem

#endif // MFEM_USE_CEED

#endif // MFEM_CEED_ALGEBRAIC_HPP
