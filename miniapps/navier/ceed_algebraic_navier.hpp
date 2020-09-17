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

#ifndef MFEM_CEED_ALGEBRAIC_NAVIER_HPP
#define MFEM_CEED_ALGEBRAIC_NAVIER_HPP

#include "mfem.hpp"
#include "../../fem/libceed/ceedsolvers-utility.h"
#include "../../fem/libceed/ceedsolvers-atpmg.h"
#include "../../fem/libceed/ceedsolvers-interpolation.h"

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
               mfem::Array<int>& ess_tdof_list,
               int max_its=500);

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
