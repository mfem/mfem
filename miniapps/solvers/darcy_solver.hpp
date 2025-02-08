// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_DARCY_SOLVER_HPP
#define MFEM_DARCY_SOLVER_HPP

#include "mfem.hpp"
#include <memory>
#include <vector>

namespace mfem
{
namespace blocksolvers
{
struct IterSolveParameters
{
   int print_level = 0;
   int max_iter = 500;
#if defined(MFEM_USE_DOUBLE)
   real_t abs_tol = 1e-12;
   real_t rel_tol = 1e-9;
#elif defined(MFEM_USE_SINGLE)
   real_t abs_tol = 1e-8;
   real_t rel_tol = 1e-5;
#else
#error "Only single and double precision are supported!"
   real_t abs_tol = 1e-12;
   real_t rel_tol = 1e-9;
#endif
};

void SetOptions(IterativeSolver& solver, const IterSolveParameters& param);

/// Abstract solver class for Darcy's flow
class DarcySolver : public Solver
{
protected:
   Array<int> offsets_;
public:
   DarcySolver(int size0, int size1) : Solver(size0 + size1),
      offsets_({0, size0, height}) { }
   virtual int GetNumIterations() const = 0;
};

/// Wrapper for the block-diagonal-preconditioned MINRES employed in ex5p.cpp
class BDPMinresSolver : public DarcySolver
{
   BlockOperator op_;
   BlockDiagonalPreconditioner prec_;
   OperatorPtr BT_;
   OperatorPtr S_;   // S_ = B diag(M)^{-1} B^T
   MINRESSolver solver_;
   Array<int> ess_zero_dofs_;
public:
   BDPMinresSolver(const HypreParMatrix& M,
                   const HypreParMatrix& B,
                   IterSolveParameters param);
   virtual void Mult(const Vector & x, Vector & y) const;
   virtual void SetOperator(const Operator &op) { }
   void SetEssZeroDofs(const Array<int>& dofs) { dofs.Copy(ess_zero_dofs_); }
   virtual int GetNumIterations() const { return solver_.GetNumIterations(); }
};
} // namespace blocksolvers
} // namespace mfem

#endif // MFEM_DARCY_SOLVER_HPP
