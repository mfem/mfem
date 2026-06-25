// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_BRAGINSKII_HDG_SOLVER
#define MFEM_BRAGINSKII_HDG_SOLVER

#include "mfem.hpp"
#include "../hdg/darcyop.hpp"
#include <memory>

namespace mfem
{
namespace plasma
{

// Convection-diffusion operator
class ConvectionDiffusionOp : public TimeDependentOperator
{
   const FiniteElementSpace &fes_p;
   hdg::DarcyOperator &darcy_op;
   Operator &conv_op;

   std::unique_ptr<DGMassInverse> Minv;
   mutable Vector bp;

public:
   ConvectionDiffusionOp(const FiniteElementSpace &fes_p_,
                         hdg::DarcyOperator &darcy_op_, Operator &conv_op_);

   void AddMult(const Vector &x, Vector &y, real_t a = 1.) const override;

   void ImplicitSolve(const real_t dt, const Vector &x, Vector &y) override;

   void SetTime(real_t t) override { darcy_op.SetTime(t); }

   void SetTolerance(real_t rtol, real_t atol)
   {
      darcy_op.SetTolerance(rtol, atol);
      Minv->SetRelTol(rtol);
      Minv->SetAbsTol(atol);
   }

   void SetMaxIters(int iters)
   {
      darcy_op.SetMaxIters(iters);
      Minv->SetMaxIter(iters);
   }
};

} // namespace plasma
} // namespace mfem

#endif // MFEM_BRAGINSKII_HDG_SOLVER
