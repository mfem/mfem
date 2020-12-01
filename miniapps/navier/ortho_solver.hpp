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

#ifndef MFEM_NAVIER_ORTHO_SOLVER_HPP
#define MFEM_NAVIER_ORTHO_SOLVER_HPP

#include "mfem.hpp"

namespace mfem
{
namespace navier
{
/// Solver wrapper which orthogonalizes the input and output vector
/**
 * OrthoSolver wraps an existing Operator and orthogonalizes the input vector
 * before passing it to the Mult method of the Operator. This is a convenience
 * implementation to handle e.g. a Poisson problem with pure Neumann boundary
 * conditions, where this procedure removes the Nullspace.
 */
class OrthoSolver : public Solver
{
public:
   OrthoSolver();

   virtual void SetOperator(const Operator &op);

   void Mult(const Vector &b, Vector &x) const;

private:
   const Operator *oper = nullptr;

   mutable Vector b_ortho;

   void Orthogonalize(const Vector &v, Vector &v_ortho) const;
};

} // namespace navier

} // namespace mfem

#endif
