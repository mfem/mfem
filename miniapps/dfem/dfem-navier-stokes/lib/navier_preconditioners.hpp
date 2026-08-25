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

#ifndef MFEM_DFEM_NAVIER_PRECONDITIONERS_HPP
#define MFEM_DFEM_NAVIER_PRECONDITIONERS_HPP

#include "mfem.hpp"

#include <memory>

namespace mfem
{
namespace dfem_navier
{

/// Project @a x orthogonally to the constant coefficient vector.
///
/// With velocity essential conditions eliminated the constant vector spans
/// the nullspace of the pressure block of the Navier-Stokes saddle system
/// (G 1_p = 0 and B^T 1_p = 0). This Euclidean projection makes a singular
/// pressure solve compatible; it is distinct from removing the FE L2 mean.
void Orthogonalize(MPI_Comm comm, Vector &x);

class PressureOrthoSolver : public Solver
{
public:
   PressureOrthoSolver(MPI_Comm comm, int size, Solver &solver);

   void SetOperator(const Operator &op) override;
   void Mult(const Vector &b, Vector &x) const override;

private:
   void ProjectPressure(Vector &x) const;

   MPI_Comm comm;
   Solver &solver;
   mutable Vector projected_rhs;
};

/// Block-diagonal Navier-Stokes preconditioner: C^{-1} approximated by one
/// AMG V-cycle and S^{-1} by one AMG V-cycle on S = D diag(C)^{-1} D^T.
class BlockDiagonalPreconditioner : public Solver
{
public:
   BlockDiagonalPreconditioner(MPI_Comm comm,
                               const Array<int> &block_offsets,
                               int velocity_block = 0,
                               int pressure_block = 1);

   void SetOperator(const Operator &op) override;
   void Mult(const Vector &x, Vector &y) const override;

private:
   MPI_Comm comm;
   int velocity_block;
   int pressure_block;
   mfem::BlockDiagonalPreconditioner block_preconditioner;
   std::unique_ptr<HypreBoomerAMG> velocity_amg;
   std::unique_ptr<HypreBoomerAMG> pressure_amg;
   std::unique_ptr<PressureOrthoSolver> pressure_ortho;
   std::unique_ptr<HypreParMatrix> pressure_schur;
};

} // namespace dfem_navier
} // namespace mfem

#endif // MFEM_DFEM_NAVIER_PRECONDITIONERS_HPP