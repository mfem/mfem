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

#include "navier_preconditioners.hpp"

namespace mfem
{
namespace dfem_navier
{

void Orthogonalize(MPI_Comm comm, Vector &x)
{
   HYPRE_BigInt global_size = x.Size();
   real_t global_sum = x.Sum();
   MPI_Allreduce(MPI_IN_PLACE, &global_size, 1, HYPRE_MPI_BIG_INT,
                 MPI_SUM, comm);
   MPI_Allreduce(MPI_IN_PLACE, &global_sum, 1,
                 MPITypeMap<real_t>::mpi_type, MPI_SUM, comm);
   x -= global_sum / global_size;
}

PressureOrthoSolver::PressureOrthoSolver(MPI_Comm comm, int size,
                                         Solver &solver)
   : Solver(size), comm(comm), solver(solver) { }

void PressureOrthoSolver::SetOperator(const Operator &op)
{
   solver.SetOperator(op);
   height = op.Height();
   width = op.Width();
}

void PressureOrthoSolver::Mult(const Vector &b, Vector &x) const
{
   projected_rhs = b;
   ProjectPressure(projected_rhs);
   solver.iterative_mode = iterative_mode;
   solver.Mult(projected_rhs, x);
   ProjectPressure(x);
}

void PressureOrthoSolver::ProjectPressure(Vector &x) const
{
   Orthogonalize(comm, x);
}

BlockDiagonalPreconditioner::BlockDiagonalPreconditioner(
   MPI_Comm comm_, const Array<int> &block_offsets,
   int velocity_block_, int pressure_block_)
   : Solver(block_offsets.Last()), comm(comm_),
     velocity_block(velocity_block_), pressure_block(pressure_block_),
     block_preconditioner(block_offsets) { }

void BlockDiagonalPreconditioner::SetOperator(const Operator &op)
{
   const auto *jacobian = dynamic_cast<const BlockOperator *>(&op);
   MFEM_VERIFY(jacobian, "expected a block Navier-Stokes Jacobian");
   const auto &velocity = dynamic_cast<const HypreParMatrix &>(
                             jacobian->GetBlock(velocity_block,
                                                velocity_block));
   const auto &divergence = dynamic_cast<const HypreParMatrix &>(
                               jacobian->GetBlock(pressure_block,
                                                  velocity_block));

   velocity_amg = std::make_unique<HypreBoomerAMG>(velocity);
   velocity_amg->SetPrintLevel(0);
   velocity_amg->iterative_mode = false;
   block_preconditioner.SetDiagonalBlock(velocity_block, velocity_amg.get());

   // Build the Schur complement S = D diag(C)^{-1} D^T and its AMG preconditioner.
   Vector velocity_diagonal;
   velocity.GetDiag(velocity_diagonal);
   std::unique_ptr<HypreParMatrix> scaled_gradient(divergence.Transpose());
   scaled_gradient->InvScaleRows(velocity_diagonal);
   pressure_schur.reset(ParMult(&divergence, scaled_gradient.get()));
   pressure_amg = std::make_unique<HypreBoomerAMG>(*pressure_schur);
   pressure_amg->SetPrintLevel(0);
   pressure_amg->iterative_mode = false;
   // S is singular (S 1_p = 0) and AMG is not nullspace aware: it maps a
   // zero-mean vector to one with a constant component. Projecting the AMG
   // output keeps the pressure block in the zero-mean subspace, where S is
   // SPD, instead of letting the nullspace accumulate.
   pressure_ortho = std::make_unique<PressureOrthoSolver>(
                       comm, pressure_schur->Height(), *pressure_amg);
   pressure_ortho->SetOperator(*pressure_schur);
   block_preconditioner.SetDiagonalBlock(pressure_block, pressure_ortho.get());
}

void BlockDiagonalPreconditioner::Mult(const Vector &x, Vector &y) const
{
   block_preconditioner.Mult(x, y);
}

} // namespace dfem_navier
} // namespace mfem