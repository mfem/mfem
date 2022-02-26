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

#include "ortho_solver.hpp"
#include "../../general/forall.hpp"

using namespace mfem;
using namespace navier;

OrthoSolver::OrthoSolver(MPI_Comm comm_) : Solver(0, true),
   comm(comm_), global_size(0) { }

void OrthoSolver::SetOperator(const Operator &op)
{
   width = op.Width();
   height = op.Height();
   MFEM_VERIFY(width == height, "OrthoSolver operator must be square.");
   oper = &op;
   MPI_Allreduce(&width, &global_size, 1, MPI_INT, MPI_SUM, comm);
   ones.SetSize(width);
   ones.UseDevice(true);
   ones = 1.0;
}

void OrthoSolver::Mult(const Vector &b, Vector &x) const
{
   // Orthogonalize input
   Orthogonalize(b, b_ortho);
   // Apply operator
   oper->Mult(b_ortho, x);
   // Orthogonalize output
   Orthogonalize(x, x);
}

void OrthoSolver::Orthogonalize(const Vector &v, Vector &v_ortho) const
{
   v_ortho.SetSize(width);

   double sum = v*ones;
   MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM, comm);

   const double ratio = sum/double(global_size);
   const auto d_v = v.Read();
   auto d_vo = v_ortho.Write();
   MFEM_FORALL(i, width,
   {
      d_vo[i] = d_v[i] - ratio;
   });
}

void OrthoSolver::Orthogonalize(Vector &v) const
{
   double sum = v*ones;
   MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM, comm);
   const double ratio = sum/double(global_size);
   v -= ratio;
}
