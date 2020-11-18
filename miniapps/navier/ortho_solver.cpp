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

#include "ortho_solver.hpp"

using namespace mfem;
using namespace navier;

OrthoSolver::OrthoSolver() : Solver(0, true) {}

void OrthoSolver::SetOperator(const Operator &op)
{
   oper = &op;
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
   double loc_sum = v.Sum();
   double global_sum = 0.0;
   int loc_size = v.Size();
   int global_size = 0;

   MPI_Allreduce(&loc_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   MPI_Allreduce(&loc_size, &global_size, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

   double ratio = global_sum / static_cast<double>(global_size);
   v_ortho.SetSize(v.Size());
   for (int i = 0; i < v_ortho.Size(); ++i)
   {
      v_ortho(i) = v(i) - ratio;
   }
}
