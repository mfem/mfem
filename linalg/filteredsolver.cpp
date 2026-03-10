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

#include "filteredsolver.hpp"
#include "sparsemat.hpp"
#ifdef MFEM_USE_PETSC
#include "petsc.hpp"
#endif

namespace mfem
{

std::unique_ptr<const Operator> FilteredSolver::GetPtAP(const Operator *Aop,
                                                        const Operator *Pop) const
{
#ifdef MFEM_USE_MPI
   const HypreParMatrix * Ah = dynamic_cast<const HypreParMatrix*>(Aop);
   const HypreParMatrix * Ph = dynamic_cast<const HypreParMatrix*>(Pop);
   if (Ah && Ph) { return std::unique_ptr<const Operator>(RAP(Ah, Ph)); }
#endif
#ifdef MFEM_USE_PETSC
   PetscParMatrix* Ap = const_cast<PetscParMatrix*>(
                           dynamic_cast<const PetscParMatrix*>(Aop));
   PetscParMatrix* Pp = const_cast<PetscParMatrix*>(
                           dynamic_cast<const PetscParMatrix*>(Pop));
   if (Ap && Pp) { return std::unique_ptr<const Operator>(RAP(Ap, Pp)); }
#endif
   const SparseMatrix * Asp = dynamic_cast<const SparseMatrix*>(Aop);
   const SparseMatrix * Psp = dynamic_cast<const SparseMatrix*>(Pop);
   if (Asp && Psp)   { return std::unique_ptr<const Operator>(RAP(*Asp, *Psp)); }

   return std::unique_ptr<const Operator>(new RAPOperator(*Pop, *Aop, *Pop));
}

void FilteredSolver::InitVectors() const
{
   MFEM_VERIFY(A, "Operator not set");
   MFEM_VERIFY(P, "Transfer operator not set");
   MFEM_VERIFY(B, "Solver is not set.");
   MFEM_VERIFY(S, "Filtered space solver is not set.");

   z.SetSize(height);
   z.UseDevice(true);
   r.SetSize(height);
   r.UseDevice(true);
   xf.SetSize(P->Width());
   xf.UseDevice(true);
   rf.SetSize(P->Width());
   rf.UseDevice(true);
}

void FilteredSolver::MakeSolver() const
{
   if (solver_set) { return; }

   InitVectors();

   // Original space solver
   B->SetOperator(*A);

   // Filtered space operator
   PtAP = GetPtAP(A, P);

   // Filtered space solver
   S->SetOperator(*PtAP);

   solver_set = true;
}

void FilteredSolver::SetOperator(const Operator &op)
{
   A = &op;
   height = op.Height();
   width = op.Width();
   solver_set = false;
}

void FilteredSolver::SetSolver(Solver &B_)
{
   B = &B_;
   solver_set = false;
}

void FilteredSolver::SetFilteredSubspaceTransferOperator(const Operator &P_)
{
   P = &P_;
   solver_set = false;
}

void FilteredSolver::SetFilteredSubspaceSolver(Solver &S_)
{
   S = &S_;
   solver_set = false;
}

void FilteredSolver::Mult(const Vector &b, Vector &x) const
{
   MFEM_VERIFY(b.Size() == x.Size(), "Inconsistent b and x size");
   MakeSolver();

   x = 0.0;
   r = b;

   // z = B x
   B->Mult(b, z);
   // x = x + z
   x+=z;

   // r = b - A x = r - A z
   A->AddMult(z, r, -1.0);

   // rf = Pᵀ r
   P->MultTranspose(r, rf);

   // xf = S rf
   S->Mult(rf, xf);

   // z = P xf
   P->Mult(xf, z);

   // x = x + z
   x+=z;

   // r = b - A x = r - A z
   A->AddMult(z, r, -1.0);

   // z = B r
   B->Mult(r, z);
   x+=z;
}

#ifdef MFEM_USE_MPI

void AMGFSolver::SetOperator(const Operator &A_)
{
   auto Ah = dynamic_cast<const HypreParMatrix*>(&A_);
   MFEM_VERIFY(Ah, "AMGFSolver::SetOperator: HypreParMatrix expected.");
   FilteredSolver::SetOperator(*Ah);
}

void AMGFSolver::SetFilteredSubspaceTransferOperator(const HypreParMatrix &Pop)
{
   FilteredSolver::SetFilteredSubspaceTransferOperator(Pop);
}

#endif

} // namespace mfem
