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


namespace mfem
{


const Operator * FilteredSolver::GetPtAP(const Operator *Aop,
                                         const Operator *Pop) const
{
#ifdef MFEM_USE_MPI
   const HypreParMatrix * Ah = dynamic_cast<const HypreParMatrix*>(Aop);
   const HypreParMatrix * Ph = dynamic_cast<const HypreParMatrix*>(Pop);
   if (Ah && Ph) { return RAP(Ah, Ph); }
#endif
#ifdef MFEM_USE_PETSC
   const PetscParMatrix* Ap = dynamic_cast<const PetscParMatrix*>(Aop);
   const PetscParMatrix* Pp = dynamic_cast<const PetscParMatrix*>(Pop);
   if (Ap && Pp) { return RAP(Ap, Pp); }
#endif
   const SparseMatrix * Asp = dynamic_cast<const SparseMatrix*>(Aop);
   const SparseMatrix * Psp = dynamic_cast<const SparseMatrix*>(Pop);
   if (Asp && Psp)   { return RAP(*Asp, *Psp); }

   return new RAPOperator(*Pop, *Aop, *Pop);
}

void FilteredSolver::MakeSolver() const
{
   if (solver_set) { return; }

   MFEM_VERIFY(A, "FilteredSolver::MakeSolver: Operator not set");
   MFEM_VERIFY(P, "FilteredSolver::MakeSolver: Transfer operator not set");
   MFEM_VERIFY(M, "FilteredSolver::MakeSolver: Solver is not set.");
   MFEM_VERIFY(Mf,"FilteredSolver::MakeSolver: Filtered space solver is not set.");

   // Original space solver
   M->SetOperator(*A);

   // Filtered space operator
   delete PtAP;
   PtAP = GetPtAP(A, P);

   // Filtered space solver
   Mf->SetOperator(*PtAP);

   z.SetSize(height);
   r.SetSize(height);
   xf.SetSize(P->Width());
   rf.SetSize(P->Width());

   solver_set = true;
}

void FilteredSolver::SetOperator(const Operator &op)
{
   A = &op;
   height = op.Height();
   width = op.Width();
   solver_set = false;
}

void FilteredSolver::SetSolver(Solver &M_)
{
   M = &M_;
   solver_set = false;
}

void FilteredSolver::SetFilteredSubspaceTransferOperator(const Operator &P_)
{
   P = &P_;
   solver_set = false;
}

void FilteredSolver::SetFilteredSubspaceSolver(Solver &Mf_)
{
   Mf = &Mf_;
   solver_set = false;
}

void FilteredSolver::Mult(const Vector &b, Vector &x) const
{
   MFEM_VERIFY(b.Size() == x.Size(),
               "FilteredSolver::Mult: Inconsistent x and y size");
   MakeSolver();

   x = 0.0;
   r = b;

   // z = M x
   M->Mult(b, z);
   // x = x + z
   x+=z;

   // r = b - A x = r - A z
   A->AddMult(z, r, -1.0);

   // rf = Pᵀ r
   P->MultTranspose(r, rf);

   // xf = Mf rf
   Mf->Mult(rf, xf);

   // z = P xf
   P->Mult(xf, z);

   // x = x + z
   x+=z;

   // r = b - A x = r - A z
   A->AddMult(z, r, -1.0);

   // z = M r
   M->Mult(r, z);
   x+=z;
}

FilteredSolver::~FilteredSolver()
{
   delete PtAP;
}


} // namespace mfem