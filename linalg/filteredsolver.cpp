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

void FilteredSolver::SetupOperatorHandle(const Operator *op,
                                         OperatorHandle &handle)
{
   handle.Clear();
   SparseMatrix * Asparse = const_cast<SparseMatrix*>
                            (dynamic_cast<const SparseMatrix*>(op));
   if (Asparse) { handle.Reset(Asparse, false); return; }
#ifdef MFEM_USE_MPI
   HypreParMatrix * Ahypre = const_cast<HypreParMatrix*>
                             (dynamic_cast<const HypreParMatrix*>(op));
   if (Ahypre) { handle.Reset(Ahypre, false); return; }
#endif
#ifdef MFEM_USE_PETSC
   PetscParMatrix * Apetsc = const_cast<PetscParMatrix*>
                             (dynamic_cast<const PetscParMatrix*>(op));
   if (Apetsc) { handle.Reset(Apetsc, false); return; }
#endif
   handle.Reset(const_cast<Operator*>(op), false);
}

void FilteredSolver::MakeSolver()
{
   if (solver_set) { return; }

   MFEM_VERIFY(Ah.Ptr(), "FilteredSolver::MakeSolver: Operator not set");
   MFEM_VERIFY(Ph.Ptr(), "FilteredSolver::MakeSolver: Transfer operator not set");
   MFEM_VERIFY(M,"FilteredSolver::MakeSolver: Solver is not set.");
   MFEM_VERIFY(Mf,
               "FilteredSolver::MakeSolver: Filtered space solver is not set.");

   // Original space solver
   M->SetOperator(*Ah.Ptr());

   // Filtered space operator
   PtAPh.Clear();
   PtAPh.MakePtAP(Ah,Ph);

   // Filtered space solver
   Mf->SetOperator(*PtAPh.Ptr());

   z.SetSize(height);
   r.SetSize(height);
   xf.SetSize(Ph.Ptr()->Width());
   rf.SetSize(Ph.Ptr()->Width());

   solver_set = true;
}

void FilteredSolver::SetOperator(const Operator &op)
{
   Reset();
   SetupOperatorHandle(&op, Ah);
   height = op.Height();
   width = op.Width();
   if (M && Mf && Ph.Ptr()) { MakeSolver(); }
}

void FilteredSolver::SetSolver(Solver &M_)
{
   Reset();
   M = &M_;
   if (Mf && Ah.Ptr() && Ph.Ptr()) { MakeSolver(); }
}

void FilteredSolver::SetFilteredSubspaceTransferOperator(const Operator &P_)
{
   Reset();
   SetupOperatorHandle(&P_, Ph);
   if (M && Mf && Ah.Ptr()) { MakeSolver(); }
}

void FilteredSolver::SetFilteredSubspaceSolver(Solver &Mf_)
{
   Reset();
   Mf = &Mf_;
   if (M && Ah.Ptr() && Ph.Ptr()) { MakeSolver(); }
}

void FilteredSolver::Mult(const Vector &b, Vector &x) const
{
   MFEM_VERIFY(b.Size() == x.Size(),
               "FilteredSolver::Mult: Inconsistent x and y size");

   x = 0.0;
   r = b;

   // z = M x
   M->Mult(b, z);
   // x = x + z
   x+=z;

   // r = b - A x = r - A z
   Ah.Ptr()->AddMult(z, r, -1.0);

   // rf = Pᵀ r
   Ph.Ptr()->MultTranspose(r, rf);

   // xf = Mf rf
   Mf->Mult(rf, xf);

   // z = P xf
   Ph.Ptr()->Mult(xf, z);

   // x = x + z
   x+=z;

   // r = b - A x = r - A z
   Ah.Ptr()->AddMult(z, r, -1.0);

   // z = M r
   M->Mult(r, z);
   x+=z;
}

FilteredSolver::~FilteredSolver() { }


} // namespace mfem