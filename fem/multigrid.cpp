// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "multigrid.hpp"

namespace mfem
{

MultigridBase::MultigridBase()
   : cycleType(CycleType::VCYCLE), preSmoothingSteps(1), postSmoothingSteps(1),
     nrhs(1)
{}

MultigridBase::MultigridBase(const Array<Operator*>& operators_,
                             const Array<Solver*>& smoothers_,
                             const Array<bool>& ownedOperators_,
                             const Array<bool>& ownedSmoothers_)
   : Solver(operators_.Last()->Height(), operators_.Last()->Width()),
     cycleType(CycleType::VCYCLE), preSmoothingSteps(1), postSmoothingSteps(1),
     nrhs(1)
{
   operators_.Copy(operators);
   smoothers_.Copy(smoothers);
   ownedOperators_.Copy(ownedOperators);
   ownedSmoothers_.Copy(ownedSmoothers);

   X.SetSize(operators.Size());
   Y.SetSize(operators.Size());
   R.SetSize(operators.Size());
   Z.SetSize(operators.Size());
   for (int i = 0; i < operators.Size(); ++i)
   {
      X[i] = new Array<Vector*>(nrhs);
      Y[i] = new Array<Vector*>(nrhs);
      R[i] = new Array<Vector*>(nrhs);
      Z[i] = new Array<Vector*>(nrhs);
      for (int c = 0; c < nrhs; ++c)
      {
         (*X[i])[c] = new Vector(operators[i]->Height());
         (*Y[i])[c] = new Vector(operators[i]->Height());
         (*R[i])[c] = new Vector(operators[i]->Height());
         (*Z[i])[c] = new Vector(operators[i]->Height());
      }
   }
}

MultigridBase::~MultigridBase()
{
   for (int i = 0; i < operators.Size(); ++i)
   {
      if (ownedOperators[i])
      {
         delete operators[i];
      }
      if (ownedSmoothers[i])
      {
         delete smoothers[i];
      }
      for (int c = 0; c < X[i]->Size(); ++c)
      {
         // nrhs <= X/Y/R/Z[i]->Size()
         delete (*X[i])[c];
         delete (*Y[i])[c];
         delete (*R[i])[c];
         delete (*Z[i])[c];
      }
      delete X[i];
      delete Y[i];
      delete R[i];
      delete Z[i];
   }
   operators.DeleteAll();
   smoothers.DeleteAll();
   X.DeleteAll();
   Y.DeleteAll();
   R.DeleteAll();
   Z.DeleteAll();
}

void MultigridBase::AddLevel(Operator* op, Solver* smoother,
                             bool ownOperator, bool ownSmoother)
{
   height = op->Height();
   width = op->Width();

   operators.Append(op);
   smoothers.Append(smoother);
   ownedOperators.Append(ownOperator);
   ownedSmoothers.Append(ownSmoother);

   X.Append(new Array<Vector*>(nrhs));
   Y.Append(new Array<Vector*>(nrhs));
   R.Append(new Array<Vector*>(nrhs));
   Z.Append(new Array<Vector*>(nrhs));
   for (int c = 0; c < nrhs; ++c)
   {
      (*X.Last())[c] = new Vector(height);
      (*Y.Last())[c] = new Vector(height);
      (*R.Last())[c] = new Vector(height);
      (*Z.Last())[c] = new Vector(height);
   }
}

void MultigridBase::SetCycleType(CycleType cycleType_, int preSmoothingSteps_,
                                 int postSmoothingSteps_)
{
   cycleType = cycleType_;
   preSmoothingSteps = preSmoothingSteps_;
   postSmoothingSteps = postSmoothingSteps_;
}

void MultigridBase::Mult(const Vector& x, Vector& y) const
{
   Array<Vector*> X_(1), Y_(1);
   X_[0] = const_cast<Vector*>(&x);
   Y_[0] = &y;
   Mult(X_, Y_);
}

void MultigridBase::Mult(const Array<Vector*>& X_, Array<Vector*>& Y_) const
{
   MFEM_ASSERT(operators.Size() > 0,
               "Multigrid solver does not have operators set!");
   MFEM_ASSERT(X_.Size() == Y_.Size(),
               "Number of columns mismatch in MultigridBase::Mult!");
   if (iterative_mode)
   {
      MFEM_WARNING("Multigrid solver does not use iterative_mode and ignores "
                   "the initial guess!");
   }
   // Add capacity as necessary
   if (X_.Size() > nrhs)
   {
      for (int i = 0; i < operators.Size(); ++i)
      {
         while (X.Size() > nrhs)
         {
            X[i]->Append(new Vector(operators[i]->Height()));
            Y[i]->Append(new Vector(operators[i]->Height()));
            R[i]->Append(new Vector(operators[i]->Height()));
            Z[i]->Append(new Vector(operators[i]->Height()));
            nrhs++;
         }
      }
   }
   else if (X_.Size() < nrhs)
   {
      nrhs = X_.Size();
   }

   // Perform a single cycle
   for (int c = 0; c < nrhs; ++c)
   {
      MFEM_ASSERT(X_[c] && Y_[c], "Missing Vector in MultigridBase::Mult!");
      *(*X.Last())[c] = *X_[c];
      *(*Y.Last())[c] = 0.0;
   }
   Cycle(GetFinestLevelIndex());
   for (int c = 0; c < nrhs; ++c)
   {
      *Y_[c] = *(*Y.Last())[c];
   }
}

void MultigridBase::UpdateResidual(int level) const
{
   // r = x - A y
   for (int c = 0; c < nrhs; ++c)
   {
      *(*R[level])[c] = *(*X[level])[c];
   }
   GetOperatorAtLevel(level)->AddMult(*Y[level], *R[level], -1.0);
}

void MultigridBase::SmoothingStep(int level) const
{
   // y = y + S (x - A y)
   UpdateResidual(level);
   GetSmootherAtLevel(level)->Mult(*R[level], *Z[level]);
   for (int c = 0; c < nrhs; ++c)
   {
      *(*Y[level])[c] += *(*Z[level])[c];
   }
}

void MultigridBase::PostSmoothingStep(int level) const
{
   // y = y + S^T (x - A y)
   UpdateResidual(level);
   GetSmootherAtLevel(level)->MultTranspose(*R[level], *Z[level]);
   for (int c = 0; c < nrhs; ++c)
   {
      *(*Y[level])[c] += *(*Z[level])[c];
   }
}

void MultigridBase::Cycle(int level) const
{
   // Pre-smooth or coarse solve (Y at the level is always zero)
   GetSmootherAtLevel(level)->Mult(*X[level], *Y[level]);
   if (level == 0) { return; }

   // Additional pre-smooth
   for (int i = 1; i < preSmoothingSteps; i++)
   {
      SmoothingStep(level);
   }

   // Compute residual and restrict
   UpdateResidual(level);
   GetProlongationAtLevel(level - 1)->MultTranspose(*R[level], *X[level - 1]);

   // Initialize zeros
   for (int c = 0; c < nrhs; ++c)
   {
      *(*Y[level - 1])[c] = 0.0;
   }

   // Corrections
   int corrections = 1;
   if (cycleType == CycleType::WCYCLE)
   {
      corrections = 2;
   }
   for (int correction = 0; correction < corrections; ++correction)
   {
      Cycle(level - 1);
   }

   // Prolongate and add
   GetProlongationAtLevel(level - 1)->AddMult(*Y[level - 1], *Y[level], 1.0);

   // Post-smooth
   for (int i = 0; i < postSmoothingSteps; i++)
   {
      PostSmoothingStep(level);
   }
}

Multigrid::Multigrid()
   : MultigridBase()
{}

Multigrid::Multigrid(const Array<Operator*>& operators_,
                     const Array<Solver*>& smoothers_,
                     const Array<Operator*>& prolongations_,
                     const Array<bool>& ownedOperators_,
                     const Array<bool>& ownedSmoothers_,
                     const Array<bool>& ownedProlongations_)
   : MultigridBase(operators_, smoothers_, ownedOperators_, ownedSmoothers_)
{
   prolongations_.Copy(prolongations);
   ownedProlongations_.Copy(ownedProlongations);
}

Multigrid::~Multigrid()
{
   for (int i = 0; i < prolongations.Size(); ++i)
   {
      if (ownedProlongations[i])
      {
         delete prolongations[i];
      }
   }
   prolongations.DeleteAll();
}

GeometricMultigrid::GeometricMultigrid(const FiniteElementSpaceHierarchy& fespaces_)
   : MultigridBase(), fespaces(fespaces_)
{}

GeometricMultigrid::~GeometricMultigrid()
{
   for (int i = 0; i < bfs.Size(); ++i)
   {
      delete bfs[i];
   }
   bfs.DeleteAll();
   for (int i = 0; i < essentialTrueDofs.Size(); ++i)
   {
      delete essentialTrueDofs[i];
   }
   essentialTrueDofs.DeleteAll();
}

void GeometricMultigrid::FormFineLinearSystem(Vector& x, Vector& b,
                                              OperatorHandle& A,
                                              Vector& X, Vector& B)
{
   bfs.Last()->FormLinearSystem(*essentialTrueDofs.Last(), x, b, A, X, B);
}

void GeometricMultigrid::RecoverFineFEMSolution(const Vector& X,
                                                const Vector& b, Vector& x)
{
   bfs.Last()->RecoverFEMSolution(X, b, x);
}

} // namespace mfem
