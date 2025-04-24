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

#include "multigrid.hpp"

namespace mfem
{

MultigridBase::MultigridBase()
   : cycleType(CycleType::VCYCLE), preSmoothingSteps(1), postSmoothingSteps(1),
     nrhs(0)
{}

MultigridBase::MultigridBase(const Array<Operator*>& operators_,
                             const Array<Solver*>& smoothers_,
                             const Array<bool>& ownedOperators_,
                             const Array<bool>& ownedSmoothers_)
   : Solver(operators_.Last()->Height(), operators_.Last()->Width()),
     cycleType(CycleType::VCYCLE), preSmoothingSteps(1), postSmoothingSteps(1),
     nrhs(0)
{
   operators_.Copy(operators);
   smoothers_.Copy(smoothers);
   ownedOperators_.Copy(ownedOperators);
   ownedSmoothers_.Copy(ownedSmoothers);
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
   }
   EraseVectors();
}

void MultigridBase::InitVectors() const
{
   if (X.NumRows() > 0 && X.NumCols() > 0) { EraseVectors(); }
   const int M = NumLevels();
   X.SetSize(M, nrhs);
   Y.SetSize(M, nrhs);
   R.SetSize(M, nrhs);
   Z.SetSize(M, nrhs);
   for (int i = 0; i < X.NumRows(); ++i)
   {
      const int n = operators[i]->Height();
      for (int j = 0; j < X.NumCols(); ++j)
      {
         X(i, j) = new Vector(n);
         Y(i, j) = new Vector(n);
         R(i, j) = new Vector(n);
         Z(i, j) = new Vector(n);
      }
   }
}

void MultigridBase::EraseVectors() const
{
   for (int i = 0; i < X.NumRows(); ++i)
   {
      for (int j = 0; j < X.NumCols(); ++j)
      {
         delete X(i, j);
         delete Y(i, j);
         delete R(i, j);
         delete Z(i, j);
      }
   }
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
   Array<const Vector*> X_(1);
   Array<Vector*> Y_(1);
   X_[0] = &x;
   Y_[0] = &y;
   ArrayMult(X_, Y_);
}

void MultigridBase::ArrayMult(const Array<const Vector*>& X_,
                              Array<Vector*>& Y_) const
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
   nrhs = X_.Size();
   if (X.NumCols() < nrhs) { InitVectors(); }

   // Perform a single cycle
   const int M = NumLevels();
   for (int j = 0; j < nrhs; ++j)
   {
      MFEM_ASSERT(X_[j] && Y_[j], "Missing Vector in MultigridBase::Mult!");
      *X(M - 1, j) = *X_[j];
      *Y(M - 1, j) = 0.0;
   }
   Cycle(M - 1);
   for (int j = 0; j < nrhs; ++j)
   {
      *Y_[j] = *Y(M - 1, j);
   }
}

void MultigridBase::SmoothingStep(int level, bool zero, bool transpose) const
{
   // y = y + S (x - A y) or y = y + S^T (x - A y)
   if (zero)
   {
      Array<Vector *> X_(X[level], nrhs), Y_(Y[level], nrhs);
      GetSmootherAtLevel(level)->ArrayMult(X_, Y_);
   }
   else
   {
      Array<Vector *> Y_(Y[level], nrhs), R_(R[level], nrhs),
            Z_(Z[level], nrhs);
      for (int j = 0; j < nrhs; ++j)
      {
         *R_[j] = *X(level, j);
      }
      GetOperatorAtLevel(level)->ArrayAddMult(Y_, R_, -1.0);
      if (transpose)
      {
         GetSmootherAtLevel(level)->ArrayMultTranspose(R_, Z_);
      }
      else
      {
         GetSmootherAtLevel(level)->ArrayMult(R_, Z_);
      }
      for (int j = 0; j < nrhs; ++j)
      {
         *Y_[j] += *Z_[j];
      }
   }
}

void MultigridBase::Cycle(int level) const
{
   // Coarse solve
   if (level == 0)
   {
      SmoothingStep(0, true, false);
      return;
   }

   // Pre-smooth
   for (int i = 0; i < preSmoothingSteps; ++i)
   {
      SmoothingStep(level, (cycleType == CycleType::VCYCLE && i == 0), false);
   }

   // Compute residual and restrict
   {
      Array<Vector *> Y_(Y[level], nrhs), R_(R[level], nrhs),
            X_(X[level - 1], nrhs);
      for (int j = 0; j < nrhs; ++j)
      {
         *R_[j] = *X(level, j);
      }
      GetOperatorAtLevel(level)->ArrayAddMult(Y_, R_, -1.0);
      GetProlongationAtLevel(level - 1)->ArrayMultTranspose(R_, X_);
      for (int j = 0; j < nrhs; ++j)
      {
         *Y(level - 1, j) = 0.0;
      }
   }

   // Corrections
   Cycle(level - 1);
   if (cycleType == CycleType::WCYCLE)
   {
      Cycle(level - 1);
   }

   // Prolongate and add
   {
      Array<Vector *> Y_(Y[level - 1], nrhs), Z_(Z[level], nrhs);
      GetProlongationAtLevel(level - 1)->ArrayMult(Y_, Z_);
      for (int j = 0; j < nrhs; ++j)
      {
         *Y(level, j) += *Z_[j];
      }
   }

   // Post-smooth
   for (int i = 0; i < postSmoothingSteps; ++i)
   {
      SmoothingStep(level, false, true);
   }
}

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
}

GeometricMultigrid::GeometricMultigrid(
   const FiniteElementSpaceHierarchy& fespaces_)
   : fespaces(fespaces_)
{
   const int nlevels = fespaces.GetNumLevels();
   ownedProlongations.SetSize(nlevels - 1);
   ownedProlongations = false;

   prolongations.SetSize(nlevels - 1);
   for (int level = 0; level < nlevels - 1; ++level)
   {
      prolongations[level] = fespaces.GetProlongationAtLevel(level);
   }
}

GeometricMultigrid::GeometricMultigrid(
   const FiniteElementSpaceHierarchy& fespaces_,
   const Array<int> &ess_bdr)
   : fespaces(fespaces_)
{
   bool have_ess_bdr = false;
   for (int i = 0; i < ess_bdr.Size(); i++)
   {
      if (ess_bdr[i]) { have_ess_bdr = true; break; }
   }

   const int nlevels = fespaces.GetNumLevels();
   ownedProlongations.SetSize(nlevels - 1);
   ownedProlongations = have_ess_bdr;

   if (have_ess_bdr)
   {
      essentialTrueDofs.SetSize(nlevels);
      for (int level = 0; level < nlevels; ++level)
      {
         essentialTrueDofs[level] = new Array<int>;
         fespaces.GetFESpaceAtLevel(level).GetEssentialTrueDofs(
            ess_bdr, *essentialTrueDofs[level]);
      }
   }

   prolongations.SetSize(nlevels - 1);
   for (int level = 0; level < nlevels - 1; ++level)
   {
      if (have_ess_bdr)
      {
         prolongations[level] = new RectangularConstrainedOperator(
            fespaces.GetProlongationAtLevel(level),
            *essentialTrueDofs[level],
            *essentialTrueDofs[level + 1]
         );
      }
      else
      {
         prolongations[level] = fespaces.GetProlongationAtLevel(level);
      }
   }
}

GeometricMultigrid::~GeometricMultigrid()
{
   for (int i = 0; i < bfs.Size(); ++i)
   {
      delete bfs[i];
   }
   for (int i = 0; i < essentialTrueDofs.Size(); ++i)
   {
      delete essentialTrueDofs[i];
   }
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
