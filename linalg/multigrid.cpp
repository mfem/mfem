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

#include "multigrid.hpp"

namespace mfem
{

MultigridOperator::MultigridOperator() {}

MultigridOperator::MultigridOperator(Operator* opr, Solver* coarseSolver,
                                     bool ownOperator, bool ownSolver)
{
   AddCoarsestLevel(opr, coarseSolver, ownOperator, ownSolver);
}

MultigridOperator::~MultigridOperator()
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

   for (int i = 0; i < prolongations.Size(); ++i)
   {
      if (ownedProlongations[i])
      {
         delete prolongations[i];
      }
   }

   operators.DeleteAll();
   smoothers.DeleteAll();
}

void MultigridOperator::AddCoarsestLevel(Operator* opr, Solver* solver,
                                         bool ownOperator, bool ownSolver)
{
   MFEM_VERIFY(NumLevels() == 0, "Coarse level already exists");
   operators.Append(opr);
   smoothers.Append(solver);
   ownedOperators.Append(ownOperator);
   ownedSmoothers.Append(ownSolver);
   width = opr->Width();
   height = opr->Height();
}

void MultigridOperator::AddLevel(Operator* opr, Solver* smoother,
                                 const Operator* prolongation, bool ownOperator,
                                 bool ownSmoother, bool ownProlongation)
{
   MFEM_VERIFY(NumLevels() > 0, "Please add a coarse level first");
   operators.Append(opr);
   smoothers.Append(smoother);
   prolongations.Append(prolongation);
   ownedOperators.Append(ownOperator);
   ownedSmoothers.Append(ownSmoother);
   ownedProlongations.Append(ownProlongation);
   width = opr->Width();
   height = opr->Height();
}

int MultigridOperator::NumLevels() const { return operators.Size(); }

int MultigridOperator::GetFinestLevelIndex() const { return NumLevels() - 1; }

void MultigridOperator::MultAtLevel(int level, const Vector& x, Vector& y) const
{
   MFEM_ASSERT(level < NumLevels(), "Level does not exist.");
   operators[level]->Mult(x, y);
}

/// Matrix vector multiplication on finest level
void MultigridOperator::Mult(const Vector& x, Vector& y) const
{
   MFEM_ASSERT(NumLevels() > 0, "At least one level needs to exist.");
   MultAtLevel(NumLevels() - 1, x, y);
}

void MultigridOperator::RestrictTo(int level, const Vector& x, Vector& y) const
{
   prolongations[level]->MultTranspose(x, y);
}

void MultigridOperator::InterpolateFrom(int level, const Vector& x,
                                        Vector& y) const
{
   prolongations[level]->Mult(x, y);
}

void MultigridOperator::ApplySmootherAtLevel(int level, const Vector& x,
                                             Vector& y) const
{
   smoothers[level]->Mult(x, y);
}

const Operator* MultigridOperator::GetOperatorAtLevel(int level) const
{
   return operators[level];
}

Operator* MultigridOperator::GetOperatorAtLevel(int level)
{
   return operators[level];
}

const Operator* MultigridOperator::GetOperatorAtFinestLevel() const
{
   return GetOperatorAtLevel(operators.Size() - 1);
}

Operator* MultigridOperator::GetOperatorAtFinestLevel()
{
   return GetOperatorAtLevel(operators.Size() - 1);
}

Solver* MultigridOperator::GetSmootherAtLevel(int level) const
{
   return smoothers[level];
}

Solver* MultigridOperator::GetSmootherAtLevel(int level)
{
   return smoothers[level];
}

MultigridSolver::MultigridSolver(const MultigridOperator* opr_,
                                 CycleType cycleType_, int preSmoothingSteps_,
                                 int postSmoothingSteps_)
   : opr(opr_), cycleType(cycleType_)
{
   Setup(preSmoothingSteps_, postSmoothingSteps_);
}

MultigridSolver::~MultigridSolver() { Reset(); }

void MultigridSolver::SetCycleType(CycleType cycleType_)
{
   cycleType = cycleType_;
}

void MultigridSolver::SetPreSmoothingSteps(int steps)
{
   preSmoothingSteps = steps;
}

void MultigridSolver::SetPreSmoothingSteps(const Array<int>& steps)
{
   MFEM_VERIFY(
      steps.Size() == preSmoothingSteps.Size(),
      "Number of step sizes needs to be the same as the number of levels");
   preSmoothingSteps = steps;
}

void MultigridSolver::SetPostSmoothingSteps(int steps)
{
   postSmoothingSteps = steps;
}

void MultigridSolver::SetPostSmoothingSteps(const Array<int>& steps)
{
   MFEM_VERIFY(
      steps.Size() == postSmoothingSteps.Size(),
      "Number of step sizes needs to be the same as the number of levels");
   postSmoothingSteps = steps;
}

void MultigridSolver::SetSmoothingSteps(int steps)
{
   SetPreSmoothingSteps(steps);
   SetPostSmoothingSteps(steps);
}

void MultigridSolver::SetSmoothingSteps(const Array<int>& steps)
{
   SetPreSmoothingSteps(steps);
   SetPostSmoothingSteps(steps);
}

void MultigridSolver::Mult(const Vector& x, Vector& y) const
{
   // Safe const_cast, since x at the finest level will never be modified
   X.Last() = const_cast<Vector*>(&x);
   y = 0.0;
   Y.Last() = &y;
   Cycle(opr->NumLevels() - 1);
   X.Last() = nullptr;
   Y.Last() = nullptr;
}

void MultigridSolver::SetOperator(const Operator& op)
{
   if (!dynamic_cast<const MultigridOperator*>(&op))
   {
      MFEM_ABORT("Unsupported operator for MultigridSolver");
   }

   Reset();
   opr = static_cast<const MultigridOperator*>(&op);
   Setup();
}

void MultigridSolver::SmoothingStep(int level) const
{
   opr->MultAtLevel(level, *Y[level], *R[level]);          // r = A x
   subtract(*X[level], *R[level], *R[level]);              // r = b - A x
   opr->ApplySmootherAtLevel(level, *R[level], *Z[level]); // z = S r
   add(*Y[level], 1.0, *Z[level], *Y[level]); // x = x + S (b - A x)
}

void MultigridSolver::Cycle(int level) const
{
   if (level == 0)
   {
      opr->ApplySmootherAtLevel(level, *X[level], *Y[level]);
      return;
   }

   for (int i = 0; i < preSmoothingSteps[level]; i++)
   {
      SmoothingStep(level);
   }

   // Compute residual
   opr->MultAtLevel(level, *Y[level], *R[level]);
   subtract(*X[level], *R[level], *R[level]);

   // Restrict residual
   opr->RestrictTo(level - 1, *R[level], *X[level - 1]);

   // Init zeros
   *Y[level - 1] = 0.0;

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

   // Prolongate
   opr->InterpolateFrom(level - 1, *Y[level - 1], *R[level]);

   // Add update
   *Y[level] += *R[level];

   // Post-smooth
   for (int i = 0; i < postSmoothingSteps[level]; i++)
   {
      SmoothingStep(level);
   }
}

void MultigridSolver::Setup(int preSmoothingSteps_, int postSmoothingSteps_)
{
   for (int level = 0; level < opr->NumLevels() - 1; ++level)
   {
      int vectorSize = opr->GetOperatorAtLevel(level)->Height();
      X.Append(new Vector(vectorSize));
      *X.Last() = 0.0;
      Y.Append(new Vector(vectorSize));
      *Y.Last() = 0.0;
      R.Append(new Vector(vectorSize));
      *R.Last() = 0.0;
      Z.Append(new Vector(vectorSize));
      *Z.Last() = 0.0;
   }

   // X and Y at the finest level will be filled by Mult
   X.Append(nullptr);
   Y.Append(nullptr);
   R.Append(new Vector(opr->GetOperatorAtFinestLevel()->Height()));
   *R.Last() = 0.0;
   Z.Append(new Vector(opr->GetOperatorAtFinestLevel()->Height()));
   *Z.Last() = 0.0;

   preSmoothingSteps.SetSize(opr->NumLevels());
   postSmoothingSteps.SetSize(opr->NumLevels());

   preSmoothingSteps = preSmoothingSteps_;
   postSmoothingSteps = postSmoothingSteps_;
}

void MultigridSolver::Reset()
{
   for (int i = 0; i < X.Size(); ++i)
   {
      delete X[i];
      delete Y[i];
      delete R[i];
      delete Z[i];
   }

   X.DeleteAll();
   Y.DeleteAll();
   R.DeleteAll();
   Z.DeleteAll();

   preSmoothingSteps.DeleteAll();
   postSmoothingSteps.DeleteAll();
}

} // namespace mfem
