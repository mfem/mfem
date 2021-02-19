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

#include "multigrid.hpp"

namespace mfem
{

Multigrid::Multigrid()
   : cycleType(CycleType::VCYCLE), preSmoothingSteps(1), postSmoothingSteps(1)
{}

Multigrid::Multigrid(const Array<Operator*>& operators_,
                     const Array<Solver*>& smoothers_,
                     const Array<Operator*>& prolongations_,
                     const Array<bool>& ownedOperators_,
                     const Array<bool>& ownedSmoothers_,
                     const Array<bool>& ownedProlongations_)
   : Solver(operators_.Last()->NumRows()), cycleType(CycleType::VCYCLE),
     preSmoothingSteps(1), postSmoothingSteps(1),
     X(operators_.Size()), Y(X.Size()), R(X.Size()), Z(X.Size())
{
   operators_.Copy(operators);
   smoothers_.Copy(smoothers);
   prolongations_.Copy(prolongations);
   ownedOperators_.Copy(ownedOperators);
   ownedSmoothers_.Copy(ownedSmoothers);
   ownedProlongations_.Copy(ownedProlongations);

   for (int level = 0; level < operators.Size(); ++level)
   {
      X[level] = new Vector(operators[level]->NumRows());
      *X[level] = 0.0;
      Y[level] = new Vector(operators[level]->NumRows());
      *Y[level] = 0.0;
      R[level] = new Vector(operators[level]->NumRows());
      *R[level] = 0.0;
      Z[level] = new Vector(operators[level]->NumRows());
      *Z[level] = 0.0;
   }
}

Multigrid::~Multigrid()
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
      delete X[i];
      delete Y[i];
      delete R[i];
      delete Z[i];
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
   prolongations.DeleteAll();
   X.DeleteAll();
   Y.DeleteAll();
   R.DeleteAll();
   Z.DeleteAll();
}

void Multigrid::AddLevel(Operator* opr, Solver* smoother, bool ownOperator,
                         bool ownSmoother)
{
   operators.Append(opr);
   smoothers.Append(smoother);
   ownedOperators.Append(ownOperator);
   ownedSmoothers.Append(ownSmoother);
   width = opr->Width();
   height = opr->Height();

   X.Append(new Vector(height));
   *X.Last() = 0.0;
   Y.Append(new Vector(height));
   *Y.Last() = 0.0;
   R.Append(new Vector(height));
   *R.Last() = 0.0;
   Z.Append(new Vector(height));
   *Z.Last() = 0.0;
}

int Multigrid::NumLevels() const { return operators.Size(); }

int Multigrid::GetFinestLevelIndex() const { return NumLevels() - 1; }

const Operator* Multigrid::GetOperatorAtLevel(int level) const
{
   return operators[level];
}

Operator* Multigrid::GetOperatorAtLevel(int level)
{
   return operators[level];
}

const Operator* Multigrid::GetOperatorAtFinestLevel() const
{
   return GetOperatorAtLevel(operators.Size() - 1);
}

Operator* Multigrid::GetOperatorAtFinestLevel()
{
   return GetOperatorAtLevel(operators.Size() - 1);
}

Solver* Multigrid::GetSmootherAtLevel(int level) const
{
   return smoothers[level];
}

Solver* Multigrid::GetSmootherAtLevel(int level)
{
   return smoothers[level];
}

void Multigrid::SetCycleType(CycleType cycleType_, int preSmoothingSteps_,
                             int postSmoothingSteps_)
{
   cycleType = cycleType_;
   preSmoothingSteps = preSmoothingSteps_;
   postSmoothingSteps = postSmoothingSteps_;
}

void Multigrid::Mult(const Vector& x, Vector& y) const
{
   MFEM_ASSERT(NumLevels() > 0, "");
   *X.Last() = x;
   *Y.Last() = 0.0;
   Cycle(GetFinestLevelIndex());
   y = *Y.Last();
}

void Multigrid::SetOperator(const Operator& op)
{
   MFEM_ABORT("SetOperator not supported in Multigrid");
}

void Multigrid::SmoothingStep(int level, bool transpose) const
{
   GetOperatorAtLevel(level)->Mult(*Y[level], *R[level]); // r = A x
   subtract(*X[level], *R[level], *R[level]);             // r = b - A x
   if (transpose)
   {
      GetSmootherAtLevel(level)->MultTranspose(*R[level], *Z[level]); // z = S r
   }
   else
   {
      GetSmootherAtLevel(level)->Mult(*R[level], *Z[level]); // z = S r
   }
   add(*Y[level], 1.0, *Z[level], *Y[level]);             // x = x + S (b - A x)
}

void Multigrid::Cycle(int level) const
{
   if (level == 0)
   {
      GetSmootherAtLevel(level)->Mult(*X[level], *Y[level]);
      return;
   }

   for (int i = 0; i < preSmoothingSteps; i++)
   {
      SmoothingStep(level, false);
   }

   // Compute residual
   GetOperatorAtLevel(level)->Mult(*Y[level], *R[level]);
   subtract(*X[level], *R[level], *R[level]);

   // Restrict residual
   GetProlongationAtLevel(level - 1)->MultTranspose(*R[level], *X[level - 1]);

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
   GetProlongationAtLevel(level - 1)->Mult(*Y[level - 1], *R[level]);

   // Add update
   *Y[level] += *R[level];

   // Post-smooth
   for (int i = 0; i < postSmoothingSteps; i++)
   {
      SmoothingStep(level, true);
   }
}

const Operator* Multigrid::GetProlongationAtLevel(int level) const
{
   return prolongations[level];
}

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

const Operator* GeometricMultigrid::GetProlongationAtLevel(int level) const
{
   return fespaces.GetProlongationAtLevel(level);
}

} // namespace mfem
