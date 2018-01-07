// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

// Implementation of BilinearFormOperator

#include "fem.hpp"

namespace mfem
{

BilinearFormOperator::~BilinearFormOperator()
{
   if (trial_gs) delete X;
   if (test_gs) delete Y;
}

static void CheckIfValid(const FiniteElement *fe)
{
   const int geom = fe->GetGeomType();
   if ((geom == Geometry::Type::TRIANGLE) || (geom == Geometry::Type::TETRAHEDRON))
   {
      mfem_error("Simplex elements are not supported.");
   }
}

BilinearFormOperator::BilinearFormOperator(BilinearForm *bf)
{
   height = bf->Height();
   width  = bf->Width();

   Init(bf->FESpace(), bf->FESpace());

   // Does not currently support simplex elements
   CheckIfValid(bf->FESpace()->GetFE(0));

   // Add the integrators from bf->fesi
   Array<LinearFESpaceIntegrator*> &other_fesi = *(bf->GetFESI());
   for (int i = 0; i < other_fesi.Size(); i++)
   {
      lfesi.Append(other_fesi[i]);
   }

   Assemble();
}

BilinearFormOperator::BilinearFormOperator(MixedBilinearForm *mbf)
{
   height = mbf->Height();
   width  = mbf->Width();

   Init(mbf->TrialFESpace(), mbf->TestFESpace());

   // Does not currently support simplex elements
   CheckIfValid(mbf->TrialFESpace()->GetFE(0));

   // Add the integrators from mbf->fesi
   Array<LinearFESpaceIntegrator*> &other_fesi = *(mbf->GetFESI());
   for (int i = 0; i < other_fesi.Size(); i++)
   {
      lfesi.Append(other_fesi[i]);
   }

   Assemble();
}

void BilinearFormOperator::Assemble()
{
   // Linear assembly
   for (int i = 0; i < lfesi.Size(); i++)
   {
      lfesi[i]->Assemble(trial_fes, test_fes);
   }
}

void BilinearFormOperator::Init(FiniteElementSpace *_trial_fes,
                                FiniteElementSpace *_test_fes)
{
   trial_fes = _trial_fes;
   test_fes = _test_fes;

   trial_gs = true;
   X = NULL;
   if (dynamic_cast<const L2_FECollection *>(trial_fes->FEColl()))
   {
      trial_gs = false;
   }
   else
   {
      X = new Vector(trial_fes->GetLocalVSize());
   }

   test_gs = true;
   Y = NULL;
   if (dynamic_cast<const L2_FECollection *>(test_fes->FEColl()))
   {
      test_gs = false;
   }
   else
   {
      Y = new Vector(test_fes->GetLocalVSize());
   }
}

void BilinearFormOperator::DoMult(bool transpose, bool add,
                                  const Vector &x, Vector &y) const
{
   if (!transpose && trial_gs)
   {
      trial_fes->ToLocalVector(x, *X);
   }
   else if (transpose && test_gs)
   {
      test_fes->ToLocalVector(x, *X);
   }
   else
   {
      X = const_cast<Vector *>(&x);
   }

   if ((!transpose && !test_gs) || (transpose && !trial_gs)) { Y = &y; }
   *Y = 0.0;

   for (int i = 0; i < lfesi.Size(); i++) lfesi[i]->AddMult(*X, *Y);
   for (int i = 0; i < nlfesi.Size(); i++) nlfesi[i]->AddMult(*X, *Y);

   if (!transpose && test_gs)
   {
      if (add)
      {
         test_fes->ToGlobalVector(*Y, y_add);
         y += y_add;
      }
      else
      {
         test_fes->ToGlobalVector(*Y, y);
      }
   }
   else if (transpose && trial_gs)
   {
      if (add)
      {
         trial_fes->ToGlobalVector(*Y, y_add);
         y += y_add;
      }
      else
      {
         trial_fes->ToGlobalVector(*Y, y);
      }
   }
}

void BilinearFormOperator::AddMult(const Vector &x, Vector &y) const
{
   DoMult(false, true, x, y);
}

void BilinearFormOperator::AddMultTranspose(const Vector &x, Vector &y) const
{
   DoMult(true, true, x, y);
}

void BilinearFormOperator::Mult(const Vector &x, Vector &y) const
{
   DoMult(false, false, x, y);
}

void BilinearFormOperator::MultTranspose(const Vector &x, Vector &y) const
{
   DoMult(true, false, x, y);
}

}
