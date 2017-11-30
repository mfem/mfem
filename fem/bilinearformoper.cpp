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

static void BuildDofMaps(FiniteElementSpace *fespace, Array<int> *&off,
                         Array<int> *&ind)
{
   // Get the total size without vdim
   int size = 0;
   const int vdim = fespace->GetVDim();
   for (int e = 0; e < fespace->GetNE(); e++)
   {
      const FiniteElement *fe = fespace->GetFE(e);
      size += fe->GetDof();
   }
   const int local_size = size * vdim;
   const int global_size = fespace->GetVSize();

   // Now we can allocate and fill the global map
   off = new Array<int>(global_size + 1);
   ind = new Array<int>(local_size);

   Array<int> &offsets = *off;
   Array<int> &indices = *ind;

   Array<int> global_map(local_size);
   Array<int> elem_vdof;

   int offset = 0;
   for (int e = 0; e < fespace->GetNE(); e++)
   {
      const FiniteElement *fe = fespace->GetFE(e);
      const int dofs = fe->GetDof();
      const int vdofs = dofs * vdim;
      const TensorBasisElement *tfe = dynamic_cast<const TensorBasisElement *>(fe);
      const Array<int> &dof_map = tfe->GetDofMap();

      fespace->GetElementVDofs(e, elem_vdof);

      for (int vd = 0; vd < vdim; vd++)
         for (int i = 0; i < vdofs; i++)
         {
            global_map[offset + dofs*vd + i] = elem_vdof[dofs*vd + dof_map[i]];
         }
      offset += vdofs;
   }

   // Store and use a set of offsets and indices instead of this map

   // Zero the offset vector
   offsets = 0;

   // Keep track of how many local dof point to its global dof
   // Count how many times each dof gets hit
   for (int i = 0; i < local_size; i++)
   {
      const int g = global_map[i];
      ++offsets[g + 1];
   }
   // Aggregate the offsets
   for (int i = 1; i <= global_size; i++)
   {
      offsets[i] += offsets[i - 1];
   }

   for (int i = 0; i < local_size; i++)
   {
      const int g = global_map[i];
      indices[offsets[g]++] = i;
   }

   // Shift the offset vector back by one, since it was used as a
   // counter above.
   for (int i = global_size; i > 0; i--)
   {
      offsets[i] = offsets[i - 1];
   }
   offsets[0] = 0;
}

BilinearFormOperator::BilinearFormOperator(IntegratorMap *_map)
   : bf(NULL), mbf(NULL),
     trial_fes(NULL), test_fes(NULL),
     trial_gs(false), test_gs(false),
     map(_map) { }

BilinearFormOperator::~BilinearFormOperator()
{
   delete map;
   Clear();
}

void BilinearFormOperator::Assemble(BilinearForm *_bf)
{
   if (_bf != bf)
   {
      bf = _bf;
      height = bf->Height();
      width = bf->Width();

      Init(bf->FESpace(), NULL);

      // Delete the old integrator list -- Note that this does not
      // delete the integrators themselves (since the original
      // bilinear form owns these)
      lfesi.DeleteAll();

      // Add the integrators from bf->fesi
      Array<LinearFESpaceIntegrator*> &other_fesi = *(bf->GetFESI());
      for (int i = 0; i < other_fesi.Size(); i++)
      {
         lfesi.Append(other_fesi[i]);
      }

      if (map)
      {
         Array<BilinearFormIntegrator*> &dbfi = *(bf->GetDBFI());
         for (int i = 0; i < dbfi.Size(); i++)
         {
            lfesi.Append(map->DomainIntegrator(dbfi[i]));
         }
      }
   }

   Assemble();
}

void BilinearFormOperator::Assemble(MixedBilinearForm *_mbf)
{
   if (_mbf != mbf)
   {
      mbf = _mbf;
      height = bf->Height();
      width = bf->Width();

      Init(mbf->TrialFESpace(), mbf->TestFESpace());

      // Delete the old integrator list -- Note that this does not
      // delete the integrators themselves (since the original mixed
      // bilinear form owns these)
      lfesi.DeleteAll();

      // Add the integrators from mbf->fesi
      Array<LinearFESpaceIntegrator*> &other_fesi = *(mbf->GetFESI());
      for (int i = 0; i < other_fesi.Size(); i++)
      {
         lfesi.Append(other_fesi[i]);
      }

      if (map)
      {
         Array<BilinearFormIntegrator*> &dbfi = *(mbf->GetDBFI());
         for (int i = 0; i < dbfi.Size(); i++)
         {
            lfesi.Append(map->DomainIntegrator(dbfi[i]));
         }
      }
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

void BilinearFormOperator::Clear()
{
   delete trial_offsets;
   delete trial_indices;

   if (test_fes)
   {
      delete test_offsets;
      delete test_indices;
   }

   delete X;
   delete Y;
}

void BilinearFormOperator::Init(FiniteElementSpace *_trial_fes,
                           FiniteElementSpace *_test_fes)
{
   if ((_trial_fes != trial_fes) || (_test_fes != test_fes))
   {
      // Clear before recreating
      Clear();

      trial_fes = _trial_fes;
      test_fes = _test_fes;
      BuildDofMaps(trial_fes, trial_offsets, trial_indices);

      if (test_fes != NULL)
      {
         BuildDofMaps(test_fes, test_offsets, test_indices);
      }
      else
      {
         // Point to the trial offsets and indices
         test_offsets = trial_offsets;
         test_indices = trial_indices;
      }

      X = new Vector(trial_indices->Size());
      Y = new Vector(test_indices->Size());
   }

   const FiniteElementSpace *actual_test_fes =
      (test_fes != NULL) ? test_fes : trial_fes;
   trial_gs = test_gs = true;
   if (dynamic_cast<const L2_FECollection *>(trial_fes->FEColl()))
   {
      trial_gs = test_gs = false;
   }
   else if (dynamic_cast<const L2_FECollection *>(actual_test_fes->FEColl()))
   {
      test_gs = false;
   }
}


void BilinearFormOperator::LToEVector(const Array<int> &offsets,
                                      const Array<int> &indices,
                                      const Vector &v, Vector &V) const
{
   const int size = v.Size();
   for (int i = 0; i < size; i++)
   {
      const int offset = offsets[i];
      const int next_offset = offsets[i + 1];
      const double dof_value = v(i);
      for (int j = offset; j < next_offset; j++) { V(indices[j]) = dof_value; }
   }
}

void BilinearFormOperator::EToLVector(const Array<int> &offsets,
                                      const Array<int> &indices,
                                      const Vector &V, Vector &v) const
{
   // NOTE: This method ADDS to the output v
   const int size = v.Size();
   for (int i = 0; i < size; i++)
   {
      const int offset = offsets[i];
      const int next_offset = offsets[i + 1];
      double dof_value = 0;
      for (int j = offset; j < next_offset; j++) { dof_value += V(indices[j]); }
      v(i) += dof_value;
   }
}

void BilinearFormOperator::AddMult(const Vector &x, Vector &y) const
{
   if (trial_gs) { LToEVector(*trial_offsets, *trial_indices, x, *X); }
   else { X = const_cast<Vector *>(&x); }

   if (!test_gs) { Y = &y; }

   *Y = 0.0;
   for (int i = 0; i < lfesi.Size(); i++) lfesi[i]->AddMult(*X, *Y);
   for (int i = 0; i < nlfesi.Size(); i++) nlfesi[i]->AddMult(*X, *Y);

   if (test_gs) { EToLVector(*test_offsets, *test_indices, *Y, y); }
}


void BilinearFormOperator::AddMultTranspose(const Vector &x, Vector &y) const
{
   if (test_gs) { LToEVector(*test_offsets, *test_indices, x, *X); }
   else { X = const_cast<Vector *>(&x); }

   if (!trial_gs) { Y = &y; }

   *Y = 0.0;
   for (int i = 0; i < lfesi.Size(); i++) lfesi[i]->AddMultTranspose(*X, *Y);
   for (int i = 0; i < nlfesi.Size(); i++) nlfesi[i]->AddMultTranspose(*X, *Y);

   if (trial_gs) { EToLVector(*trial_offsets, *trial_indices, *Y, y); }
}

void BilinearFormOperator::Mult(const Vector &x, Vector &y) const
{ y = 0.0; AddMult(x, y); }

}
