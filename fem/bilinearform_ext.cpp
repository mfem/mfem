// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// Implementations of classes FABilinearFormExtension, EABilinearFormExtension,
// PABilinearFormExtension and MFBilinearFormExtension.

#include "../general/forall.hpp"
#include "bilinearform.hpp"
#include "pbilinearform.hpp"
#include "pgridfunc.hpp"
#include "ceed/interface/util.hpp"

namespace mfem
{

/// Base class for extensions to the BilinearForm class
BilinearFormExtension::BilinearFormExtension(BilinearForm *form)
   : Operator(form->Size()), a(form)
{
}

const Operator *BilinearFormExtension::GetProlongation() const
{
   return a->GetProlongation();
}

const Operator *BilinearFormExtension::GetRestriction() const
{
   return a->GetRestriction();
}

/// Data and methods for matrix-free bilinear forms
MFBilinearFormExtension::MFBilinearFormExtension(BilinearForm *form)
   : BilinearFormExtension(form)
{
   Update();
}

void MFBilinearFormExtension::SetupRestrictionOperators(const L2FaceValues m)
{
   if (DeviceCanUseCeed()) { return; }
   ElementDofOrdering ordering = UsesTensorBasis(*fes) ?
                                 ElementDofOrdering::LEXICOGRAPHIC :
                                 ElementDofOrdering::NATIVE;
   elem_restrict = fes->GetElementRestriction(ordering);
   if (elem_restrict)
   {
      local_x.SetSize(elem_restrict->Height(), Device::GetDeviceMemoryType());
      local_y.SetSize(elem_restrict->Height(), Device::GetDeviceMemoryType());
      local_y.UseDevice(true); // ensure 'local_y = 0.0' is done on device
   }

   // Construct face restriction operators only if the bilinear form has
   // interior or boundary face integrators
   if (int_face_restrict_lex == nullptr && a->GetFBFI()->Size() > 0)
   {
      int_face_restrict_lex = fes->GetFaceRestriction(
                                 ElementDofOrdering::LEXICOGRAPHIC,
                                 FaceType::Interior);
      int_face_x.SetSize(int_face_restrict_lex->Height(),
                         Device::GetDeviceMemoryType());
      int_face_y.SetSize(int_face_restrict_lex->Height(),
                         Device::GetDeviceMemoryType());
      int_face_y.UseDevice(true);
   }

   if (bdr_face_restrict_lex == nullptr &&
       (a->GetBFBFI()->Size() > 0 || a->GetBBFI()->Size() > 0))
   {
      bdr_face_restrict_lex = fes->GetFaceRestriction(
                                 ElementDofOrdering::LEXICOGRAPHIC,
                                 FaceType::Boundary,
                                 m);
      bdr_face_x.SetSize(bdr_face_restrict_lex->Height(),
                         Device::GetDeviceMemoryType());
      bdr_face_y.SetSize(bdr_face_restrict_lex->Height(),
                         Device::GetDeviceMemoryType());
      bdr_face_y.UseDevice(true);
   }
}

void MFBilinearFormExtension::Assemble()
{
   SetupRestrictionOperators(L2FaceValues::DoubleValued);

   Array<BilinearFormIntegrator *> &integrators = *a->GetDBFI();
   for (BilinearFormIntegrator *integ : integrators)
   {
      integ->AssembleMF(*fes);
   }

   Array<BilinearFormIntegrator *> &bdr_integrators = *a->GetBBFI();
   for (BilinearFormIntegrator *integ : bdr_integrators)
   {
      integ->AssembleMFBoundary(*fes);
   }

   MFEM_VERIFY(a->GetFBFI()->Size() == 0, "AddInteriorFaceIntegrator is not "
               "currently supported in MFBilinearFormExtension");

   MFEM_VERIFY(a->GetBFBFI()->Size() == 0, "AddBdrFaceIntegrator is not "
               "currently supported in MFBilinearFormExtension");
}

void MFBilinearFormExtension::AssembleDiagonal(Vector &diag) const
{
   Array<BilinearFormIntegrator *> &integrators = *a->GetDBFI();
   if (integrators.Size() > 0 && elem_restrict)
   {
      local_y = 0.0;
      for (BilinearFormIntegrator *integ : integrators)
      {
         integ->AssembleDiagonalMF(local_y);
      }
      elem_restrict->MultTransposeUnsigned(local_y, diag);
   }
   else
   {
      diag.UseDevice(true); // typically this is a large vector, so store on device
      diag = 0.0;
      for (BilinearFormIntegrator *integ : integrators)
      {
         integ->AssembleDiagonalMF(diag);
      }
   }

   Array<BilinearFormIntegrator *> &bdr_integrators = *a->GetBBFI();
   if (bdr_integrators.Size() > 0)
   {
      if (bdr_face_restrict_lex)
      {
         bdr_face_y = 0.0;
         for (BilinearFormIntegrator *integ : bdr_integrators)
         {
            integ->AssembleDiagonalMF(bdr_face_y);
         }
         bdr_face_restrict_lex->AddMultTransposeUnsigned(bdr_face_y, diag);
      }
      else
      {
         for (BilinearFormIntegrator *integ : bdr_integrators)
         {
            integ->AssembleDiagonalMF(diag);
         }
      }
   }
}

void MFBilinearFormExtension::Mult(const Vector &x, Vector &y) const
{
   Array<BilinearFormIntegrator *> &integrators = *a->GetDBFI();
   if (integrators.Size() > 0 && elem_restrict)
   {
      elem_restrict->Mult(x, local_x);
      local_y = 0.0;
      for (BilinearFormIntegrator *integ : integrators)
      {
         integ->AddMultMF(local_x, local_y);
      }
      elem_restrict->MultTranspose(local_y, y);
   }
   else
   {
      y.UseDevice(true); // typically this is a large vector, so store on device
      y = 0.0;
      for (BilinearFormIntegrator *integ : integrators)
      {
         integ->AddMultMF(x, y);
      }
   }

   Array<BilinearFormIntegrator *> &bdr_integrators = *a->GetBBFI();
   if (bdr_integrators.Size() > 0)
   {
      if (bdr_face_restrict_lex)
      {
         bdr_face_restrict_lex->Mult(x, bdr_face_x);
         if (bdr_face_x.Size() > 0)
         {
            bdr_face_y = 0.0;
            for (BilinearFormIntegrator *integ : bdr_integrators)
            {
               integ->AddMultMF(bdr_face_x, bdr_face_y);
            }
            bdr_face_restrict_lex->AddMultTransposeInPlace(bdr_face_y, y);
         }
      }
      else
      {
         for (BilinearFormIntegrator *integ : bdr_integrators)
         {
            integ->AddMultMF(x, y);
         }
      }
   }
}

void MFBilinearFormExtension::AddMult(const Vector &x, Vector &y,
                                      const double c) const
{
   Array<BilinearFormIntegrator *> &integrators = *a->GetDBFI();
   if (integrators.Size() > 0)
   {
      if (elem_restrict)
      {
         elem_restrict->Mult(x, local_x);
         local_y = 0.0;
         for (BilinearFormIntegrator *integ : integrators)
         {
            integ->AddMultMF(local_x, local_y);
         }
         if (c != 1.0)
         {
            local_y *= c;
         }
         elem_restrict->AddMultTranspose(local_y, y);
      }
      else
      {
         y.UseDevice(true); // typically this is a large vector, so store on device
         if (c != 1.0)
         {
            temp_y.SetSize(y.Size());
            temp_y.UseDevice(true);
            temp_y = 0.0;
            for (BilinearFormIntegrator *integ : integrators)
            {
               integ->AddMultMF(x, temp_y);
            }
            y.Add(c, temp_y);
         }
         else
         {
            for (BilinearFormIntegrator *integ : integrators)
            {
               integ->AddMultMF(x, y);
            }
         }
      }
   }

   Array<BilinearFormIntegrator *> &bdr_integrators = *a->GetBBFI();
   if (bdr_integrators.Size() > 0)
   {
      if (bdr_face_restrict_lex)
      {
         bdr_face_restrict_lex->Mult(x, bdr_face_x);
         if (bdr_face_x.Size() > 0)
         {
            bdr_face_y = 0.0;
            for (BilinearFormIntegrator *integ : bdr_integrators)
            {
               integ->AddMultMF(bdr_face_x, bdr_face_y);
            }
            if (c != 1.0)
            {
               bdr_face_y *= c;
            }
            bdr_face_restrict_lex->AddMultTransposeInPlace(bdr_face_y, y);
         }
      }
      else
      {
         if (c != 1.0)
         {
            temp_y.SetSize(y.Size());
            temp_y.UseDevice(true);
            temp_y = 0.0;
            for (BilinearFormIntegrator *integ : bdr_integrators)
            {
               integ->AddMultMF(x, temp_y);
            }
            y.Add(c, temp_y);
         }
         else
         {
            for (BilinearFormIntegrator *integ : bdr_integrators)
            {
               integ->AddMultMF(x, y);
            }
         }
      }
   }
}

void MFBilinearFormExtension::MultTranspose(const Vector &x, Vector &y) const
{
   Array<BilinearFormIntegrator *> &integrators = *a->GetDBFI();
   if (integrators.Size() > 0 && elem_restrict)
   {
      elem_restrict->Mult(x, local_x);
      local_y = 0.0;
      for (BilinearFormIntegrator *integ : integrators)
      {
         integ->AddMultTransposeMF(local_x, local_y);
      }
      elem_restrict->MultTranspose(local_y, y);
   }
   else
   {
      y.UseDevice(true); // typically this is a large vector, so store on device
      y = 0.0;
      for (BilinearFormIntegrator *integ : integrators)
      {
         integ->AddMultTransposeMF(x, y);
      }
   }

   Array<BilinearFormIntegrator *> &bdr_integrators = *a->GetBBFI();
   if (bdr_integrators.Size() > 0)
   {
      if (bdr_face_restrict_lex)
      {
         bdr_face_restrict_lex->Mult(x, bdr_face_x);
         if (bdr_face_x.Size() > 0)
         {
            bdr_face_y = 0.0;
            for (BilinearFormIntegrator *integ : bdr_integrators)
            {
               integ->AddMultTransposeMF(bdr_face_x, bdr_face_y);
            }
            bdr_face_restrict_lex->AddMultTransposeInPlace(bdr_face_y, y);
         }
      }
      else
      {
         for (BilinearFormIntegrator *integ : bdr_integrators)
         {
            integ->AddMultTransposeMF(x, y);
         }
      }
   }
}

void MFBilinearFormExtension::AddMultTranspose(const Vector &x, Vector &y,
                                               const double c) const
{
   Array<BilinearFormIntegrator *> &integrators = *a->GetDBFI();
   if (integrators.Size() > 0)
   {
      if (elem_restrict)
      {
         elem_restrict->Mult(x, local_x);
         local_y = 0.0;
         for (BilinearFormIntegrator *integ : integrators)
         {
            integ->AddMultTransposeMF(local_x, local_y);
         }
         if (c != 1.0)
         {
            local_y *= c;
         }
         elem_restrict->AddMultTranspose(local_y, y);
      }
      else
      {
         y.UseDevice(true); // typically this is a large vector, so store on device
         if (c != 1.0)
         {
            temp_y.SetSize(y.Size());
            temp_y.UseDevice(true);
            temp_y = 0.0;
            for (BilinearFormIntegrator *integ : integrators)
            {
               integ->AddMultTransposeMF(x, temp_y);
            }
            y.Add(c, temp_y);
         }
         else
         {
            for (BilinearFormIntegrator *integ : integrators)
            {
               integ->AddMultTransposeMF(x, y);
            }
         }
      }
   }

   Array<BilinearFormIntegrator *> &bdr_integrators = *a->GetBBFI();
   if (bdr_integrators.Size() > 0)
   {
      if (bdr_face_restrict_lex)
      {
         bdr_face_restrict_lex->Mult(x, bdr_face_x);
         if (bdr_face_x.Size() > 0)
         {
            bdr_face_y = 0.0;
            for (BilinearFormIntegrator *integ : bdr_integrators)
            {
               integ->AddMultTransposeMF(bdr_face_x, bdr_face_y);
            }
            if (c != 1.0)
            {
               bdr_face_y *= c;
            }
            bdr_face_restrict_lex->AddMultTransposeInPlace(bdr_face_y, y);
         }
      }
      else
      {
         if (c != 1.0)
         {
            temp_y.SetSize(y.Size());
            temp_y.UseDevice(true);
            temp_y = 0.0;
            for (BilinearFormIntegrator *integ : bdr_integrators)
            {
               integ->AddMultTransposeMF(x, temp_y);
            }
            y.Add(c, temp_y);
         }
         else
         {
            for (BilinearFormIntegrator *integ : bdr_integrators)
            {
               integ->AddMultTransposeMF(x, y);
            }
         }
      }
   }
}

void MFBilinearFormExtension::Update()
{
   fes = a->FESpace();
   height = width = fes->GetVSize();

   elem_restrict = nullptr;
   int_face_restrict_lex = nullptr;
   bdr_face_restrict_lex = nullptr;
}

/// Data and methods for partially-assembled bilinear forms
PABilinearFormExtension::PABilinearFormExtension(BilinearForm *form)
   : MFBilinearFormExtension(form)
{
}

void PABilinearFormExtension::Assemble()
{
   SetupRestrictionOperators(L2FaceValues::DoubleValued);

   Array<BilinearFormIntegrator *> &integrators = *a->GetDBFI();
   for (BilinearFormIntegrator *integ : integrators)
   {
      integ->AssemblePA(*fes);
   }

   Array<BilinearFormIntegrator *> &bdr_integrators = *a->GetBBFI();
   for (BilinearFormIntegrator *integ : bdr_integrators)
   {
      integ->AssemblePABoundary(*fes);
   }

   Array<BilinearFormIntegrator *> &int_face_integrators = *a->GetFBFI();
   for (BilinearFormIntegrator *integ : int_face_integrators)
   {
      integ->AssemblePAInteriorFaces(*fes);
   }

   Array<BilinearFormIntegrator *> &bdr_face_integrators = *a->GetBFBFI();
   for (BilinearFormIntegrator *integ : bdr_face_integrators)
   {
      integ->AssemblePABoundaryFaces(*fes);
   }
}

void PABilinearFormExtension::AssembleDiagonal(Vector &diag) const
{
   Array<BilinearFormIntegrator *> &integrators = *a->GetDBFI();
   if (integrators.Size() > 0 && elem_restrict)
   {
      local_y = 0.0;
      for (BilinearFormIntegrator *integ : integrators)
      {
         integ->AssembleDiagonalPA(local_y);
      }
      elem_restrict->MultTransposeUnsigned(local_y, diag);
   }
   else
   {
      diag.UseDevice(true); // typically this is a large vector, so store on device
      diag = 0.0;
      for (BilinearFormIntegrator *integ : integrators)
      {
         integ->AssembleDiagonalPA(diag);
      }
   }

   Array<BilinearFormIntegrator *> &bdr_integrators = *a->GetBBFI();
   if (bdr_integrators.Size() > 0)
   {
      if (bdr_face_restrict_lex)
      {
         bdr_face_y = 0.0;
         for (BilinearFormIntegrator *integ : bdr_integrators)
         {
            integ->AssembleDiagonalPA(bdr_face_y);
         }
         bdr_face_restrict_lex->AddMultTransposeUnsigned(bdr_face_y, diag);
      }
      else
      {
         for (BilinearFormIntegrator *integ : bdr_integrators)
         {
            integ->AssembleDiagonalPA(diag);
         }
      }
   }
}

void PABilinearFormExtension::Mult(const Vector &x, Vector &y) const
{
   Array<BilinearFormIntegrator *> &integrators = *a->GetDBFI();
   if (integrators.Size() > 0 && elem_restrict)
   {
      elem_restrict->Mult(x, local_x);
      local_y = 0.0;
      for (BilinearFormIntegrator *integ : integrators)
      {
         integ->AddMultPA(local_x, local_y);
      }
      elem_restrict->MultTranspose(local_y, y);
   }
   else
   {
      y.UseDevice(true); // typically this is a large vector, so store on device
      y = 0.0;
      for (BilinearFormIntegrator *integ : integrators)
      {
         integ->AddMultPA(x, y);
      }
   }

   Array<BilinearFormIntegrator *> &int_face_integrators = *a->GetFBFI();
   if (int_face_integrators.Size() > 0)
   {
      if (int_face_restrict_lex)
      {
         int_face_restrict_lex->Mult(x, int_face_x);
         if (int_face_x.Size() > 0)
         {
            int_face_y = 0.0;
            for (BilinearFormIntegrator *integ : int_face_integrators)
            {
               integ->AddMultPA(int_face_x, int_face_y);
            }
            int_face_restrict_lex->AddMultTransposeInPlace(int_face_y, y);
         }
      }
      else
      {
         for (BilinearFormIntegrator *integ : int_face_integrators)
         {
            integ->AddMultPA(x, y);
         }
      }
   }

   Array<BilinearFormIntegrator *> &bdr_integrators = *a->GetBBFI();
   Array<BilinearFormIntegrator *> &bdr_face_integrators = *a->GetBFBFI();
   if (bdr_integrators.Size() > 0 || bdr_face_integrators.Size() > 0)
   {
      if (bdr_face_restrict_lex)
      {
         bdr_face_restrict_lex->Mult(x, bdr_face_x);
         if (bdr_face_x.Size() > 0)
         {
            bdr_face_y = 0.0;
            for (BilinearFormIntegrator *integ : bdr_integrators)
            {
               integ->AddMultPA(bdr_face_x, bdr_face_y);
            }
            for (BilinearFormIntegrator *integ : bdr_face_integrators)
            {
               integ->AddMultPA(bdr_face_x, bdr_face_y);
            }
            bdr_face_restrict_lex->AddMultTransposeInPlace(bdr_face_y, y);
         }
      }
      else
      {
         for (BilinearFormIntegrator *integ : bdr_integrators)
         {
            integ->AddMultPA(x, y);
         }
         for (BilinearFormIntegrator *integ : bdr_face_integrators)
         {
            integ->AddMultPA(x, y);
         }
      }
   }
}

void PABilinearFormExtension::AddMult(const Vector &x, Vector &y,
                                      const double c) const
{
   Array<BilinearFormIntegrator *> &integrators = *a->GetDBFI();
   if (integrators.Size() > 0)
   {
      if (elem_restrict)
      {
         elem_restrict->Mult(x, local_x);
         local_y = 0.0;
         for (BilinearFormIntegrator *integ : integrators)
         {
            integ->AddMultPA(local_x, local_y);
         }
         if (c != 1.0)
         {
            local_y *= c;
         }
         elem_restrict->AddMultTranspose(local_y, y);
      }
      else
      {
         y.UseDevice(true); // typically this is a large vector, so store on device
         if (c != 1.0)
         {
            temp_y.SetSize(y.Size());
            temp_y.UseDevice(true);
            temp_y = 0.0;
            for (BilinearFormIntegrator *integ : integrators)
            {
               integ->AddMultPA(x, temp_y);
            }
            y.Add(c, temp_y);
         }
         else
         {
            for (BilinearFormIntegrator *integ : integrators)
            {
               integ->AddMultPA(x, y);
            }
         }
      }
   }

   Array<BilinearFormIntegrator *> &int_face_integrators = *a->GetFBFI();
   if (int_face_integrators.Size() > 0)
   {
      if (int_face_restrict_lex)
      {
         int_face_restrict_lex->Mult(x, int_face_x);
         if (int_face_x.Size() > 0)
         {
            int_face_y = 0.0;
            for (BilinearFormIntegrator *integ : int_face_integrators)
            {
               integ->AddMultPA(int_face_x, int_face_y);
            }
            if (c != 1.0)
            {
               int_face_y *= c;
            }
            int_face_restrict_lex->AddMultTransposeInPlace(int_face_y, y);
         }
      }
      else
      {
         if (c != 1.0)
         {
            temp_y.SetSize(y.Size());
            temp_y.UseDevice(true);
            temp_y = 0.0;
            for (BilinearFormIntegrator *integ : int_face_integrators)
            {
               integ->AddMultPA(x, temp_y);
            }
            y.Add(c, temp_y);
         }
         else
         {
            for (BilinearFormIntegrator *integ : int_face_integrators)
            {
               integ->AddMultPA(x, y);
            }
         }
      }
   }

   Array<BilinearFormIntegrator *> &bdr_integrators = *a->GetBBFI();
   Array<BilinearFormIntegrator *> &bdr_face_integrators = *a->GetBFBFI();
   if (bdr_integrators.Size() > 0 || bdr_face_integrators.Size() > 0)
   {
      if (bdr_face_restrict_lex)
      {
         bdr_face_restrict_lex->Mult(x, bdr_face_x);
         if (bdr_face_x.Size() > 0)
         {
            bdr_face_y = 0.0;
            for (BilinearFormIntegrator *integ : bdr_integrators)
            {
               integ->AddMultPA(bdr_face_x, bdr_face_y);
            }
            for (BilinearFormIntegrator *integ : bdr_face_integrators)
            {
               integ->AddMultPA(bdr_face_x, bdr_face_y);
            }
            if (c != 1.0)
            {
               bdr_face_y *= c;
            }
            bdr_face_restrict_lex->AddMultTransposeInPlace(bdr_face_y, y);
         }
      }
      else
      {
         if (c != 1.0)
         {
            temp_y.SetSize(y.Size());
            temp_y.UseDevice(true);
            temp_y = 0.0;
            for (BilinearFormIntegrator *integ : bdr_integrators)
            {
               integ->AddMultPA(x, temp_y);
            }
            for (BilinearFormIntegrator *integ : bdr_face_integrators)
            {
               integ->AddMultPA(x, temp_y);
            }
            y.Add(c, temp_y);
         }
         else
         {
            for (BilinearFormIntegrator *integ : bdr_integrators)
            {
               integ->AddMultPA(x, y);
            }
            for (BilinearFormIntegrator *integ : bdr_face_integrators)
            {
               integ->AddMultPA(x, y);
            }
         }
      }
   }
}

void PABilinearFormExtension::MultTranspose(const Vector &x, Vector &y) const
{
   Array<BilinearFormIntegrator *> &integrators = *a->GetDBFI();
   if (integrators.Size() > 0 && elem_restrict)
   {
      elem_restrict->Mult(x, local_x);
      local_y = 0.0;
      for (BilinearFormIntegrator *integ : integrators)
      {
         integ->AddMultTransposePA(local_x, local_y);
      }
      elem_restrict->MultTranspose(local_y, y);
   }
   else
   {
      y.UseDevice(true);
      y = 0.0;
      for (BilinearFormIntegrator *integ : integrators)
      {
         integ->AddMultTransposePA(x, y);
      }
   }

   Array<BilinearFormIntegrator *> &int_face_integrators = *a->GetFBFI();
   if (int_face_integrators.Size() > 0)
   {
      if (int_face_restrict_lex)
      {
         int_face_restrict_lex->Mult(x, int_face_x);
         if (int_face_x.Size() > 0)
         {
            int_face_y = 0.0;
            for (BilinearFormIntegrator *integ : int_face_integrators)
            {
               integ->AddMultTransposePA(int_face_x, int_face_y);
            }
            int_face_restrict_lex->AddMultTransposeInPlace(int_face_y, y);
         }
      }
      else
      {
         for (BilinearFormIntegrator *integ : int_face_integrators)
         {
            integ->AddMultTransposePA(x, y);
         }
      }
   }

   Array<BilinearFormIntegrator *> &bdr_integrators = *a->GetBBFI();
   Array<BilinearFormIntegrator *> &bdr_face_integrators = *a->GetBFBFI();
   if (bdr_integrators.Size() > 0 || bdr_face_integrators.Size() > 0)
   {
      if (bdr_face_restrict_lex)
      {
         bdr_face_restrict_lex->Mult(x, bdr_face_x);
         if (bdr_face_x.Size() > 0)
         {
            bdr_face_y = 0.0;
            for (BilinearFormIntegrator *integ : bdr_integrators)
            {
               integ->AddMultTransposePA(bdr_face_x, bdr_face_y);
            }
            for (BilinearFormIntegrator *integ : bdr_face_integrators)
            {
               integ->AddMultTransposePA(bdr_face_x, bdr_face_y);
            }
            bdr_face_restrict_lex->AddMultTransposeInPlace(bdr_face_y, y);
         }
      }
      else
      {
         for (BilinearFormIntegrator *integ : bdr_integrators)
         {
            integ->AddMultTransposePA(x, y);
         }
         for (BilinearFormIntegrator *integ : bdr_face_integrators)
         {
            integ->AddMultTransposePA(x, y);
         }
      }
   }
}

void PABilinearFormExtension::AddMultTranspose(const Vector &x, Vector &y,
                                               const double c) const
{
   Array<BilinearFormIntegrator *> &integrators = *a->GetDBFI();
   if (integrators.Size() > 0)
   {
      if (elem_restrict)
      {
         elem_restrict->Mult(x, local_x);
         local_y = 0.0;
         for (BilinearFormIntegrator *integ : integrators)
         {
            integ->AddMultTransposePA(local_x, local_y);
         }
         if (c != 1.0)
         {
            local_y *= c;
         }
         elem_restrict->AddMultTranspose(local_y, y);
      }
      else
      {
         y.UseDevice(true); // typically this is a large vector, so store on device
         if (c != 1.0)
         {
            temp_y.SetSize(y.Size());
            temp_y.UseDevice(true);
            temp_y = 0.0;
            for (BilinearFormIntegrator *integ : integrators)
            {
               integ->AddMultTransposePA(x, temp_y);
            }
            y.Add(c, temp_y);
         }
         else
         {
            for (BilinearFormIntegrator *integ : integrators)
            {
               integ->AddMultTransposePA(x, y);
            }
         }
      }
   }

   Array<BilinearFormIntegrator *> &int_face_integrators = *a->GetFBFI();
   if (int_face_integrators.Size() > 0)
   {
      if (int_face_restrict_lex)
      {
         int_face_restrict_lex->Mult(x, int_face_x);
         if (int_face_x.Size() > 0)
         {
            int_face_y = 0.0;
            for (BilinearFormIntegrator *integ : int_face_integrators)
            {
               integ->AddMultTransposePA(int_face_x, int_face_y);
            }
            if (c != 1.0)
            {
               int_face_y *= c;
            }
            int_face_restrict_lex->AddMultTransposeInPlace(int_face_y, y);
         }
      }
      else
      {
         if (c != 1.0)
         {
            temp_y.SetSize(y.Size());
            temp_y.UseDevice(true);
            temp_y = 0.0;
            for (BilinearFormIntegrator *integ : int_face_integrators)
            {
               integ->AddMultTransposePA(x, temp_y);
            }
            y.Add(c, temp_y);
         }
         else
         {
            for (BilinearFormIntegrator *integ : int_face_integrators)
            {
               integ->AddMultTransposePA(x, y);
            }
         }
      }
   }

   Array<BilinearFormIntegrator *> &bdr_integrators = *a->GetBBFI();
   Array<BilinearFormIntegrator *> &bdr_face_integrators = *a->GetBFBFI();
   if (bdr_integrators.Size() > 0 || bdr_face_integrators.Size() > 0)
   {
      if (bdr_face_restrict_lex)
      {
         bdr_face_restrict_lex->Mult(x, bdr_face_x);
         if (bdr_face_x.Size() > 0)
         {
            bdr_face_y = 0.0;
            for (BilinearFormIntegrator *integ : bdr_integrators)
            {
               integ->AddMultTransposePA(bdr_face_x, bdr_face_y);
            }
            for (BilinearFormIntegrator *integ : bdr_face_integrators)
            {
               integ->AddMultTransposePA(bdr_face_x, bdr_face_y);
            }
            if (c != 1.0)
            {
               bdr_face_y *= c;
            }
            bdr_face_restrict_lex->AddMultTransposeInPlace(bdr_face_y, y);
         }
      }
      else
      {
         if (c != 1.0)
         {
            temp_y.SetSize(y.Size());
            temp_y.UseDevice(true);
            temp_y = 0.0;
            for (BilinearFormIntegrator *integ : bdr_integrators)
            {
               integ->AddMultTransposePA(x, temp_y);
            }
            for (BilinearFormIntegrator *integ : bdr_face_integrators)
            {
               integ->AddMultTransposePA(x, temp_y);
            }
            y.Add(c, temp_y);
         }
         else
         {
            for (BilinearFormIntegrator *integ : bdr_integrators)
            {
               integ->AddMultTransposePA(x, y);
            }
            for (BilinearFormIntegrator *integ : bdr_face_integrators)
            {
               integ->AddMultTransposePA(x, y);
            }
         }
      }
   }
}

/// Data and methods for element-assembled bilinear forms
EABilinearFormExtension::EABilinearFormExtension(BilinearForm *form)
   : PABilinearFormExtension(form),
     factorize_face_terms(fes->IsDGSpace() && fes->Conforming())
{
}

void EABilinearFormExtension::Assemble()
{
   SetupRestrictionOperators(L2FaceValues::SingleValued);

   ne = fes->GetNE();
   elem_dofs = fes->GetFE(0)->GetDof();

   Array<BilinearFormIntegrator *> &integrators = *a->GetDBFI();
   if (integrators.Size() > 0)
   {
      ea_data.SetSize(ne * elem_dofs * elem_dofs, Device::GetMemoryType());
      ea_data.UseDevice(true);
      ea_data = 0.0;
      for (BilinearFormIntegrator *integ : integrators)
      {
         integ->AssembleEA(*fes, ea_data);
      }
   }

   MFEM_VERIFY(a->GetBBFI()->Size() == 0,
               "Element assembly does not support AddBoundaryIntegrator yet.");

   nf_int = fes->GetNFbyType(FaceType::Interior);
   nf_bdr = fes->GetNFbyType(FaceType::Boundary);
   face_dofs = fes->GetTraceElement(0,
                                    fes->GetMesh()->GetFaceGeometry(0))->GetDof();

   Array<BilinearFormIntegrator *> &int_face_integrators = *a->GetFBFI();
   if (int_face_integrators.Size() > 0)
   {
      ea_data_int.SetSize(2 * nf_int * face_dofs * face_dofs,
                          Device::GetMemoryType());
      ea_data_ext.SetSize(2 * nf_int * face_dofs * face_dofs,
                          Device::GetMemoryType());
      ea_data_int = 0.0;
      ea_data_ext = 0.0;
      for (BilinearFormIntegrator *integ : int_face_integrators)
      {
         integ->AssembleEAInteriorFaces(*fes, ea_data_int, ea_data_ext);
      }
   }

   Array<BilinearFormIntegrator *> &bdr_face_integrators = *a->GetBFBFI();
   if (bdr_face_integrators.Size() > 0)
   {
      ea_data_bdr.SetSize(nf_bdr * face_dofs * face_dofs, Device::GetMemoryType());
      ea_data_bdr = 0.0;
      for (BilinearFormIntegrator *integ : bdr_face_integrators)
      {
         integ->AssembleEABoundaryFaces(*fes, ea_data_bdr);
      }
   }

   if (factorize_face_terms && int_face_restrict_lex)
   {
      auto l2_face_restrict = dynamic_cast<const L2FaceRestriction &>
                              (*int_face_restrict_lex);
      l2_face_restrict.AddFaceMatricesToElementMatrices(ea_data_int, ea_data);
   }
   if (factorize_face_terms && bdr_face_restrict_lex)
   {
      auto l2_face_restrict = dynamic_cast<const L2FaceRestriction &>
                              (*bdr_face_restrict_lex);
      l2_face_restrict.AddFaceMatricesToElementMatrices(ea_data_bdr, ea_data);
   }
}

void EABilinearFormExtension::Mult(const Vector &x, Vector &y) const
{
   Array<BilinearFormIntegrator *> &integrators = *a->GetDBFI();
   auto Apply = [](const int nelem, const int ndofs, const Vector &data,
                   const Vector &x, Vector &y)
   {
      auto X = Reshape(x.Read(), ndofs, nelem);
      auto Y = Reshape(y.ReadWrite(), ndofs, nelem);
      auto A = Reshape(data.Read(), ndofs, ndofs, nelem);
      mfem::forall(nelem * ndofs, [=] MFEM_HOST_DEVICE (int k)
      {
         const int e = k / ndofs;
         const int j = k % ndofs;
         double res = 0.0;
         for (int i = 0; i < ndofs; i++)
         {
            res += A(i, j, e) * X(i, e);
         }
         Y(j, e) += res;
      });
   };
   if (integrators.Size() > 0 && elem_restrict)
   {
      elem_restrict->Mult(x, local_x);
      local_y = 0.0;
      Apply(ne, elem_dofs, ea_data, local_x, local_y);
      elem_restrict->MultTranspose(local_y, y);
   }
   else
   {
      y.UseDevice(true); // typically this is a large vector, so store on device
      y = 0.0;
      if (integrators.Size() > 0)
      {
         Apply(ne, elem_dofs, ea_data, x, y);
      }
   }

   // Treatment of interior faces
   Array<BilinearFormIntegrator *> &int_face_integrators = *a->GetFBFI();
   auto ApplyIntFace = [](const int nface, const int ndofs, const Vector &data,
                          const Vector &x, Vector &y)
   {
      auto X = Reshape(x.Read(), ndofs, 2, nface);
      auto Y = Reshape(y.ReadWrite(), ndofs, 2, nface);
      auto A = Reshape(data.Read(), ndofs, ndofs, 2, nface);
      mfem::forall(nface * ndofs, [=] MFEM_HOST_DEVICE (int k)
      {
         const int f = k / ndofs;
         const int j = k % ndofs;
         double res = 0.0;
         for (int i = 0; i < ndofs; i++)
         {
            res += A(i, j, 0, f) * X(i, 0, f);
         }
         Y(j, 0, f) += res;
         res = 0.0;
         for (int i = 0; i < ndofs; i++)
         {
            res += A(i, j, 1, f) * X(i, 1, f);
         }
         Y(j, 1, f) += res;
      });
   };
   auto ApplyExtFace = [](const int nface, const int ndofs, const Vector &data,
                          const Vector &x, Vector &y)
   {
      auto X = Reshape(x.Read(), ndofs, 2, nface);
      auto Y = Reshape(y.ReadWrite(), ndofs, 2, nface);
      auto A = Reshape(data.Read(), ndofs, ndofs, 2, nface);
      mfem::forall(nface * ndofs, [=] MFEM_HOST_DEVICE (int k)
      {
         const int f = k / ndofs;
         const int j = k % ndofs;
         double res = 0.0;
         for (int i = 0; i < ndofs; i++)
         {
            res += A(i, j, 0, f) * X(i, 0, f);
         }
         Y(j, 1, f) += res;
         res = 0.0;
         for (int i = 0; i < ndofs; i++)
         {
            res += A(i, j, 1, f) * X(i, 1, f);
         }
         Y(j, 0, f) += res;
      });
   };
   if (int_face_restrict_lex && int_face_integrators.Size() > 0)
   {
      int_face_restrict_lex->Mult(x, int_face_x);
      if (int_face_x.Size() > 0)
      {
         int_face_y = 0.0;
         if (!factorize_face_terms)
         {
            ApplyIntFace(nf_int, face_dofs, ea_data_int, int_face_x, int_face_y);
         }
         ApplyExtFace(nf_int, face_dofs, ea_data_ext, int_face_x, int_face_y);
         int_face_restrict_lex->AddMultTransposeInPlace(int_face_y, y);
      }
   }

   // Treatment of boundary faces
   Array<BilinearFormIntegrator *> &bdr_face_integrators = *a->GetBFBFI();
   if (!factorize_face_terms && bdr_face_restrict_lex &&
       bdr_face_integrators.Size() > 0)
   {
      bdr_face_restrict_lex->Mult(x, bdr_face_x);
      if (bdr_face_x.Size() > 0)
      {
         bdr_face_y = 0.0;
         Apply(nf_bdr, face_dofs, ea_data_bdr, bdr_face_x, bdr_face_y);
         bdr_face_restrict_lex->AddMultTransposeInPlace(bdr_face_y, y);
      }
   }
}

void EABilinearFormExtension::MultTranspose(const Vector &x, Vector &y) const
{
   Array<BilinearFormIntegrator *> &integrators = *a->GetDBFI();
   auto ApplyTranspose = [](const int nelem, const int ndofs, const Vector &data,
                            const Vector &x, Vector &y)
   {
      auto X = Reshape(x.Read(), ndofs, nelem);
      auto Y = Reshape(y.ReadWrite(), ndofs, nelem);
      auto A = Reshape(data.Read(), ndofs, ndofs, nelem);
      mfem::forall(nelem * ndofs, [=] MFEM_HOST_DEVICE (int k)
      {
         const int e = k / ndofs;
         const int j = k % ndofs;
         double res = 0.0;
         for (int i = 0; i < ndofs; i++)
         {
            res += A(j, i, e) * X(i, e);
         }
         Y(j, e) += res;
      });
   };
   if (integrators.Size() > 0 && elem_restrict)
   {
      elem_restrict->Mult(x, local_x);
      local_y = 0.0;
      ApplyTranspose(ne, elem_dofs, ea_data, local_x, local_y);
      elem_restrict->MultTranspose(local_y, y);
   }
   else
   {
      y.UseDevice(true); // typically this is a large vector, so store on device
      y = 0.0;
      if (integrators.Size() > 0)
      {
         ApplyTranspose(ne, elem_dofs, ea_data, x, y);
      }
   }

   // Treatment of interior faces
   Array<BilinearFormIntegrator *> &int_face_integrators = *a->GetFBFI();
   auto ApplyIntFaceTranspose = [](const int nface, const int ndofs,
                                   const Vector &data, const Vector &x, Vector &y)
   {
      auto X = Reshape(x.Read(), ndofs, 2, nface);
      auto Y = Reshape(y.ReadWrite(), ndofs, 2, nface);
      auto A = Reshape(data.Read(), ndofs, ndofs, 2, nface);
      mfem::forall(nface * ndofs, [=] MFEM_HOST_DEVICE (int k)
      {
         const int f = k / ndofs;
         const int j = k % ndofs;
         double res = 0.0;
         for (int i = 0; i < ndofs; i++)
         {
            res += A(j, i, 0, f) * X(i, 0, f);
         }
         Y(j, 0, f) += res;
         res = 0.0;
         for (int i = 0; i < ndofs; i++)
         {
            res += A(j, i, 1, f) * X(i, 1, f);
         }
         Y(j, 1, f) += res;
      });
   };
   auto ApplyExtFaceTranspose = [](const int nface, const int ndofs,
                                   const Vector &data, const Vector &x, Vector &y)
   {
      auto X = Reshape(x.Read(), ndofs, 2, nface);
      auto Y = Reshape(y.ReadWrite(), ndofs, 2, nface);
      auto A = Reshape(data.Read(), ndofs, ndofs, 2, nface);
      mfem::forall(nface * ndofs, [=] MFEM_HOST_DEVICE (int k)
      {
         const int f = k / ndofs;
         const int j = k % ndofs;
         double res = 0.0;
         for (int i = 0; i < ndofs; i++)
         {
            res += A(j, i, 1, f) * X(i, 0, f);
         }
         Y(j, 1, f) += res;
         res = 0.0;
         for (int i = 0; i < ndofs; i++)
         {
            res += A(j, i, 0, f) * X(i, 1, f);
         }
         Y(j, 0, f) += res;
      });
   };
   if (int_face_restrict_lex && int_face_integrators.Size() > 0)
   {
      int_face_restrict_lex->Mult(x, int_face_x);
      if (int_face_x.Size() > 0)
      {
         int_face_y = 0.0;
         if (!factorize_face_terms)
         {
            ApplyIntFaceTranspose(nf_int, face_dofs, ea_data_int, int_face_x, int_face_y);
         }
         ApplyExtFaceTranspose(nf_int, face_dofs, ea_data_ext, int_face_x, int_face_y);
         int_face_restrict_lex->AddMultTransposeInPlace(int_face_y, y);
      }
   }

   // Treatment of boundary faces
   Array<BilinearFormIntegrator *> &bdr_face_integrators = *a->GetBFBFI();
   if (!factorize_face_terms && bdr_face_restrict_lex &&
       bdr_face_integrators.Size() > 0)
   {
      bdr_face_restrict_lex->Mult(x, bdr_face_x);
      if (bdr_face_x.Size() > 0)
      {
         bdr_face_y = 0.0;
         ApplyTranspose(nf_bdr, face_dofs, ea_data_bdr, bdr_face_x, bdr_face_y);
         bdr_face_restrict_lex->AddMultTransposeInPlace(bdr_face_y, y);
      }
   }
}

/// Data and methods for fully-assembled bilinear forms
FABilinearFormExtension::FABilinearFormExtension(BilinearForm *form)
   : EABilinearFormExtension(form),
     mat(a->mat)
{
#ifdef MFEM_USE_MPI
   const ParFiniteElementSpace *pfes = nullptr;
   if (a->GetFBFI()->Size() > 0 &&
       (pfes = dynamic_cast<const ParFiniteElementSpace *>(form->FESpace())))
   {
      const_cast<ParFiniteElementSpace *>(pfes)->ExchangeFaceNbrData();
   }
#endif
}

void FABilinearFormExtension::Assemble()
{
   EABilinearFormExtension::Assemble();

   int width = fes->GetVSize();
   int height = fes->GetVSize();
   bool keep_nbr_block = false;
#ifdef MFEM_USE_MPI
   const ParFiniteElementSpace *pfes = nullptr;
   if (a->GetFBFI()->Size() > 0 &&
       (pfes = dynamic_cast<const ParFiniteElementSpace *>(fes)))
   {
      const_cast<ParFiniteElementSpace *>(pfes)->ExchangeFaceNbrData();
      width += pfes->GetFaceNbrVSize();
      dg_x.SetSize(width);
      ParBilinearForm *pb = nullptr;
      if ((pb = dynamic_cast<ParBilinearForm *>(a)) && pb->keep_nbr_block)
      {
         height += pfes->GetFaceNbrVSize();
         dg_y.SetSize(height);
         keep_nbr_block = true;
      }
   }
#endif
   if (a->mat) // We reuse the sparse matrix memory
   {
      if (fes->IsDGSpace())
      {
         const auto *restE =
            static_cast<const L2ElementRestriction *>(elem_restrict);
         const auto *restF =
            static_cast<const L2FaceRestriction *>(int_face_restrict_lex);
         MFEM_VERIFY(fes->Conforming(),
                     "Full Assembly not yet supported on NCMesh.");
         // 1. Fill J and Data
         // 1.1 Fill J and Data with Elem ea_data
         restE->FillJAndData(ea_data, *mat);
         // 1.2 Fill J and Data with Face ea_data_ext
         if (restF) { restF->FillJAndData(ea_data_ext, *mat, keep_nbr_block); }
         // 1.3 Shift indirections in I back to original
         auto I = mat->HostReadWriteI();
         for (int i = height; i > 0; i--)
         {
            I[i] = I[i-1];
         }
         I[0] = 0;
      }
      else
      {
         const auto &rest =
            static_cast<const ConformingElementRestriction&>(*elem_restrict);
         rest.FillJAndData(ea_data, *mat);
      }
   }
   else // We create, compute the sparsity, and fill the sparse matrix
   {
      mat = new SparseMatrix;
      mat->OverrideSize(height, width);
      if (fes->IsDGSpace())
      {
         const auto *restE =
            static_cast<const L2ElementRestriction *>(elem_restrict);
         const auto *restF =
            static_cast<const L2FaceRestriction *>(int_face_restrict_lex);
         MFEM_VERIFY(fes->Conforming(),
                     "Full Assembly not yet supported on NCMesh.");
         // 1. Fill I
         mat->GetMemoryI().New(height+1, mat->GetMemoryI().GetMemoryType());
         //  1.1 Increment with restE
         restE->FillI(*mat);
         //  1.2 Increment with restF
         if (restF) { restF->FillI(*mat, keep_nbr_block); }
         //  1.3 Sum the non-zeros in I
         auto h_I = mat->HostReadWriteI();
         int cpt = 0;
         for (int i = 0; i < height; i++)
         {
            const int nnz = h_I[i];
            h_I[i] = cpt;
            cpt += nnz;
         }
         const int nnz = cpt;
         h_I[height] = nnz;
         mat->GetMemoryJ().New(nnz, mat->GetMemoryJ().GetMemoryType());
         mat->GetMemoryData().New(nnz, mat->GetMemoryData().GetMemoryType());
         // 2. Fill J and Data
         // 2.1 Fill J and Data with Elem ea_data
         restE->FillJAndData(ea_data, *mat);
         // 2.2 Fill J and Data with Face ea_data_ext
         if (restF) { restF->FillJAndData(ea_data_ext, *mat, keep_nbr_block); }
         // 2.3 Shift indirections in I back to original
         auto I = mat->HostReadWriteI();
         for (int i = height; i > 0; i--)
         {
            I[i] = I[i-1];
         }
         I[0] = 0;
      }
      else
      {
         const auto &rest =
            static_cast<const ConformingElementRestriction &>(*elem_restrict);
         rest.FillSparseMatrix(ea_data, *mat);
      }
      a->mat = mat;
   }
   if (a->sort_sparse_matrix)
   {
      a->mat->SortColumnIndices();
   }
}

void FABilinearFormExtension::DGMult(const Vector &x, Vector &y) const
{
#ifdef MFEM_USE_MPI
   if (const auto pfes = dynamic_cast<const ParFiniteElementSpace *>(fes))
   {
      // DG Prolongation
      ParGridFunction x_gf;
      x_gf.MakeRef(const_cast<ParFiniteElementSpace *>(pfes),
                   const_cast<Vector &>(x), 0);
      x_gf.ExchangeFaceNbrData();
      Vector &shared_x = x_gf.FaceNbrData();
      const int local_size = fes->GetVSize();
      auto dg_x_ptr = dg_x.Write();
      auto x_ptr = x.Read();
      mfem::forall(local_size, [=] MFEM_HOST_DEVICE (int i)
      {
         dg_x_ptr[i] = x_ptr[i];
      });
      const int shared_size = shared_x.Size();
      auto shared_x_ptr = shared_x.Read();
      mfem::forall(shared_size, [=] MFEM_HOST_DEVICE (int i)
      {
         dg_x_ptr[local_size+i] = shared_x_ptr[i];
      });
      ParBilinearForm *pb = nullptr;
      if ((pb = dynamic_cast<ParBilinearForm *>(a)) && pb->keep_nbr_block)
      {
         mat->Mult(dg_x, dg_y);
         // DG Restriction
         auto dg_y_ptr = dg_y.Read();
         auto y_ptr = y.ReadWrite();
         mfem::forall(local_size, [=] MFEM_HOST_DEVICE (int i)
         {
            y_ptr[i] += dg_y_ptr[i];
         });
      }
      else
      {
         mat->Mult(dg_x, y);
      }
   }
   else
#endif
   {
      mat->Mult(x, y);
   }
}

void FABilinearFormExtension::Mult(const Vector &x, Vector &y) const
{
   if (a->GetFBFI()->Size() > 0)
   {
      DGMult(x, y);
   }
   else
   {
      mat->Mult(x, y);
   }
}

void FABilinearFormExtension::DGMultTranspose(const Vector &x, Vector &y) const
{
#ifdef MFEM_USE_MPI
   if (const auto pfes = dynamic_cast<const ParFiniteElementSpace *>(fes))
   {
      // DG Prolongation
      ParGridFunction x_gf;
      x_gf.MakeRef(const_cast<ParFiniteElementSpace *>(pfes),
                   const_cast<Vector &>(x), 0);
      x_gf.ExchangeFaceNbrData();
      Vector &shared_x = x_gf.FaceNbrData();
      const int local_size = fes->GetVSize();
      auto dg_x_ptr = dg_x.Write();
      auto x_ptr = x.Read();
      mfem::forall(local_size, [=] MFEM_HOST_DEVICE (int i)
      {
         dg_x_ptr[i] = x_ptr[i];
      });
      const int shared_size = shared_x.Size();
      auto shared_x_ptr = shared_x.Read();
      mfem::forall(shared_size, [=] MFEM_HOST_DEVICE (int i)
      {
         dg_x_ptr[local_size+i] = shared_x_ptr[i];
      });
      ParBilinearForm *pb = nullptr;
      if ((pb = dynamic_cast<ParBilinearForm *>(a)) && (pb->keep_nbr_block))
      {
         mat->MultTranspose(dg_x, dg_y);
         // DG Restriction
         auto dg_y_ptr = dg_y.Read();
         auto y_ptr = y.ReadWrite();
         mfem::forall(local_size, [=] MFEM_HOST_DEVICE (int i)
         {
            y_ptr[i] += dg_y_ptr[i];
         });
      }
      else
      {
         mat->MultTranspose(dg_x, y);
      }
   }
   else
#endif
   {
      mat->MultTranspose(x, y);
   }
}

void FABilinearFormExtension::MultTranspose(const Vector &x, Vector &y) const
{
   if (a->GetFBFI()->Size() > 0)
   {
      DGMultTranspose(x, y);
   }
   else
   {
      mat->MultTranspose(x, y);
   }
}


/// Base class for extensions to the MixedBilinearForm class
MixedBilinearFormExtension::MixedBilinearFormExtension(MixedBilinearForm *form)
   : Operator(form->Height(), form->Width()), a(form)
{
}

const Operator *MixedBilinearFormExtension::GetProlongation() const
{
   return a->GetProlongation();
}

const Operator *MixedBilinearFormExtension::GetRestriction() const
{
   return a->GetRestriction();
}

const Operator *MixedBilinearFormExtension::GetOutputProlongation() const
{
   return a->GetOutputProlongation();
}

const Operator *MixedBilinearFormExtension::GetOutputRestriction() const
{
   return a->GetOutputRestriction();
}

/// Data and methods for matrix-free mixed bilinear forms
MFMixedBilinearFormExtension::MFMixedBilinearFormExtension(
   MixedBilinearForm *form)
   : MixedBilinearFormExtension(form)
{
   Update();
}

void MFMixedBilinearFormExtension::SetupRestrictionOperators(
   const L2FaceValues m)
{
   if (DeviceCanUseCeed()) { return; }
   ElementDofOrdering trial_ordering = UsesTensorBasis(*trial_fes) ?
                                       ElementDofOrdering::LEXICOGRAPHIC :
                                       ElementDofOrdering::NATIVE;
   ElementDofOrdering test_ordering = UsesTensorBasis(*test_fes) ?
                                      ElementDofOrdering::LEXICOGRAPHIC :
                                      ElementDofOrdering::NATIVE;
   elem_restrict_trial = trial_fes->GetElementRestriction(trial_ordering);
   elem_restrict_test = test_fes->GetElementRestriction(test_ordering);
   if (elem_restrict_trial)
   {
      local_trial.SetSize(elem_restrict_trial->Height(),
                          Device::GetDeviceMemoryType());
      local_trial.UseDevice(true); // ensure 'local_trial = 0.0' is done on device
   }
   if (elem_restrict_test)
   {
      local_test.SetSize(elem_restrict_test->Height(),
                         Device::GetDeviceMemoryType());
      local_test.UseDevice(true); // ensure 'local_test = 0.0' is done on device
   }

   // Construct face restriction operators only if the bilinear form has
   // interior or boundary face integrators
   if (a->GetTFBFI()->Size() > 0)
   {
      if (int_face_restrict_lex_trial == nullptr)
      {
         int_face_restrict_lex_trial = trial_fes->GetFaceRestriction(
                                          ElementDofOrdering::LEXICOGRAPHIC,
                                          FaceType::Interior);
         int_face_trial.SetSize(int_face_restrict_lex_trial->Height(),
                                Device::GetDeviceMemoryType());
         int_face_trial.UseDevice(true);
      }
      if (int_face_restrict_lex_test == nullptr)
      {
         int_face_restrict_lex_test = test_fes->GetFaceRestriction(
                                         ElementDofOrdering::LEXICOGRAPHIC,
                                         FaceType::Interior);
         int_face_test.SetSize(int_face_restrict_lex_test->Height(),
                               Device::GetDeviceMemoryType());
         int_face_test.UseDevice(true);
      }
   }

   if (a->GetBTFBFI()->Size() > 0 || a->GetBBFI()->Size() > 0)
   {
      if (bdr_face_restrict_lex_trial == nullptr)
      {
         bdr_face_restrict_lex_trial = trial_fes->GetFaceRestriction(
                                          ElementDofOrdering::LEXICOGRAPHIC,
                                          FaceType::Boundary,
                                          m);
         bdr_face_trial.SetSize(bdr_face_restrict_lex_trial->Height(),
                                Device::GetDeviceMemoryType());
         bdr_face_trial.UseDevice(true);
      }
      if (bdr_face_restrict_lex_test == nullptr)
      {
         bdr_face_restrict_lex_test = test_fes->GetFaceRestriction(
                                         ElementDofOrdering::LEXICOGRAPHIC,
                                         FaceType::Boundary,
                                         m);
         bdr_face_test.SetSize(bdr_face_restrict_lex_test->Height(),
                               Device::GetDeviceMemoryType());
         bdr_face_test.UseDevice(true);
      }
   }
}

void MFMixedBilinearFormExtension::Assemble()
{
   SetupRestrictionOperators(L2FaceValues::DoubleValued);

   Array<BilinearFormIntegrator *> &integrators = *a->GetDBFI();
   for (BilinearFormIntegrator *integ : integrators)
   {
      integ->AssembleMF(*trial_fes, *test_fes);
   }

   Array<BilinearFormIntegrator *> &bdr_integrators = *a->GetBBFI();
   for (BilinearFormIntegrator *integ : bdr_integrators)
   {
      integ->AssembleMFBoundary(*trial_fes, *test_fes);
   }

   MFEM_VERIFY(a->GetTFBFI()->Size() == 0, "AddInteriorFaceIntegrator is not "
               "currently supported in MFMixedBilinearFormExtension");

   MFEM_VERIFY(a->GetBTFBFI()->Size() == 0, "AddBdrFaceIntegrator is not "
               "currently supported in MFMixedBilinearFormExtension");
}

void MFMixedBilinearFormExtension::Mult(const Vector &x, Vector &y) const
{
   y = 0.0;
   AddMult(x, y);
}

void MFMixedBilinearFormExtension::AddMult(const Vector &x, Vector &y,
                                           const double c) const
{
   Array<BilinearFormIntegrator *> &integrators = *a->GetDBFI();
   if (integrators.Size() > 0)
   {
      if (elem_restrict_trial)
      {
         elem_restrict_trial->Mult(x, local_trial);
      }
      if (elem_restrict_test)
      {
         local_test = 0.0;
         for (BilinearFormIntegrator *integ : integrators)
         {
            integ->AddMultMF(elem_restrict_trial ? local_trial : x, local_test);
         }
         if (c != 1.0)
         {
            local_test *= c;
         }
         elem_restrict_test->AddMultTranspose(local_test, y);
      }
      else
      {
         y.UseDevice(true); // typically this is a large vector, so store on device
         if (c != 1.0)
         {
            temp_test.SetSize(y.Size());
            temp_test.UseDevice(true);
            temp_test = 0.0;
            for (BilinearFormIntegrator *integ : integrators)
            {
               integ->AddMultMF(elem_restrict_trial ? local_trial : x, temp_test);
            }
            y.Add(c, temp_test);
         }
         else
         {
            for (BilinearFormIntegrator *integ : integrators)
            {
               integ->AddMultMF(elem_restrict_trial ? local_trial : x, y);
            }
         }
      }
   }

   Array<BilinearFormIntegrator *> &bdr_integrators = *a->GetBBFI();
   if (bdr_integrators.Size() > 0)
   {
      if (bdr_face_restrict_lex_trial)
      {
         bdr_face_restrict_lex_trial->Mult(x, bdr_face_trial);
      }
      if (bdr_face_restrict_lex_test)
      {
         bdr_face_test = 0.0;
         for (BilinearFormIntegrator *integ : bdr_integrators)
         {
            integ->AddMultMF(bdr_face_restrict_lex_trial ? bdr_face_trial : x,
                             bdr_face_test);
         }
         if (c != 1.0)
         {
            bdr_face_test *= c;
         }
         bdr_face_restrict_lex_test->AddMultTranspose(bdr_face_test, y);
      }
      else
      {
         if (c != 1.0)
         {
            temp_test.SetSize(y.Size());
            temp_test.UseDevice(true);
            temp_test = 0.0;
            for (BilinearFormIntegrator *integ : bdr_integrators)
            {
               integ->AddMultMF(bdr_face_restrict_lex_trial ? bdr_face_trial : x,
                                temp_test);
            }
            y.Add(c, temp_test);
         }
         else
         {
            for (BilinearFormIntegrator *integ : bdr_integrators)
            {
               integ->AddMultMF(bdr_face_restrict_lex_trial ? bdr_face_trial : x, y);
            }
         }
      }
   }
}

void MFMixedBilinearFormExtension::MultTranspose(const Vector &x,
                                                 Vector &y) const
{
   y = 0.0;
   AddMultTranspose(x, y);
}

void MFMixedBilinearFormExtension::AddMultTranspose(const Vector &x, Vector &y,
                                                    const double c) const
{
   Array<BilinearFormIntegrator *> &integrators = *a->GetDBFI();
   if (integrators.Size() > 0)
   {
      if (elem_restrict_test)
      {
         elem_restrict_test->Mult(x, local_test);
      }
      if (elem_restrict_trial)
      {
         local_trial = 0.0;
         for (BilinearFormIntegrator *integ : integrators)
         {
            integ->AddMultTransposeMF(elem_restrict_test ? local_test : x,
                                      local_trial);
         }
         if (c != 1.0)
         {
            local_trial *= c;
         }
         elem_restrict_trial->AddMultTranspose(local_trial, y);
      }
      else
      {
         y.UseDevice(true); // typically this is a large vector, so store on device
         if (c != 1.0)
         {
            temp_trial.SetSize(y.Size());
            temp_trial.UseDevice(true);
            temp_trial = 0.0;
            for (BilinearFormIntegrator *integ : integrators)
            {
               integ->AddMultTransposeMF(elem_restrict_test ? local_test : x,
                                         temp_trial);
            }
            y.Add(c, temp_trial);
         }
         else
         {
            for (BilinearFormIntegrator *integ : integrators)
            {
               integ->AddMultTransposeMF(elem_restrict_test ? local_test : x, y);
            }
         }
      }
   }

   Array<BilinearFormIntegrator *> &bdr_integrators = *a->GetBBFI();
   if (bdr_integrators.Size() > 0)
   {
      if (bdr_face_restrict_lex_test)
      {
         bdr_face_restrict_lex_test->Mult(x, bdr_face_test);
      }
      if (bdr_face_restrict_lex_trial)
      {
         bdr_face_trial = 0.0;
         for (BilinearFormIntegrator *integ : bdr_integrators)
         {
            integ->AddMultTransposeMF(bdr_face_restrict_lex_test ? bdr_face_test : x,
                                      bdr_face_trial);
         }
         if (c != 1.0)
         {
            bdr_face_trial *= c;
         }
         bdr_face_restrict_lex_trial->AddMultTranspose(bdr_face_trial, y);
      }
      else
      {
         y.UseDevice(true); // typically this is a large vector, so store on device
         if (c != 1.0)
         {
            temp_trial.SetSize(y.Size());
            temp_trial.UseDevice(true);
            temp_trial = 0.0;
            for (BilinearFormIntegrator *integ : bdr_integrators)
            {
               integ->AddMultTransposeMF(bdr_face_restrict_lex_test ? bdr_face_test : x,
                                         temp_trial);
            }
            y.Add(c, temp_trial);
         }
         else
         {
            for (BilinearFormIntegrator *integ : integrators)
            {
               integ->AddMultTransposeMF(bdr_face_restrict_lex_test ? bdr_face_test : x, y);
            }
         }
      }
   }
}

void MFMixedBilinearFormExtension::Update()
{
   trial_fes = a->TrialFESpace();
   test_fes  = a->TestFESpace();
   height = test_fes->GetVSize();
   width  = trial_fes->GetVSize();

   elem_restrict_trial = nullptr;
   elem_restrict_test = nullptr;
   int_face_restrict_lex_trial = nullptr;
   int_face_restrict_lex_test = nullptr;
   bdr_face_restrict_lex_trial = nullptr;
   bdr_face_restrict_lex_test = nullptr;
}

/// Data and methods for partially-assembled mixed bilinear forms
PAMixedBilinearFormExtension::PAMixedBilinearFormExtension(
   MixedBilinearForm *form)
   : MFMixedBilinearFormExtension(form)
{
}

void PAMixedBilinearFormExtension::Assemble()
{
   SetupRestrictionOperators(L2FaceValues::DoubleValued);

   Array<BilinearFormIntegrator *> &integrators = *a->GetDBFI();
   for (BilinearFormIntegrator *integ : integrators)
   {
      integ->AssemblePA(*trial_fes, *test_fes);
   }

   Array<BilinearFormIntegrator *> &bdr_integrators = *a->GetBBFI();
   for (BilinearFormIntegrator *integ : bdr_integrators)
   {
      integ->AssemblePABoundary(*trial_fes, *test_fes);
   }

   MFEM_VERIFY(a->GetTFBFI()->Size() == 0, "AddInteriorFaceIntegrator is not "
               "currently supported in PAMixedBilinearFormExtension");

   MFEM_VERIFY(a->GetBTFBFI()->Size() == 0, "AddBdrFaceIntegrator is not "
               "currently supported in PAMixedBilinearFormExtension");
}

void PAMixedBilinearFormExtension::AssembleDiagonal_ADAt(const Vector &D,
                                                         Vector &diag) const
{
   Array<BilinearFormIntegrator *> &integrators = *a->GetDBFI();
   if (integrators.Size() > 0)
   {
      if (elem_restrict_trial)
      {
         elem_restrict_trial->MultUnsigned(D, local_trial);
      }
      if (elem_restrict_test)
      {
         local_test = 0.0;
         for (BilinearFormIntegrator *integ : integrators)
         {
            integ->AssembleDiagonalPA_ADAt(elem_restrict_trial ? local_trial : D,
                                           local_test);
         }
         elem_restrict_test->MultTransposeUnsigned(local_test, diag);
      }
   }
   else
   {
      diag.UseDevice(true); // typically this is a large vector, so store on device
      diag = 0.0;
      for (BilinearFormIntegrator *integ : integrators)
      {
         integ->AssembleDiagonalPA_ADAt(elem_restrict_trial ? local_trial : D, diag);
      }
   }

   Array<BilinearFormIntegrator *> &bdr_integrators = *a->GetBBFI();
   if (bdr_integrators.Size() > 0)
   {
      if (bdr_face_restrict_lex_trial)
      {
         bdr_face_restrict_lex_trial->MultUnsigned(D, bdr_face_trial);
      }
      if (bdr_face_restrict_lex_test)
      {
         bdr_face_test = 0.0;
         for (BilinearFormIntegrator *integ : bdr_integrators)
         {
            integ->AssembleDiagonalPA_ADAt(bdr_face_restrict_lex_trial ? bdr_face_trial : D,
                                           bdr_face_test);
         }
         bdr_face_restrict_lex_test->AddMultTransposeUnsigned(bdr_face_test, diag);
      }
      else
      {
         for (BilinearFormIntegrator *integ : bdr_integrators)
         {
            integ->AssembleDiagonalPA_ADAt(bdr_face_restrict_lex_trial ? bdr_face_trial : D,
                                           diag);
         }
      }
   }
}

void PAMixedBilinearFormExtension::AddMult(const Vector &x, Vector &y,
                                           const double c) const
{
   Array<BilinearFormIntegrator *> &integrators = *a->GetDBFI();
   if (integrators.Size() > 0)
   {
      if (elem_restrict_trial)
      {
         elem_restrict_trial->Mult(x, local_trial);
      }
      if (elem_restrict_test)
      {
         local_test = 0.0;
         for (BilinearFormIntegrator *integ : integrators)
         {
            integ->AddMultPA(elem_restrict_trial ? local_trial : x, local_test);
         }
         if (c != 1.0)
         {
            local_test *= c;
         }
         elem_restrict_test->AddMultTranspose(local_test, y);
      }
      else
      {
         y.UseDevice(true); // typically this is a large vector, so store on device
         if (c != 1.0)
         {
            temp_test.SetSize(y.Size());
            temp_test.UseDevice(true);
            temp_test = 0.0;
            for (BilinearFormIntegrator *integ : integrators)
            {
               integ->AddMultPA(elem_restrict_trial ? local_trial : x, temp_test);
            }
            y.Add(c, temp_test);
         }
         else
         {
            for (BilinearFormIntegrator *integ : integrators)
            {
               integ->AddMultPA(elem_restrict_trial ? local_trial : x, y);
            }
         }
      }
   }

   Array<BilinearFormIntegrator *> &bdr_integrators = *a->GetBBFI();
   if (bdr_integrators.Size() > 0)
   {
      if (bdr_face_restrict_lex_trial)
      {
         bdr_face_restrict_lex_trial->Mult(x, bdr_face_trial);
      }
      if (bdr_face_restrict_lex_test)
      {
         bdr_face_test = 0.0;
         for (BilinearFormIntegrator *integ : bdr_integrators)
         {
            integ->AddMultPA(bdr_face_restrict_lex_trial ? bdr_face_trial : x,
                             bdr_face_test);
         }
         if (c != 1.0)
         {
            bdr_face_test *= c;
         }
         bdr_face_restrict_lex_test->AddMultTranspose(bdr_face_test, y);
      }
      else
      {
         y.UseDevice(true); // typically this is a large vector, so store on device
         if (c != 1.0)
         {
            temp_test.SetSize(y.Size());
            temp_test.UseDevice(true);
            temp_test = 0.0;
            for (BilinearFormIntegrator *integ : bdr_integrators)
            {
               integ->AddMultPA(bdr_face_restrict_lex_trial ? bdr_face_trial : x,
                                temp_test);
            }
            y.Add(c, temp_test);
         }
         else
         {
            for (BilinearFormIntegrator *integ : bdr_integrators)
            {
               integ->AddMultPA(bdr_face_restrict_lex_trial ? bdr_face_trial : x, y);
            }
         }
      }
   }
}

void PAMixedBilinearFormExtension::AddMultTranspose(const Vector &x, Vector &y,
                                                    const double c) const
{
   Array<BilinearFormIntegrator *> &integrators = *a->GetDBFI();
   if (integrators.Size() > 0)
   {
      if (elem_restrict_test)
      {
         elem_restrict_test->Mult(x, local_test);
      }
      if (elem_restrict_trial)
      {
         local_trial = 0.0;
         for (BilinearFormIntegrator *integ : integrators)
         {
            integ->AddMultTransposePA(elem_restrict_test ? local_test : x,
                                      local_trial);
         }
         if (c != 1.0)
         {
            local_trial *= c;
         }
         elem_restrict_trial->AddMultTranspose(local_trial, y);
      }
      else
      {
         y.UseDevice(true); // typically this is a large vector, so store on device
         if (c != 1.0)
         {
            temp_trial.SetSize(y.Size());
            temp_trial.UseDevice(true);
            temp_trial = 0.0;
            for (BilinearFormIntegrator *integ : integrators)
            {
               integ->AddMultTransposePA(elem_restrict_test ? local_test : x,
                                         temp_trial);
            }
            y.Add(c, temp_trial);
         }
         else
         {
            for (BilinearFormIntegrator *integ : integrators)
            {
               integ->AddMultTransposePA(elem_restrict_test ? local_test : x, y);
            }
         }
      }
   }

   Array<BilinearFormIntegrator *> &bdr_integrators = *a->GetBBFI();
   if (bdr_integrators.Size() > 0)
   {
      if (bdr_face_restrict_lex_test)
      {
         bdr_face_restrict_lex_test->Mult(x, bdr_face_test);
      }
      if (bdr_face_restrict_lex_trial)
      {
         bdr_face_trial = 0.0;
         for (BilinearFormIntegrator *integ : bdr_integrators)
         {
            integ->AddMultTransposePA(bdr_face_restrict_lex_test ? bdr_face_test : x,
                                      bdr_face_trial);
         }
         if (c != 1.0)
         {
            bdr_face_trial *= c;
         }
         bdr_face_restrict_lex_trial->AddMultTranspose(bdr_face_trial, y);
      }
      else
      {
         y.UseDevice(true); // typically this is a large vector, so store on device
         if (c != 1.0)
         {
            temp_trial.SetSize(y.Size());
            temp_trial.UseDevice(true);
            temp_trial = 0.0;
            for (BilinearFormIntegrator *integ : bdr_integrators)
            {
               integ->AddMultTransposePA(bdr_face_restrict_lex_test ? bdr_face_test : x,
                                         temp_trial);
            }
            y.Add(c, temp_trial);
         }
         else
         {
            for (BilinearFormIntegrator *integ : integrators)
            {
               integ->AddMultTransposePA(bdr_face_restrict_lex_test ? bdr_face_test : x, y);
            }
         }
      }
   }
}

/// Data and methods for partially-assembled discrete linear operators
PADiscreteLinearOperatorExtension::PADiscreteLinearOperatorExtension(
   DiscreteLinearOperator *linop) :
   PAMixedBilinearFormExtension(linop)
{
}

void PADiscreteLinearOperatorExtension::Assemble()
{
   PAMixedBilinearFormExtension::Assemble();

   // Construct element vdof multiplicity (avoid use of elem_restrict_test
   // because it might not exist for libCEED)
   test_multiplicity.SetSize(height);
   test_multiplicity.UseDevice(true);
   test_multiplicity = 0.0;
   Array<int> dofs;
   auto d_mult = test_multiplicity.HostReadWrite();
   for (int i = 0; i < test_fes->GetNE(); i++)
   {
      test_fes->GetElementVDofs(i, dofs);
      for (int j = 0; j < dofs.Size(); j++)
      {
         const int k = dofs[j];
         d_mult[(k >= 0) ? k : -1 - k] += 1.0;
      }
   }
   test_multiplicity.Reciprocal();


   // //XX TODO DEBUG
   // std::cout << "\nINV MULTIPLICITY:\n\n";
   // test_multiplicity.Print();

}

void PADiscreteLinearOperatorExtension::Mult(const Vector &x, Vector &y) const
{
   Array<BilinearFormIntegrator *> &interpolators = *a->GetDBFI();
   if (elem_restrict_trial)
   {
      elem_restrict_trial->Mult(x, local_trial);
   }
   if (elem_restrict_test)
   {
      local_test = 0.0;
      for (BilinearFormIntegrator *interp : interpolators)
      {
         interp->AddMultPA(elem_restrict_trial ? local_trial : x, local_test);
      }
      elem_restrict_test->MultTranspose(local_test, y);
   }
   else
   {
      y = 0.0;
      for (BilinearFormIntegrator *interp : interpolators)
      {
         interp->AddMultPA(elem_restrict_trial ? local_trial : x, y);
      }
   }
   y *= test_multiplicity;
}

void PADiscreteLinearOperatorExtension::AddMult(const Vector &x, Vector &y,
                                                const double c) const
{
   Array<BilinearFormIntegrator *> &interpolators = *a->GetDBFI();
   temp_test.SetSize(y.Size());
   temp_test.UseDevice(true);
   if (elem_restrict_trial)
   {
      elem_restrict_trial->Mult(x, local_trial);
   }
   if (elem_restrict_test)
   {
      local_test = 0.0;
      for (BilinearFormIntegrator *interp : interpolators)
      {
         interp->AddMultPA(elem_restrict_trial ? local_trial : x, local_test);
      }
      elem_restrict_test->MultTranspose(local_test, temp_test);
   }
   else
   {
      temp_test = 0.0;
      for (BilinearFormIntegrator *interp : interpolators)
      {
         interp->AddMultPA(elem_restrict_trial ? local_trial : x, temp_test);
      }
   }
   temp_test *= test_multiplicity;
   y.Add(c, temp_test);
}

void PADiscreteLinearOperatorExtension::AddMultTranspose(const Vector &x,
                                                         Vector &y,
                                                         const double c) const
{
   MFEM_VERIFY(c == 1.0,
               "General coefficient case for PADiscreteLinearOperatorExtension::"
               "AddMultTranspose is not yet supported!");
   Array<BilinearFormIntegrator *> &interpolators = *a->GetDBFI();
   temp_test = x;
   temp_test *= test_multiplicity;
   if (elem_restrict_test)
   {
      elem_restrict_test->Mult(temp_test, local_test);
   }
   if (elem_restrict_trial)
   {
      local_trial = 0.0;
      for (BilinearFormIntegrator *interp : interpolators)
      {
         interp->AddMultTransposePA(elem_restrict_test ? local_test : temp_test,
                                    local_trial);
      }
      elem_restrict_trial->AddMultTranspose(local_trial, y);
   }
   else
   {
      y.UseDevice(true); // typically this is a large vector, so store on device
      for (BilinearFormIntegrator *interp : interpolators)
      {
         interp->AddMultTransposePA(elem_restrict_test ? local_test : temp_test, y);
      }
   }
}

} // namespace mfem
