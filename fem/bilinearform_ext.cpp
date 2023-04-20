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

/// Data and methods for partially-assembled bilinear forms
PABilinearFormExtension::PABilinearFormExtension(BilinearForm *form)
   : BilinearFormExtension(form)
{
   Update();
}

void PABilinearFormExtension::SetupRestrictionOperators(const L2FaceValues m)
{
   if (DeviceCanUseCeed()) { return; }
   ElementDofOrdering ordering = UsesTensorBasis(*fes) ?
                                 ElementDofOrdering::LEXICOGRAPHIC :
                                 ElementDofOrdering::NATIVE;
   elem_restrict = fes->GetElementRestriction(ordering);
   if (elem_restrict)
   {
      localX.SetSize(elem_restrict->Height(), Device::GetDeviceMemoryType());
      localY.SetSize(elem_restrict->Height(), Device::GetDeviceMemoryType());
      localY.UseDevice(true); // ensure 'localY = 0.0' is done on device
   }

   // Construct face restriction operators only if the bilinear form has
   // interior or boundary face integrators
   if (int_face_restrict_lex == nullptr && a->GetFBFI()->Size() > 0)
   {
      int_face_restrict_lex = fes->GetFaceRestriction(
                                 ElementDofOrdering::LEXICOGRAPHIC,
                                 FaceType::Interior);
      int_face_X.SetSize(int_face_restrict_lex->Height(), Device::GetMemoryType());
      int_face_Y.SetSize(int_face_restrict_lex->Height(), Device::GetMemoryType());
      int_face_Y.UseDevice(true); // ensure 'int_face_Y = 0.0' is done on device
   }

   const bool has_bdr_integs = (a->GetBFBFI()->Size() > 0 ||
                                a->GetBBFI()->Size() > 0);
   if (bdr_face_restrict_lex == nullptr && has_bdr_integs)
   {
      bdr_face_restrict_lex = fes->GetFaceRestriction(
                                 ElementDofOrdering::LEXICOGRAPHIC,
                                 FaceType::Boundary,
                                 m);
      bdr_face_X.SetSize(bdr_face_restrict_lex->Height(), Device::GetMemoryType());
      bdr_face_Y.SetSize(bdr_face_restrict_lex->Height(), Device::GetMemoryType());
      bdr_face_Y.UseDevice(true); // ensure 'faceBoundY = 0.0' is done on device
   }
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

void PABilinearFormExtension::AssembleDiagonal(Vector &y) const
{
   Array<BilinearFormIntegrator *> &integrators = *a->GetDBFI();
   if (elem_restrict)
   {
      if (integrators.Size() > 0)
      {
         localY = 0.0;
         for (BilinearFormIntegrator *integ : integrators)
         {
            integ->AssembleDiagonalPA(localY);
         }
         const ElementRestriction *H1elem_restrict =
            dynamic_cast<const ElementRestriction *>(elem_restrict);
         if (H1elem_restrict)
         {
            H1elem_restrict->MultTransposeUnsigned(localY, y);
         }
         else
         {
            elem_restrict->MultTranspose(localY, y);
         }
      }
      else
      {
         y = 0.0;
      }
   }
   else
   {
      y.UseDevice(true); // typically this is a large vector, so store on device
      y = 0.0;
      for (BilinearFormIntegrator *integ : integrators)
      {
         integ->AssembleDiagonalPA(y);
      }
   }

   Array<BilinearFormIntegrator *> &bdr_integrators = *a->GetBBFI();
   if (bdr_face_restrict_lex)
   {
      if (bdr_integrators.Size() > 0)
      {
         bdr_face_Y = 0.0;
         for (BilinearFormIntegrator *integ : bdr_integrators)
         {
            integ->AssembleDiagonalPA(bdr_face_Y);
         }
         bdr_face_restrict_lex->AddMultTransposeUnsigned(bdr_face_Y, y);
      }
   }
   else
   {
      for (BilinearFormIntegrator *integ : bdr_integrators)
      {
         integ->AssembleDiagonalPA(y);
      }
   }
}

void PABilinearFormExtension::Update()
{
   fes = a->FESpace();
   height = width = fes->GetVSize();

   elem_restrict = nullptr;
   int_face_restrict_lex = nullptr;
   bdr_face_restrict_lex = nullptr;
}

void PABilinearFormExtension::FormSystemMatrix(const Array<int> &ess_tdof_list,
                                               OperatorHandle &A)
{
   Operator *oper;
   Operator::FormSystemOperator(ess_tdof_list, oper);
   A.Reset(oper); // A will own oper
}

void PABilinearFormExtension::FormLinearSystem(const Array<int> &ess_tdof_list,
                                               Vector &x, Vector &b,
                                               OperatorHandle &A,
                                               Vector &X, Vector &B,
                                               int copy_interior)
{
   Operator *oper;
   Operator::FormLinearSystem(ess_tdof_list, x, b, oper, X, B, copy_interior);
   A.Reset(oper); // A will own oper
}

void PABilinearFormExtension::Mult(const Vector &x, Vector &y) const
{
   Array<BilinearFormIntegrator *> &integrators = *a->GetDBFI();
   if (elem_restrict)
   {
      if (integrators.Size() > 0)
      {
         elem_restrict->Mult(x, localX);
         localY = 0.0;
         for (BilinearFormIntegrator *integ : integrators)
         {
            integ->AddMultPA(localX, localY);
         }
         elem_restrict->MultTranspose(localY, y);
      }
      else
      {
         y = 0.0;
      }
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
   if (int_face_restrict_lex)
   {
      if (int_face_integrators.Size() > 0)
      {
         int_face_restrict_lex->Mult(x, int_face_X);
         if (int_face_X.Size() > 0)
         {
            int_face_Y = 0.0;
            for (BilinearFormIntegrator *integ : int_face_integrators)
            {
               integ->AddMultPA(int_face_X, int_face_Y);
            }
            int_face_restrict_lex->AddMultTransposeInPlace(int_face_Y, y);
         }
      }
   }
   else
   {
      for (BilinearFormIntegrator *integ : int_face_integrators)
      {
         integ->AddMultPA(x, y);
      }
   }

   Array<BilinearFormIntegrator *> &bdr_integrators = *a->GetBBFI();
   Array<BilinearFormIntegrator *> &bdr_face_integrators = *a->GetBFBFI();
   if (bdr_face_restrict_lex)
   {
      if (bdr_integrators.Size() > 0 || bdr_face_integrators.Size() > 0)
      {
         bdr_face_restrict_lex->Mult(x, bdr_face_X);
         if (bdr_face_X.Size() > 0)
         {
            bdr_face_Y = 0.0;
            for (BilinearFormIntegrator *integ : bdr_integrators)
            {
               integ->AddMultPA(bdr_face_X, bdr_face_Y);
            }
            for (BilinearFormIntegrator *integ : bdr_face_integrators)
            {
               integ->AddMultPA(bdr_face_X, bdr_face_Y);
            }
            bdr_face_restrict_lex->AddMultTransposeInPlace(bdr_face_Y, y);
         }
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

void PABilinearFormExtension::MultTranspose(const Vector &x, Vector &y) const
{
   Array<BilinearFormIntegrator *> &integrators = *a->GetDBFI();
   if (elem_restrict)
   {
      if (integrators.Size() > 0)
      {
         elem_restrict->Mult(x, localX);
         localY = 0.0;
         for (BilinearFormIntegrator *integ : integrators)
         {
            integ->AddMultTransposePA(localX, localY);
         }
         elem_restrict->MultTranspose(localY, y);
      }
      else
      {
         y = 0.0;
      }
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
   if (int_face_restrict_lex)
   {
      if (int_face_integrators.Size() > 0)
      {
         int_face_restrict_lex->Mult(x, int_face_X);
         if (int_face_X.Size() > 0)
         {
            int_face_Y = 0.0;
            for (BilinearFormIntegrator *integ : int_face_integrators)
            {
               integ->AddMultTransposePA(int_face_X, int_face_Y);
            }
            int_face_restrict_lex->AddMultTransposeInPlace(int_face_Y, y);
         }
      }
   }
   else
   {
      for (BilinearFormIntegrator *integ : int_face_integrators)
      {
         integ->AddMultTransposePA(x, y);
      }
   }

   Array<BilinearFormIntegrator *> &bdr_integrators = *a->GetBBFI();
   Array<BilinearFormIntegrator *> &bdr_face_integrators = *a->GetBFBFI();
   if (bdr_face_restrict_lex)
   {
      if (bdr_integrators.Size() > 0 || bdr_face_integrators.Size() > 0)
      {
         bdr_face_restrict_lex->Mult(x, bdr_face_X);
         if (bdr_face_X.Size() > 0)
         {
            bdr_face_Y = 0.0;
            for (BilinearFormIntegrator *integ : bdr_integrators)
            {
               integ->AddMultTransposePA(bdr_face_X, bdr_face_Y);
            }
            for (BilinearFormIntegrator *integ : bdr_face_integrators)
            {
               integ->AddMultTransposePA(bdr_face_X, bdr_face_Y);
            }
            bdr_face_restrict_lex->AddMultTransposeInPlace(bdr_face_Y, y);
         }
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

/// Data and methods for element-assembled bilinear forms
EABilinearFormExtension::EABilinearFormExtension(BilinearForm *form)
   : PABilinearFormExtension(form)
{
   factorize_face_terms = (fes->IsDGSpace() && fes->Conforming());
}

void EABilinearFormExtension::Assemble()
{
   SetupRestrictionOperators(L2FaceValues::SingleValued);

   ne = fes->GetMesh()->GetNE();
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
   auto Apply = [](const int ne, const int ndofs, const Vector &data,
                   const Vector &x, Vector &y)
   {
      auto X = Reshape(x.Read(), ndofs, ne);
      auto Y = Reshape(y.ReadWrite(), ndofs, ne);
      auto A = Reshape(data.Read(), ndofs, ndofs, ne);
      mfem::forall(ne * ndofs, [=] MFEM_HOST_DEVICE (int k)
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
   if (elem_restrict)
   {
      if (integrators.Size() > 0)
      {
         elem_restrict->Mult(x, localX);
         localY = 0.0;
         Apply(ne, elem_dofs, ea_data, localX, localY);
         elem_restrict->MultTranspose(localY, y);
      }
      else
      {
         y = 0.0;
      }
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
   auto ApplyIntFace = [](const int ne, const int ndofs, const Vector &data,
                          const Vector &x, Vector &y)
   {
      auto X = Reshape(x.Read(), ndofs, 2, ne);
      auto Y = Reshape(y.ReadWrite(), ndofs, 2, ne);
      auto A = Reshape(data.Read(), ndofs, ndofs, 2, ne);
      mfem::forall(ne * ndofs, [=] MFEM_HOST_DEVICE (int k)
      {
         const int e = k / ndofs;
         const int j = k % ndofs;
         double res = 0.0;
         for (int i = 0; i < ndofs; i++)
         {
            res += A(i, j, 0, e) * X(i, 0, e);
         }
         Y(j, 0, e) += res;
         res = 0.0;
         for (int i = 0; i < ndofs; i++)
         {
            res += A(i, j, 1, e) * X(i, 1, e);
         }
         Y(j, 1, e) += res;
      });
   };
   auto ApplyExtFace = [](const int ne, const int ndofs, const Vector &data,
                          const Vector &x, Vector &y)
   {
      auto X = Reshape(x.Read(), ndofs, 2, ne);
      auto Y = Reshape(y.ReadWrite(), ndofs, 2, ne);
      auto A = Reshape(data.Read(), ndofs, ndofs, 2, ne);
      mfem::forall(ne * ndofs, [=] MFEM_HOST_DEVICE (int k)
      {
         const int e = k / ndofs;
         const int j = k % ndofs;
         double res = 0.0;
         for (int i = 0; i < ndofs; i++)
         {
            res += A(i, j, 1, e) * X(i, 0, e);
         }
         Y(j, 1, e) += res;
         res = 0.0;
         for (int i = 0; i < ndofs; i++)
         {
            res += A(i, j, 0, e) * X(i, 1, e);
         }
         Y(j, 0, e) += res;
      });
   };
   if (int_face_restrict_lex && int_face_integrators.Size() > 0)
   {
      int_face_restrict_lex->Mult(x, int_face_X);
      if (int_face_X.Size() > 0)
      {
         int_face_Y = 0.0;
         if (!factorize_face_terms)
         {
            ApplyIntFace(nf_int, face_dofs, ea_data_int, int_face_X, int_face_Y);
         }
         ApplyExtFace(nf_int, face_dofs, ea_data_ext, int_face_X, int_face_Y);
         int_face_restrict_lex->AddMultTransposeInPlace(int_face_Y, y);
      }
   }

   // Treatment of boundary faces
   Array<BilinearFormIntegrator *> &bdr_face_integrators = *a->GetBFBFI();
   if (!factorize_face_terms && bdr_face_restrict_lex &&
       bdr_face_integrators.Size() > 0)
   {
      bdr_face_restrict_lex->Mult(x, bdr_face_X);
      if (bdr_face_X.Size() > 0)
      {
         bdr_face_Y = 0.0;
         Apply(nf_bdr, face_dofs, ea_data_bdr, bdr_face_X, bdr_face_Y);
         bdr_face_restrict_lex->AddMultTransposeInPlace(bdr_face_Y, y);
      }
   }
}

void EABilinearFormExtension::MultTranspose(const Vector &x, Vector &y) const
{
   Array<BilinearFormIntegrator *> &integrators = *a->GetDBFI();
   auto ApplyTranspose = [](const int ne, const int ndofs, const Vector &data,
                            const Vector &x, Vector &y)
   {
      auto X = Reshape(x.Read(), ndofs, ne);
      auto Y = Reshape(y.ReadWrite(), ndofs, ne);
      auto A = Reshape(data.Read(), ndofs, ndofs, ne);
      mfem::forall(ne * ndofs, [=] MFEM_HOST_DEVICE (int k)
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
   if (elem_restrict)
   {
      if (integrators.Size() > 0)
      {
         elem_restrict->Mult(x, localX);
         localY = 0.0;
         ApplyTranspose(ne, elem_dofs, ea_data, localX, localY);
         elem_restrict->MultTranspose(localY, y);
      }
      else
      {
         y = 0.0;
      }
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
   auto ApplyIntFaceTranspose = [](const int ne, const int ndofs,
                                   const Vector &data, const Vector &x, Vector &y)
   {
      auto X = Reshape(x.Read(), ndofs, 2, ne);
      auto Y = Reshape(y.ReadWrite(), ndofs, 2, ne);
      auto A = Reshape(data.Read(), ndofs, ndofs, 2, ne);
      mfem::forall(ne * ndofs, [=] MFEM_HOST_DEVICE (int k)
      {
         const int e = k / ndofs;
         const int j = k % ndofs;
         double res = 0.0;
         for (int i = 0; i < ndofs; i++)
         {
            res += A(j, i, 0, e) * X(i, 0, e);
         }
         Y(j, 0, e) += res;
         res = 0.0;
         for (int i = 0; i < ndofs; i++)
         {
            res += A(j, i, 1, e) * X(i, 1, e);
         }
         Y(j, 1, e) += res;
      });
   };
   auto ApplyExtFaceTranspose = [](const int ne, const int ndofs,
                                   const Vector &data, const Vector &x, Vector &y)
   {
      auto X = Reshape(x.Read(), ndofs, 2, ne);
      auto Y = Reshape(y.ReadWrite(), ndofs, 2, ne);
      auto A = Reshape(data.Read(), ndofs, ndofs, 2, ne);
      mfem::forall(ne * ndofs, [=] MFEM_HOST_DEVICE (int k)
      {
         const int e = k / ndofs;
         const int j = k % ndofs;
         double res = 0.0;
         for (int i = 0; i < ndofs; i++)
         {
            res += A(j, i, 1, e) * X(i, 0, e);
         }
         Y(j, 1, e) += res;
         res = 0.0;
         for (int i = 0; i < ndofs; i++)
         {
            res += A(j, i, 0, e) * X(i, 1, e);
         }
         Y(j, 0, e) += res;
      });
   };
   if (int_face_restrict_lex && int_face_integrators.Size() > 0)
   {
      int_face_restrict_lex->Mult(x, int_face_X);
      if (int_face_X.Size() > 0)
      {
         int_face_Y = 0.0;
         if (!factorize_face_terms)
         {
            ApplyIntFaceTranspose(nf_int, face_dofs, ea_data_int, int_face_X, int_face_Y);
         }
         ApplyExtFaceTranspose(nf_int, face_dofs, ea_data_ext, int_face_X, int_face_Y);
         int_face_restrict_lex->AddMultTransposeInPlace(int_face_Y, y);
      }
   }

   // Treatment of boundary faces
   Array<BilinearFormIntegrator *> &bdr_face_integrators = *a->GetBFBFI();
   if (!factorize_face_terms && bdr_face_restrict_lex &&
       bdr_face_integrators.Size() > 0)
   {
      bdr_face_restrict_lex->Mult(x, bdr_face_X);
      if (bdr_face_X.Size() > 0)
      {
         bdr_face_Y = 0.0;
         ApplyTranspose(nf_bdr, face_dofs, ea_data_bdr, bdr_face_X, bdr_face_Y);
         bdr_face_restrict_lex->AddMultTransposeInPlace(bdr_face_Y, y);
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
         const L2ElementRestriction *restE =
            static_cast<const L2ElementRestriction *>(elem_restrict);
         const L2FaceRestriction *restF =
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
         const ElementRestriction &rest =
            static_cast<const ElementRestriction&>(*elem_restrict);
         rest.FillJAndData(ea_data, *mat);
      }
   }
   else // We create, compute the sparsity, and fill the sparse matrix
   {
      mat = new SparseMatrix;
      mat->OverrideSize(height, width);
      if (fes->IsDGSpace())
      {
         const L2ElementRestriction *restE =
            static_cast<const L2ElementRestriction *>(elem_restrict);
         const L2FaceRestriction *restF =
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
      else // Continuous Galerkin case
      {
         const ElementRestriction &rest =
            static_cast<const ElementRestriction &>(*elem_restrict);
         rest.FillSparseMatrix(ea_data, *mat);
      }
      a->mat = mat;
   }
   if (a->sort_sparse_matrix)
   {
      a->mat->SortColumnIndices();
   }
}

void FABilinearFormExtension::RAP(OperatorHandle &A)
{
#ifdef MFEM_USE_MPI
   if (auto pa = dynamic_cast<ParBilinearForm* >(a))
   {
      pa->ParallelRAP(*pa->mat, A);
   }
   else
#endif
   {
      a->SerialRAP(A);
   }
}

void FABilinearFormExtension::EliminateBC(const Array<int> &ess_dofs,
                                          OperatorHandle &A)
{
   MFEM_VERIFY(a->diag_policy == DiagonalPolicy::DIAG_ONE,
               "Only DiagonalPolicy::DIAG_ONE supported with"
               " FABilinearFormExtension.");
#ifdef MFEM_USE_MPI
   if (dynamic_cast<ParBilinearForm *>(a))
   {
      A.As<HypreParMatrix>()->EliminateBC(ess_dofs,
                                          DiagonalPolicy::DIAG_ONE);
   }
   else
#endif
   {
      A.As<SparseMatrix>()->EliminateBC(ess_dofs,
                                        DiagonalPolicy::DIAG_ONE);
   }
}

void FABilinearFormExtension::FormSystemMatrix(const Array<int> &ess_dofs,
                                               OperatorHandle &A)
{
   RAP(A);
   EliminateBC(ess_dofs, A);
}

void FABilinearFormExtension::FormLinearSystem(const Array<int> &ess_tdof_list,
                                               Vector &x, Vector &b,
                                               OperatorHandle &A,
                                               Vector &X, Vector &B,
                                               int copy_interior)
{
   Operator *A_out;
   Operator::FormLinearSystem(ess_tdof_list, x, b, A_out, X, B, copy_interior);
   delete A_out;
   FormSystemMatrix(ess_tdof_list, A);
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

/// Data and methods for matrix-free bilinear forms
MFBilinearFormExtension::MFBilinearFormExtension(BilinearForm *form)
   : BilinearFormExtension(form)
{
   Update();
}

void MFBilinearFormExtension::Assemble()
{
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

void MFBilinearFormExtension::AssembleDiagonal(Vector &y) const
{
   Array<BilinearFormIntegrator *> &integrators = *a->GetDBFI();
   if (elem_restrict)
   {
      if (integrators.Size() > 0)
      {
         localY = 0.0;
         for (BilinearFormIntegrator *integ : integrators)
         {
            integ->AssembleDiagonalMF(localY);
         }
         const ElementRestriction *H1elem_restrict =
            dynamic_cast<const ElementRestriction *>(elem_restrict);
         if (H1elem_restrict)
         {
            H1elem_restrict->MultTransposeUnsigned(localY, y);
         }
         else
         {
            elem_restrict->MultTranspose(localY, y);
         }
      }
      else
      {
         y = 0.0;
      }
   }
   else
   {
      y.UseDevice(true); // typically this is a large vector, so store on device
      y = 0.0;
      for (BilinearFormIntegrator *integ : integrators)
      {
         integ->AssembleDiagonalMF(y);
      }
   }

   Array<BilinearFormIntegrator *> &bdr_integrators = *a->GetBBFI();
   if (bdr_face_restrict_lex)
   {
      if (bdr_integrators.Size() > 0)
      {
         bdr_face_Y = 0.0;
         for (BilinearFormIntegrator *integ : bdr_integrators)
         {
            integ->AssembleDiagonalMF(bdr_face_Y);
         }
         bdr_face_restrict_lex->AddMultTransposeUnsigned(bdr_face_Y, y);
      }
   }
   else
   {
      for (BilinearFormIntegrator *integ : bdr_integrators)
      {
         integ->AssembleDiagonalMF(y);
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

void MFBilinearFormExtension::FormSystemMatrix(const Array<int> &ess_tdof_list,
                                               OperatorHandle &A)
{
   Operator *oper;
   Operator::FormSystemOperator(ess_tdof_list, oper);
   A.Reset(oper); // A will own oper
}

void MFBilinearFormExtension::FormLinearSystem(const Array<int> &ess_tdof_list,
                                               Vector &x, Vector &b,
                                               OperatorHandle &A,
                                               Vector &X, Vector &B,
                                               int copy_interior)
{
   Operator *oper;
   Operator::FormLinearSystem(ess_tdof_list, x, b, oper, X, B, copy_interior);
   A.Reset(oper); // A will own oper
}

void MFBilinearFormExtension::Mult(const Vector &x, Vector &y) const
{
   Array<BilinearFormIntegrator *> &integrators = *a->GetDBFI();
   if (elem_restrict)
   {
      if (integrators.Size() > 0)
      {
         elem_restrict->Mult(x, localX);
         localY = 0.0;
         for (BilinearFormIntegrator *integ : integrators)
         {
            integ->AddMultMF(localX, localY);
         }
         elem_restrict->MultTranspose(localY, y);
      }
      else
      {
         y = 0.0;
      }
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
   if (bdr_face_restrict_lex)
   {
      if (bdr_integrators.Size() > 0)
      {
         bdr_face_restrict_lex->Mult(x, bdr_face_X);
         if (bdr_face_X.Size() > 0)
         {
            bdr_face_Y = 0.0;
            for (BilinearFormIntegrator *integ : bdr_integrators)
            {
               integ->AddMultMF(bdr_face_X, bdr_face_Y);
            }
            bdr_face_restrict_lex->AddMultTransposeInPlace(bdr_face_Y, y);
         }
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

void MFBilinearFormExtension::MultTranspose(const Vector &x, Vector &y) const
{
   Array<BilinearFormIntegrator *> &integrators = *a->GetDBFI();
   if (elem_restrict)
   {
      if (integrators.Size() > 0)
      {
         elem_restrict->Mult(x, localX);
         localY = 0.0;
         for (BilinearFormIntegrator *integ : integrators)
         {
            integ->AddMultTransposeMF(localX, localY);
         }
         elem_restrict->MultTranspose(localY, y);
      }
      else
      {
         y = 0.0;
      }
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
   if (bdr_face_restrict_lex)
   {
      if (bdr_integrators.Size() > 0)
      {
         bdr_face_restrict_lex->Mult(x, bdr_face_X);
         if (bdr_face_X.Size() > 0)
         {
            bdr_face_Y = 0.0;
            for (BilinearFormIntegrator *integ : bdr_integrators)
            {
               integ->AddMultTransposeMF(bdr_face_X, bdr_face_Y);
            }
            bdr_face_restrict_lex->AddMultTransposeInPlace(bdr_face_Y, y);
         }
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

/// Data and methods for partially-assembled mixed bilinear forms
PAMixedBilinearFormExtension::PAMixedBilinearFormExtension(
   MixedBilinearForm *form)
   : MixedBilinearFormExtension(form)
{
   Update();
}

void PAMixedBilinearFormExtension::Assemble()
{
   Array<BilinearFormIntegrator *> &integrators = *a->GetDBFI();
   for (BilinearFormIntegrator *integ : integrators)
   {
      integ->AssemblePA(*trial_fes, *test_fes);
   }

   MFEM_VERIFY(a->GetBBFI()->Size() == 0,
               "Partial assembly does not support AddBoundaryIntegrator yet.");

   MFEM_VERIFY(a->GetTFBFI()->Size() == 0,
               "Partial assembly does not support AddTraceFaceIntegrator yet.");

   MFEM_VERIFY(a->GetBTFBFI()->Size() == 0,
               "Partial assembly does not support AddBdrTraceFaceIntegrator yet.");
}

void PAMixedBilinearFormExtension::AssembleDiagonal_ADAt(const Vector &D,
                                                         Vector &diag) const
{
   Array<BilinearFormIntegrator *> &integrators = *a->GetDBFI();
   if (elem_restrict_trial)
   {
      const ElementRestriction *H1elem_restrict_trial =
         dynamic_cast<const ElementRestriction *>(elem_restrict_trial);
      if (H1elem_restrict_trial)
      {
         H1elem_restrict_trial->MultUnsigned(D, localTrial);
      }
      else
      {
         elem_restrict_trial->Mult(D, localTrial);
      }
   }
   if (elem_restrict_test)
   {
      localTest = 0.0;
      for (BilinearFormIntegrator *integ : integrators)
      {
         if (elem_restrict_trial)
         {
            integ->AssembleDiagonalPA_ADAt(localTrial, localTest);
         }
         else
         {
            integ->AssembleDiagonalPA_ADAt(D, localTest);
         }
      }
      const ElementRestriction *H1elem_restrict_test =
         dynamic_cast<const ElementRestriction *>(elem_restrict_test);
      if (H1elem_restrict_test)
      {
         H1elem_restrict_test->MultTransposeUnsigned(localTest, diag);
      }
      else
      {
         elem_restrict_test->MultTranspose(localTest, diag);
      }
   }
   else
   {
      diag.UseDevice(true); // typically this is a large vector, so store on device
      diag = 0.0;
      for (BilinearFormIntegrator *integ : integrators)
      {
         if (elem_restrict_trial)
         {
            integ->AssembleDiagonalPA_ADAt(localTrial, diag);
         }
         else
         {
            integ->AssembleDiagonalPA_ADAt(D, diag);
         }
      }
   }
}

void PAMixedBilinearFormExtension::Update()
{
   trial_fes = a->TrialFESpace();
   test_fes  = a->TestFESpace();
   height = test_fes->GetVSize();
   width  = trial_fes->GetVSize();

   if (DeviceCanUseCeed())
   {
      elem_restrict_trial = nullptr;
      elem_restrict_test = nullptr;
      return;
   }
   ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   elem_restrict_trial = trial_fes->GetElementRestriction(ordering);
   elem_restrict_test  = test_fes->GetElementRestriction(ordering);
   if (elem_restrict_trial)
   {
      localTrial.SetSize(elem_restrict_trial->Height(), Device::GetMemoryType());
      localTrial.UseDevice(true); // ensure 'localTrial = 0.0' is done on device
   }
   if (elem_restrict_test)
   {
      localTest.SetSize(elem_restrict_test->Height(), Device::GetMemoryType());
      localTest.UseDevice(true); // ensure 'localTest = 0.0' is done on device
   }
}

void PAMixedBilinearFormExtension::FormRectangularSystemOperator(
   const Array<int> &trial_tdof_list,
   const Array<int> &test_tdof_list,
   OperatorHandle &A)
{
   Operator *oper;
   Operator::FormRectangularSystemOperator(trial_tdof_list, test_tdof_list,
                                           oper);
   A.Reset(oper); // A will own oper
}

void PAMixedBilinearFormExtension::FormRectangularLinearSystem(
   const Array<int> &trial_tdof_list,
   const Array<int> &test_tdof_list,
   Vector &x, Vector &b,
   OperatorHandle &A,
   Vector &X, Vector &B)
{
   Operator *oper;
   Operator::FormRectangularLinearSystem(trial_tdof_list, test_tdof_list, x, b,
                                         oper, X, B);
   A.Reset(oper); // A will own oper
}

void PAMixedBilinearFormExtension::SetupMultInputs(
   const Operator *elem_restrict_x,
   const Vector &x,
   Vector &localX,
   const Operator *elem_restrict_y,
   Vector &y,
   Vector &localY,
   const double c) const
{
   if (elem_restrict_x)
   {
      elem_restrict_x->Mult(x, localX);
      if (c != 1.0)
      {
         localX *= c;
      }
   }
   else
   {
      if (c == 1.0)
      {
         localX.SyncAliasMemory(x);
      }
      else
      {
         localX.Set(c, x);
      }
   }
   if (elem_restrict_y)
   {
      localY = 0.0;
   }
   else
   {
      y.UseDevice(true);
      localY.SyncAliasMemory(y);
   }
}

void PAMixedBilinearFormExtension::Mult(const Vector &x, Vector &y) const
{
   y = 0.0;
   AddMult(x, y);
}

void PAMixedBilinearFormExtension::AddMult(const Vector &x, Vector &y,
                                           const double c) const
{
   // G operation
   SetupMultInputs(elem_restrict_trial, x, localTrial,
                   elem_restrict_test, y, localTest, c);

   // B^T D B operation
   Array<BilinearFormIntegrator *> &integrators = *a->GetDBFI();
   for (BilinearFormIntegrator *integ : integrators)
   {
      integ->AddMultPA(localTrial, localTest);
   }

   // G^T operation
   if (elem_restrict_test)
   {
      tempY.SetSize(y.Size());
      elem_restrict_test->MultTranspose(localTest, tempY);
      y += tempY;
   }
}

void PAMixedBilinearFormExtension::MultTranspose(const Vector &x,
                                                 Vector &y) const
{
   y = 0.0;
   AddMultTranspose(x, y);
}

void PAMixedBilinearFormExtension::AddMultTranspose(const Vector &x, Vector &y,
                                                    const double c) const
{
   // G operation
   SetupMultInputs(elem_restrict_test, x, localTest,
                   elem_restrict_trial, y, localTrial, c);

   // B^T D^T B operation
   Array<BilinearFormIntegrator *> &integrators = *a->GetDBFI();
   for (BilinearFormIntegrator *integ : integrators)
   {
      integ->AddMultTransposePA(localTest, localTrial);
   }

   // G^T operation
   if (elem_restrict_trial)
   {
      tempY.SetSize(y.Size());
      elem_restrict_trial->MultTranspose(localTrial, tempY);
      y += tempY;
   }
}

/// Data and methods for partially-assembled discrete linear operators
PADiscreteLinearOperatorExtension::PADiscreteLinearOperatorExtension(
   DiscreteLinearOperator *linop) :
   PAMixedBilinearFormExtension(linop)
{
}

const Operator
*PADiscreteLinearOperatorExtension::GetOutputRestrictionTranspose() const
{
   return a->GetOutputRestrictionTranspose();
}

void PADiscreteLinearOperatorExtension::Assemble()
{
   PAMixedBilinearFormExtension::Assemble();

   test_multiplicity.UseDevice(true);
   test_multiplicity.SetSize(elem_restrict_test->Width()); // l-vector
   Vector ones(elem_restrict_test->Height()); // e-vector
   ones = 1.0;

   const ElementRestriction *elem_restrict =
      dynamic_cast<const ElementRestriction *>(elem_restrict_test);
   MFEM_VERIFY(elem_restrict,
               "A real ElementRestriction is required in this setting!");
   elem_restrict->MultTransposeUnsigned(ones, test_multiplicity);
   test_multiplicity.Reciprocal();
}

void PADiscreteLinearOperatorExtension::FormRectangularSystemOperator(
   const Array<int> &trial_tdof_list,
   const Array<int> &test_tdof_list,
   OperatorHandle &A)
{
   // This acts very much like PAMixedBilinearFormExtension, but emulates 'Set'
   // rather than 'Add' in the assembly case.
   const Operator *Pi = GetProlongation();
   const Operator *RoT = GetOutputRestrictionTranspose();
   Operator *rap = SetupRAP(Pi, RoT);
   RectangularConstrainedOperator *Arco
      = new RectangularConstrainedOperator(rap, trial_tdof_list, test_tdof_list,
                                           rap != this);
   A.Reset(Arco);
}

void PADiscreteLinearOperatorExtension::AddMult(
   const Vector &x, Vector &y, const double c) const
{
   // G operation
   SetupMultInputs(elem_restrict_trial, x, localTrial,
                   elem_restrict_test, y, localTest, c);

   // B^T D B operation
   Array<BilinearFormIntegrator *> &integrators = *a->GetDBFI();
   for (BilinearFormIntegrator *integ : integrators)
   {
      integ->AddMultPA(localTrial, localTest);
   }

   // G^T operation (kind of...): Do a kind of "set" rather than "add" in the
   // below operation as compared to the BilinearForm case
   const ElementRestriction *elem_restrict =
      dynamic_cast<const ElementRestriction *>(elem_restrict_test);
   MFEM_VERIFY(elem_restrict,
               "A real ElementRestriction is required in this setting!");
   tempY.SetSize(y.Size());
   elem_restrict->MultLeftInverse(localTest, tempY);
   y += tempY;
}

void PADiscreteLinearOperatorExtension::AddMultTranspose(
   const Vector &x, Vector &y, const double c) const
{
   // G operation (kind of...): Do a kind of "set" rather than "add" in the
   // below operation as compared to the BilinearForm case
   Vector xscaled(x);
   xscaled *= test_multiplicity;
   SetupMultInputs(elem_restrict_test, xscaled, localTest,
                   elem_restrict_trial, y, localTrial, c);

   // B^T D^T B operation
   Array<BilinearFormIntegrator *> &integrators = *a->GetDBFI();
   for (BilinearFormIntegrator *integ : integrators)
   {
      integ->AddMultTransposePA(localTest, localTrial);
   }

   // G^T operation
   MFEM_VERIFY(elem_restrict_trial, "Trial ElementRestriction not defined!");
   tempY.SetSize(y.Size());
   elem_restrict_trial->MultTranspose(localTrial, tempY);
   y += tempY;
}

} // namespace mfem
