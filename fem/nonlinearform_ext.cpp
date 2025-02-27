// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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

#include "nonlinearform.hpp"
#include "ceed/interface/util.hpp"
#include "../general/forall.hpp"
#include "../linalg/densemat.hpp"
#include "../linalg/libBatchSolver.hpp"

namespace mfem
{

NonlinearFormExtension::NonlinearFormExtension(const NonlinearForm *nlf)
   : Operator(nlf->FESpace()->GetVSize()), nlf(nlf) { }

PANonlinearFormExtension::PANonlinearFormExtension(const NonlinearForm *nlf, const ElementDofOrdering edf_):
   NonlinearFormExtension(nlf),
   fes(*nlf->FESpace()),
   dnfi(*nlf->GetDNFI()),
   edf(edf_),
   elemR(fes.GetElementRestriction(edf)),
   paGrad(*this)
{
   if (!DeviceCanUseCeed())
   {
      //elemR = fes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
      // TODO: optimize for the case when 'elemR' is identity
      xe.SetSize(elemR->Height(), Device::GetMemoryType());
      ye.SetSize(elemR->Height(), Device::GetMemoryType());
   }
   ye.UseDevice(true);
}

real_t PANonlinearFormExtension::GetGridFunctionEnergy(const Vector &x) const
{
   real_t energy = 0.0;

   elemR->Mult(x, xe);
   for (int i = 0; i < dnfi.Size(); i++)
   {
      energy += dnfi[i]->GetLocalStateEnergyPA(xe);
   }
   return energy;
}

void PANonlinearFormExtension::Assemble()
{
   MFEM_VERIFY(nlf->GetInteriorFaceIntegrators().Size() == 0 &&
               nlf->GetBdrFaceIntegrators().Size() == 0,
               "face integrators are not supported yet");

   for (int i = 0; i < dnfi.Size(); ++i) { dnfi[i]->AssemblePA(fes); }
}

void PANonlinearFormExtension::Mult(const Vector &x, Vector &y) const
{
   if (!DeviceCanUseCeed())
   {
      ye = 0.0;
      elemR->Mult(x, xe);
      for (int i = 0; i < dnfi.Size(); ++i) { dnfi[i]->AddMultPA(xe, ye); }
      elemR->MultTranspose(ye, y);
   }
   else
   {
      y.UseDevice(true); // typically this is a large vector, so store on device
      y = 0.0;
      for (int i = 0; i < dnfi.Size(); ++i)
      {
         dnfi[i]->AddMultPA(x, y);
      }
   }
}

Operator &PANonlinearFormExtension::GetGradient(const Vector &x) const
{
   paGrad.AssembleGrad(x);
   return paGrad;
}

void PANonlinearFormExtension::Update()
{
   height = width = fes.GetVSize();
   elemR = fes.GetElementRestriction(edf);
   xe.SetSize(elemR->Height());
   ye.SetSize(elemR->Height());
   paGrad.Update();
}

PANonlinearFormExtension::PAGradient::PAGradient(const PANonlinearFormExtension &e):
   Operator(e.Height()), ext(e)
{ }

void PANonlinearFormExtension::PAGradient::AssembleGrad(const Vector &g)
{
   ext.elemR->Mult(g, ext.xe);
   for (int i = 0; i < ext.dnfi.Size(); ++i)
   {
      ext.dnfi[i]->AssembleGradPA(ext.xe, ext.fes);
   }
}

void PANonlinearFormExtension::PAGradient::Mult(const Vector &x, Vector &y) const
{
   MFEM_PERF_SCOPE("PANonlinearFormExtension::PAGradient::Mult");
   ext.ye = 0.0;
   ext.elemR->Mult(x, ext.xe);
   for (int i = 0; i < ext.dnfi.Size(); ++i)
   {
      ext.dnfi[i]->AddMultGradPA(ext.xe, ext.ye);
   }
   ext.elemR->MultTranspose(ext.ye, y);
}

void PANonlinearFormExtension::PAGradient::AssembleDiagonal(Vector &diag) const
{
   MFEM_ASSERT(diag.Size() == Height(),
               "Vector for holding diagonal has wrong size!");
   ext.ye = 0.0;
   for (int i = 0; i < ext.dnfi.Size(); ++i)
   {
      ext.dnfi[i]->AssembleGradDiagonalPA(ext.ye);
   }
   ext.elemR->MultTranspose(ext.ye, diag);
}

void PANonlinearFormExtension::PAGradient::Update()
{
   height = width = ext.Height();
}

EANonlinearFormExtension::EANonlinearFormExtension(const NonlinearForm *nlf, const ElementDofOrdering edf_):
  PANonlinearFormExtension(nlf, edf_), eaGrad(*this)
{
   ne = fes.GetMesh()->GetNE();
   elemDofs = fes.GetFE(0)->GetDof() * fes.GetFE(0)->GetDim();
   ea_data.SetSize(ne * elemDofs * elemDofs, Device::GetMemoryType());
   ea_data.UseDevice(true);
   eaGradDT.UseExternalData(ea_data.ReadWrite(), elemDofs, elemDofs, ne);
   if (Device::Allows(Backend::DEVICE_MASK))
   {
      batchMult = new LibBatchMult(eaGradDT);
   }
}

EANonlinearFormExtension::~EANonlinearFormExtension()
{
  if (batchMult) {
      delete batchMult;
  }
}

EANonlinearFormExtension::EAGradient::EAGradient(const EANonlinearFormExtension &e):
   Operator(e.Height()), ext(e)
{ }

void EANonlinearFormExtension::EAGradient::AssembleGrad(const Vector &g)
{
   ext.elemR->Mult(g, ext.xe);
   for (int i = 0; i < ext.dnfi.Size(); ++i)
   {
      ext.dnfi[i]->AssembleGradEA(ext.xe, ext.fes, ext.ea_data);
   }
   ext.eaGradDT.UseExternalData(ext.ea_data.ReadWrite(), ext.elemDofs, ext.elemDofs, ext.ne);
}

void EANonlinearFormExtension::EAGradient::Mult(const Vector &x, Vector &y) const
{
   MFEM_PERF_SCOPE("EANonlinearFormExtension::EAGradient::Mult");
   ext.ye = 0.0;
   ext.elemR->Mult(x, ext.xe);
   MFEM_PERF_BEGIN("EANonlinearFormExtension::EAGradient::Mult::MatVecMult");
   // Apply the Element Matrices
   if (ext.batchMult)
   {
      ext.batchMult->Mult(ext.xe, ext.ye);
   }
   else
   {
      const int NDOFS = ext.elemDofs;
      auto X = Reshape(ext.xe.Read(), NDOFS, ext.ne);
      auto Y = Reshape(ext.ye.ReadWrite(), NDOFS, ext.ne);
      auto A = Reshape(ext.ea_data.Read(), NDOFS, NDOFS, ext.ne);
      MFEM_FORALL(glob_j, ext.ne * NDOFS,
      {
         const int e = glob_j / NDOFS;
         const int j = glob_j % NDOFS;
         double res = 0.0;
         for (int i = 0; i < NDOFS; i++)
         {
            res += A(i, j, e) * X(i, e);
         }
         Y(j, e) += res;
      });
   }
   MFEM_PERF_END("EANonlinearFormExtension::EAGradient::Mult::MatVecMult");
   ext.elemR->MultTranspose(ext.ye, y);
}

void EANonlinearFormExtension::EAGradient::AssembleDiagonal(Vector &diag) const
{
   MFEM_ASSERT(diag.Size() == Height(),
               "Vector for holding diagonal has wrong size!");
   ext.ye = 0.0;

   // Apply the Element Matrices
   const int NDOFS = ext.elemDofs;
   auto Y = Reshape(ext.ye.ReadWrite(), NDOFS, ext.ne);
   auto A = Reshape(ext.ea_data.Read(), NDOFS, NDOFS, ext.ne);
   MFEM_FORALL(glob_j, ext.ne * NDOFS,
   {
      const int e = glob_j / NDOFS;
      const int j = glob_j % NDOFS;
      Y(j, e) += A(j, j, e);
   });

   ext.elemR->MultTranspose(ext.ye, diag);
}

void EANonlinearFormExtension::EAGradient::Update()
{
   height = width = ext.Height();
}

Operator &EANonlinearFormExtension::GetGradient(const Vector &x) const
{
   ea_data = 0.0;
   eaGrad.AssembleGrad(x);
   return eaGrad;
}

FANonlinearFormExtension::FANonlinearFormExtension(const NonlinearForm *nlf, const ElementDofOrdering edf_):
   EANonlinearFormExtension(nlf, edf_), faGrad(*this)
{ }

FANonlinearFormExtension::FAGradient::FAGradient(const FANonlinearFormExtension &e):
   Operator(e.Height()), ext(e)
{ }

void FANonlinearFormExtension::FAGradient::AssembleGrad(const Vector &g)
{
   ext.EANonlinearFormExtension::GetGradient(g);
   int width = ext.fes.GetVSize();
   int height = ext.fes.GetVSize();
   if (ext.mat) // We reuse the sparse matrix memory
   {
         const ElementRestriction &rest =
            static_cast<const ElementRestriction&>(*ext.elemR);
         rest.FillJAndData(ext.ea_data, *ext.mat);
   }
   else // We create, compute the sparsity, and fill the sparse matrix
   {
      ext.mat = new SparseMatrix(height, width, 0);
      const ElementRestriction &rest =
         static_cast<const ElementRestriction&>(*ext.elemR);
      rest.FillSparseMatrix(ext.ea_data, *ext.mat);
   }


}

void FANonlinearFormExtension::FAGradient::Mult(const Vector &x, Vector &y) const
{
   MFEM_PERF_SCOPE("FANonlinearFormExtension::FAGradient::Mult");
   // not certain this is the behavior we want but...
   y = 0.0;
   ext.mat->Mult(x, y);
}

void FANonlinearFormExtension::FAGradient::AssembleDiagonal(Vector &diag) const
{
   MFEM_ASSERT(diag.Size() == Height(),
               "Vector for holding diagonal has wrong size!");
   // not certain this is the behavior we want but...
   diag = 0.0;
   ext.mat->AssembleDiagonal(diag);
}

void FANonlinearFormExtension::FAGradient::Update()
{
   height = width = ext.Height();
}

Operator &FANonlinearFormExtension::GetGradient(const Vector &x) const
{
   faGrad.AssembleGrad(x);
   return faGrad;
}

MFNonlinearFormExtension::MFNonlinearFormExtension(const NonlinearForm *form, const ElementDofOrdering edf_):
   NonlinearFormExtension(form), fes(*form->FESpace()), edf(edf_)
{
   elem_restrict_lex = fes.GetElementRestriction(edf);
   if (elem_restrict_lex) // replace with a check for not identity
   if (!DeviceCanUseCeed())
   {
      const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
      elem_restrict_lex = fes.GetElementRestriction(ordering);
      if (elem_restrict_lex) // replace with a check for not identity
      {
         localX.SetSize(elem_restrict_lex->Height(), Device::GetMemoryType());
         localY.SetSize(elem_restrict_lex->Height(), Device::GetMemoryType());
         localY.UseDevice(true); // ensure 'localY = 0.0' is done on device
      }
   }
}

void MFNonlinearFormExtension::Assemble()
{
   const Array<NonlinearFormIntegrator*> &integrators = *nlf->GetDNFI();
   const int Ni = integrators.Size();
   for (int i = 0; i < Ni; ++i)
   {
      integrators[i]->AssembleMF(fes);
   }
}

void MFNonlinearFormExtension::Mult(const Vector &x, Vector &y) const
{
   const Array<NonlinearFormIntegrator*> &integrators = *nlf->GetDNFI();
   const int iSz = integrators.Size();
   // replace the check 'elem_restrict_lex' with a check for not identity
   if (elem_restrict_lex && !DeviceCanUseCeed())
   {
      elem_restrict_lex->Mult(x, localX);
      localY = 0.0;
      for (int i = 0; i < iSz; ++i)
      {
         integrators[i]->AddMultMF(localX, localY);
      }
      elem_restrict_lex->MultTranspose(localY, y);
   }
   else
   {
      y.UseDevice(true); // typically this is a large vector, so store on device
      y = 0.0;
      for (int i = 0; i < iSz; ++i)
      {
         integrators[i]->AddMultMF(x, y);
      }
   }
}

void MFNonlinearFormExtension::Update()
{
   height = width = fes.GetVSize();
   elem_restrict_lex = fes.GetElementRestriction(edf);
   if (elem_restrict_lex) // replace with a check for not identity
   {
      localX.SetSize(elem_restrict_lex->Height(), Device::GetMemoryType());
      localY.SetSize(elem_restrict_lex->Height(), Device::GetMemoryType());
   }
}

} // namespace mfem
