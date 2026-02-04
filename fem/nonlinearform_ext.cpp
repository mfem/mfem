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

// Implementations of classes FANonlinearFormExtension, EANonlinearFormExtension,
// PANonlinearFormExtension and MFNonlinearFormExtension.

#include "pnonlinearform.hpp"
#include "../general/forall.hpp"
#include "../linalg/dtensor.hpp"
#include "nonlinearform.hpp"
#include "ceed/interface/util.hpp"

namespace mfem
{

NonlinearFormExtension::NonlinearFormExtension(const NonlinearForm *nlf)
   : Operator(nlf->FESpace()->GetVSize()), nlf(nlf) { }

PANonlinearFormExtension::PANonlinearFormExtension(const NonlinearForm *nlf,
                                                   const ElementDofOrdering edf_)
    : NonlinearFormExtension(nlf), fes(*nlf->FESpace()), dnfi(*nlf->GetDNFI()),
      elemR(nullptr), Grad(*this),
      edf(edf_)
{
   if (!DeviceCanUseCeed())
   {
      elemR = fes.GetElementRestriction(edf);
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
   Grad.AssembleGrad(x);
   return Grad;
}

void PANonlinearFormExtension::Update()
{
   height = width = fes.GetVSize();
   if (!elemR)
      elemR = fes.GetElementRestriction(edf);
   xe.SetSize(elemR->Height());
   ye.SetSize(elemR->Height());
   Grad.Update();
}

PANonlinearFormExtension::Gradient::Gradient(const PANonlinearFormExtension &e):
   Operator(e.Height()), ext(e)
{ }

void PANonlinearFormExtension::Gradient::AssembleGrad(const Vector &g)
{
   ext.elemR->Mult(g, ext.xe);
   for (int i = 0; i < ext.dnfi.Size(); ++i)
   {
      ext.dnfi[i]->AssembleGradPA(ext.xe, ext.fes);
   }
}

void PANonlinearFormExtension::Gradient::Mult(const Vector &x, Vector &y) const
{
   ext.ye = 0.0;
   ext.elemR->Mult(x, ext.xe);
   for (int i = 0; i < ext.dnfi.Size(); ++i)
   {
      ext.dnfi[i]->AddMultGradPA(ext.xe, ext.ye);
   }
   ext.elemR->MultTranspose(ext.ye, y);
}

void PANonlinearFormExtension::Gradient::AssembleDiagonal(Vector &diag) const
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

void PANonlinearFormExtension::Gradient::Update()
{
   height = width = ext.Height();
}

EANonlinearFormExtension::EANonlinearFormExtension(const NonlinearForm *nlf, const ElementDofOrdering edf_):
  PANonlinearFormExtension(nlf, edf_), eaGrad(*this)
{
   ne = fes.GetMesh()->GetNE();
   elem_vdofs = fes.GetFE(0)->GetDof() * fes.GetFE(0)->GetDim();
   ea_data.SetSize(ne * elem_vdofs * elem_vdofs, Device::GetMemoryType());
   ea_data.UseDevice(true);
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
}

void EANonlinearFormExtension::EAGradient::Mult(const Vector &x,
                                                Vector &y) const
{
   ext.ye = 0.0;
   ext.elemR->Mult(x, ext.xe);
   const int elem_vdofs = ext.elem_vdofs;
   auto X = Reshape(ext.xe.Read(), elem_vdofs, ext.ne);
   auto Y = Reshape(ext.ye.ReadWrite(), elem_vdofs, ext.ne);
   auto A = Reshape(ext.ea_data.Read(), elem_vdofs, elem_vdofs, ext.ne);
   mfem::forall(ext.ne * elem_vdofs,
                [=] MFEM_HOST_DEVICE(int glob_j)
                {
                   const int e = glob_j / elem_vdofs;
                   const int j = glob_j % elem_vdofs;
                   double res = 0.0;
                   for (int i = 0; i < elem_vdofs; i++)
                   {
                      res += A(i, j, e) * X(i, e);
                   }
                   Y(j, e) += res;
                });
   ext.elemR->MultTranspose(ext.ye, y);
}

void EANonlinearFormExtension::EAGradient::AssembleDiagonal(Vector &diag) const
{
   MFEM_ASSERT(diag.Size() == Height(),
               "Vector for holding diagonal has wrong size!");
   ext.ye = 0.0;

   // Apply the Element Matrices
   const int elem_vdofs = ext.elem_vdofs;
   auto Y = Reshape(ext.ye.ReadWrite(), elem_vdofs, ext.ne);
   auto A = Reshape(ext.ea_data.Read(), elem_vdofs, elem_vdofs, ext.ne);
   mfem::forall(ext.ne * elem_vdofs,
                [=] MFEM_HOST_DEVICE(int glob_j)
                {
                   const int e = glob_j / elem_vdofs;
                   const int j = glob_j % elem_vdofs;
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

FANonlinearFormExtension::FANonlinearFormExtension(const NonlinearForm *nlf,
                                                   const ElementDofOrdering edf_)
    : EANonlinearFormExtension(nlf, edf_), faGrad(*this)
{
}

FANonlinearFormExtension::FAGradient::FAGradient(const FANonlinearFormExtension &e)
    : Operator(e.Height()), ext(e)
{
}

void FANonlinearFormExtension::FAGradient::AssembleGrad(const Vector &g)
{
   ext.EANonlinearFormExtension::GetGradient(g);
   int width = ext.fes.GetVSize();
   int height = ext.fes.GetVSize();
   if (ext.mat) // We reuse the sparse matrix memory
   {
      const ElementRestriction &rest = static_cast<const ElementRestriction &>(*ext.elemR);
      rest.FillJAndData(ext.ea_data, *ext.mat);
   }
   else // We create, compute the sparsity, and fill the sparse matrix
   {
      ext.mat = new SparseMatrix(height, width, 0);
      const ElementRestriction &rest = static_cast<const ElementRestriction &>(*ext.elemR);
      rest.FillSparseMatrix(ext.ea_data, *ext.mat);
   }
}

void FANonlinearFormExtension::FAGradient::Mult(const Vector &x,
                                                Vector &y) const
{
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

void FANonlinearFormExtension::RAP(OperatorHandle &A) const
{
   MFEM_ASSERT(this->mat, "We should have at least allocated our mat by now");

#ifdef MFEM_USE_MPI
   if ( auto pnlf = dynamic_cast<const ParNonlinearForm*>(nlf) )
   {
      const auto *const pfespace = pnlf->ParFESpace();
      MFEM_ASSERT(pfespace, "Need the parallel nonlinar form to have its parallel finite element space populated");
      mfem::ParallelRAP(*pfespace, *this->mat, A);
   }
   else
#endif
   {
      std::vector<SparseMatrix *> mats_to_assemble = {this->mat};
      mfem::ConformingAssemble(this->fes, mats_to_assemble);
      this->mat = mats_to_assemble.front();
      A.Reset(this->mat, false);
   }
}

void FANonlinearFormExtension::EliminateBC(const Array<int> &ess_dofs,
                                          OperatorHandle &A) const
{
#ifdef MFEM_USE_MPI
   if ( dynamic_cast<const ParNonlinearForm*>(nlf) )
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

void FANonlinearFormExtension::FAGradient::FormSystemOperator(const Array<int> &ess_tdof_list,
                                                              Operator *&A) const
{
   OperatorHandle handleA;
   ext.RAP(handleA);
   ext.EliminateBC(ess_tdof_list, handleA);
   handleA.SetOperatorOwner(false); // Don't delete the operator when this function goes out of scope
   A = handleA.Ptr();
}

Operator &FANonlinearFormExtension::GetGradient(const Vector &x) const
{
   faGrad.AssembleGrad(x);
   return faGrad;
}

MFNonlinearFormExtension::MFNonlinearFormExtension(const NonlinearForm *form):
   NonlinearFormExtension(form), fes(*form->FESpace())
{
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
   const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   elem_restrict_lex = fes.GetElementRestriction(ordering);
   if (elem_restrict_lex) // replace with a check for not identity
   {
      localX.SetSize(elem_restrict_lex->Height(), Device::GetMemoryType());
      localY.SetSize(elem_restrict_lex->Height(), Device::GetMemoryType());
   }
}

} // namespace mfem
