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

// Implementations of classes FABilinearFormExtension, EABilinearFormExtension,
// PABilinearFormExtension and MFBilinearFormExtension.

#include "../general/forall.hpp"
#include "bilinearform.hpp"

namespace mfem
{

ElemRestriction::ElemRestriction(const FiniteElementSpace &f)
   : fes(f),
     ne(fes.GetNE()),
     vdim(fes.GetVDim()),
     byvdim(fes.GetOrdering() == Ordering::byVDIM),
     ndofs(fes.GetNDofs()),
     dof(ne > 0 ? fes.GetFE(0)->GetDof() : 0),
     nedofs(ne*dof)
{
   height = nedofs*vdim;
   width = ndofs*vdim;
   if (ne == 0)
   {
      dofs_edofs.SetSize(0, 0);
      return;
   }
   for (int e = 0; e < ne; ++e)
   {
      const FiniteElement *fe = fes.GetFE(e);
      const TensorBasisElement* el =
         dynamic_cast<const TensorBasisElement*>(fe);
      if (el) { continue; }
      mfem_error("Finite element not supported with partial assembly");
   }
   const FiniteElement *fe = fes.GetFE(0);
   const TensorBasisElement* el = dynamic_cast<const TensorBasisElement*>(fe);
   const Array<int> &dof_map = el->GetDofMap();
   const bool dof_map_is_identity = (dof_map.Size()==0);
   const Table& e2dTable = fes.GetElementToDofTable();
   const int* elementMap = e2dTable.GetJ();

   dofs_edofs.MakeI(ndofs);
   for (int k = 0; k < nedofs; k++)
   {
      dofs_edofs.AddAColumnInRow(elementMap[k]);
   }
   dofs_edofs.MakeJ();
   for (int i = 0; i < ne; i++)
   {
      MFEM_ASSERT(e2dTable.RowSize(i) == dof,
                  "incompatible element-to-dof Table!");
      for (int j_lex = 0; j_lex < dof; j_lex++)
      {
         const int j = dof_map_is_identity ? j_lex : dof_map[j_lex];
         dofs_edofs.AddConnection(elementMap[j+dof*i], j_lex+dof*i);
      }
   }
   dofs_edofs.ShiftUpI();
}

void ElemRestriction::Mult(const Vector& x, Vector& y) const
{
   const int vd = vdim;
   const bool t = byvdim;
   auto d_offsets = ReadAccess(dofs_edofs.GetIMemory(), ndofs+1);
   auto d_indices = ReadAccess(dofs_edofs.GetJMemory(), nedofs);
   // TODO: add support for different strides in the 2D arrays d_x and d_y -
   // this way we can avoid the branching created by using expressions like
   // "t?c:i".
   auto d_x = Reshape(x.ReadAccess(), t?vd:ndofs, t?ndofs:vd);
   auto d_y = Reshape(y.WriteAccess(), t?vd:nedofs, t?nedofs:vd);
   MFEM_FORALL(i, ndofs,
   {
      const int offset = d_offsets[i];
      const int nextOffset = d_offsets[i+1];
      for (int c = 0; c < vd; ++c)
      {
         const double dofValue = d_x(t?c:i,t?i:c);
         for (int j = offset; j < nextOffset; ++j)
         {
            const int idx_j = d_indices[j];
            d_y(t?c:idx_j,t?idx_j:c) = dofValue;
         }
      }
   });
}

void ElemRestriction::MultTranspose(const Vector& x, Vector& y) const
{
   const int vd = vdim;
   const bool t = byvdim;
   auto d_offsets = ReadAccess(dofs_edofs.GetIMemory(), ndofs+1);
   auto d_indices = ReadAccess(dofs_edofs.GetJMemory(), nedofs);
   auto d_x = Reshape(x.ReadAccess(), t?vd:nedofs, t?nedofs:vd);
   auto d_y = Reshape(y.WriteAccess(), t?vd:ndofs, t?ndofs:vd);
   MFEM_FORALL(i, ndofs,
   {
      const int offset = d_offsets[i];
      const int nextOffset = d_offsets[i + 1];
      for (int c = 0; c < vd; ++c)
      {
         double dofValue = 0;
         for (int j = offset; j < nextOffset; ++j)
         {
            const int idx_j = d_indices[j];
            dofValue +=  d_x(t?c:idx_j,t?idx_j:c);
         }
         d_y(t?c:i,t?i:c) = dofValue;
      }
   });
}


BilinearFormExtension::BilinearFormExtension(BilinearForm *form)
   : Operator(form->Size()), a(form)
{
   // empty
}

const Operator *BilinearFormExtension::GetProlongation() const
{
   return a->GetProlongation();
}

const Operator *BilinearFormExtension::GetRestriction() const
{
   return a->GetRestriction();
}


// Data and methods for partially-assembled bilinear forms
PABilinearFormExtension::PABilinearFormExtension(BilinearForm *form)
   : BilinearFormExtension(form),
     trialFes(a->FESpace()), testFes(a->FESpace()),
     elem_restrict(new ElemRestriction(*a->FESpace()))
{
   const Table &el_dof = trialFes->GetElementToDofTable();
   const int esize = el_dof.Size_of_connections()*trialFes->GetVDim();
   localX.SetSize(esize, Device::GetMemoryType());
   localY.SetSize(esize, Device::GetMemoryType());
   localY.UseDevice(); // ensure 'localY = 0.0' is done on device
}

PABilinearFormExtension::~PABilinearFormExtension()
{
   delete elem_restrict;
}

void PABilinearFormExtension::Assemble()
{
   Array<BilinearFormIntegrator*> &integrators = *a->GetDBFI();
   const int integratorCount = integrators.Size();
   for (int i = 0; i < integratorCount; ++i)
   {
      integrators[i]->AssemblePA(*a->FESpace());
   }
}

void PABilinearFormExtension::Update()
{
   FiniteElementSpace *fes = a->FESpace();
   height = width = fes->GetVSize();
   trialFes = fes;
   testFes = fes;
   const Table &el_dof = trialFes->GetElementToDofTable();
   const int esize = el_dof.Size_of_connections()*trialFes->GetVDim();
   localX.SetSize(esize);
   localY.SetSize(esize);
   delete elem_restrict;
   elem_restrict = new ElemRestriction(*fes);
}

void PABilinearFormExtension::FormSystemMatrix(const Array<int> &ess_tdof_list,
                                               OperatorHandle &A)
{
   const Operator* trialP = trialFes->GetProlongationMatrix();
   const Operator* testP  = testFes->GetProlongationMatrix();
   Operator *rap = this;
   if (trialP) { rap = new RAPOperator(*testP, *this, *trialP); }
   const bool own_A = (rap!=this);
   A.Reset(new ConstrainedOperator(rap, ess_tdof_list, own_A));
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
   Array<BilinearFormIntegrator*> &integrators = *a->GetDBFI();
   elem_restrict->Mult(x, localX);
   localY = 0.0;
   const int iSz = integrators.Size();
   for (int i = 0; i < iSz; ++i)
   {
      integrators[i]->AddMultPA(localX, localY);
   }
   elem_restrict->MultTranspose(localY, y);
}

void PABilinearFormExtension::MultTranspose(const Vector &x, Vector &y) const
{
   Array<BilinearFormIntegrator*> &integrators = *a->GetDBFI();
   elem_restrict->Mult(x, localX);
   localY = 0.0;
   const int iSz = integrators.Size();
   for (int i = 0; i < iSz; ++i)
   {
      integrators[i]->AddMultTransposePA(localX, localY);
   }
   elem_restrict->MultTranspose(localY, y);
}

} // namespace mfem
