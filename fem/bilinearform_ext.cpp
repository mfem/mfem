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
PABilinearFormExtension::PABilinearFormExtension(BilinearForm *form) :
   BilinearFormExtension(form),
   trialFes(a->FESpace()), testFes(a->FESpace()),
   localX(trialFes->GetNE() * trialFes->GetFE(0)->GetDof() * trialFes->GetVDim()),
   localY( testFes->GetNE() * testFes->GetFE(0)->GetDof() * testFes->GetVDim()),
   elem_restrict(new ElemRestriction(*a->FESpace())) { }

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
      integrators[i]->Assemble(*a->FESpace());
   }
}

void PABilinearFormExtension::Update()
{
   FiniteElementSpace *fes = a->FESpace();
   height = width = fes->GetVSize();
   trialFes = fes;
   testFes = fes;
   localX.SetSize(trialFes->GetNE() * trialFes->GetFE(0)->GetDof() *
                  trialFes->GetVDim());
   localY.SetSize(testFes->GetNE() * testFes->GetFE(0)->GetDof() *
                  testFes->GetVDim());
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
      integrators[i]->MultAssembled(localX, localY);
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
      integrators[i]->MultAssembledTranspose(localX, localY);
   }
   elem_restrict->MultTranspose(localY, y);
}


ElemRestriction::ElemRestriction(const FiniteElementSpace &f)
   : fes(f),
     ne(fes.GetNE()),
     vdim(fes.GetVDim()),
     byvdim(fes.GetOrdering() == Ordering::byVDIM),
     ndofs(fes.GetNDofs()),
     dof(fes.GetFE(0)->GetDof()),
     nedofs(ne*dof),
     offsets(ndofs+1),
     indices(ne*dof)
{
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
   // We'll be keeping a count of how many local nodes point to its global dof
   for (int i = 0; i <= ndofs; ++i)
   {
      offsets[i] = 0;
   }
   for (int e = 0; e < ne; ++e)
   {
      for (int d = 0; d < dof; ++d)
      {
         const int gid = elementMap[dof*e + d];
         ++offsets[gid + 1];
      }
   }
   // Aggregate to find offsets for each global dof
   for (int i = 1; i <= ndofs; ++i)
   {
      offsets[i] += offsets[i - 1];
   }
   // For each global dof, fill in all local nodes that point   to it
   for (int e = 0; e < ne; ++e)
   {
      for (int d = 0; d < dof; ++d)
      {
         const int did = dof_map_is_identity?d:dof_map[d];
         const int gid = elementMap[dof*e + did];
         const int lid = dof*e + d;
         indices[offsets[gid]++] = lid;
      }
   }
   // We shifted the offsets vector by 1 by using it as a counter
   // Now we shift it back.
   for (int i = ndofs; i > 0; --i)
   {
      offsets[i] = offsets[i - 1];
   }
   offsets[0] = 0;
}

void ElemRestriction::Mult(const Vector& x, Vector& y) const
{
   const int vd = vdim;
   const bool t = byvdim;
   const DeviceArray d_offsets(offsets, ndofs+1);
   const DeviceArray d_indices(indices, nedofs);
   const DeviceMatrix d_x(x, t?vd:ndofs, t?ndofs:vd);
   DeviceMatrix d_y(y, t?vd:nedofs, t?nedofs:vd);
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
   const DeviceArray d_offsets(offsets, ndofs+1);
   const DeviceArray d_indices(indices, nedofs);
   const DeviceMatrix d_x(x, t?vd:nedofs, t?nedofs:vd);
   DeviceMatrix d_y(y, t?vd:ndofs, t?ndofs:vd);
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

} // namespace mfem
