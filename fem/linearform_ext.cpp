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

// Implementations of classes FABilinearFormExtension, EABilinearFormExtension,
// PABilinearFormExtension and MFBilinearFormExtension.

#include "linearform.hpp"

#include "../general/forall.hpp"

namespace mfem
{

LinearFormExtension::LinearFormExtension(LinearForm *lf): lf(lf) { }

FullLinearFormExtension::FullLinearFormExtension(LinearForm *lf):
   LinearFormExtension(lf)
{
   const int ne = lf->FESpace()->GetNE();
   const Mesh &mesh = *lf->FESpace()->GetMesh();

   marks.SetSize(ne);
   marks.UseDevice(true);

   attributes.SetSize(ne);
   attributes.UseDevice(true);

   // Fill the attributes vector on the host
   for (int i = 0; i < ne; ++i)
   {
      attributes[i] = mesh.GetAttribute(i);
   }
}

void FullLinearFormExtension::Assemble()
{
   // This operation is executed on device
   lf->Vector::operator=(0.0);

   // Filter out the unhandled integrators
   MFEM_VERIFY(lf->GetBLFI()->Size() == 0 && // Boundary
               lf->GetDLFI_Delta()->Size() == 0 && // Delta coefficients
               lf->GetIFLFI()->Size() == 0 && // Internal faces
               lf->GetFLFI()->Size() == 0, // Boundary faces
               "Unsupported integrators!");

   const FiniteElementSpace &fes = *lf->FESpace();
   const Array<LinearFormIntegrator*> &domain_integs = *lf->GetDLFI();
   const Array<Array<int>*> &domain_integs_marker = *lf->GetDLFIM();
   const int mesh_attributes_size = fes.GetMesh()->attributes.Size();

   for (int k = 0; k < domain_integs.Size(); ++k)
   {
      if (domain_integs_marker[k] != nullptr)
      {
         MFEM_VERIFY(mesh_attributes_size == domain_integs_marker[k]->Size(),
                     "invalid element marker for domain linear form "
                     "integrator #" << k << ", counting from zero");
      }

      marks = 0.0;

      const int NE = fes.GetNE();
      const Array<int> *dimks = domain_integs_marker[k];
      const bool no_dimk =  dimks == nullptr;
      const auto dimk_r = no_dimk ? nullptr : dimks->Read();
      const auto attr_r = attributes.Read();
      auto mark_rw = marks.ReadWrite();

      MFEM_FORALL(i, NE,
      {
         const int elem_attr = attr_r[i];
         const bool elem_attr_eq_1 = no_dimk ? false : dimk_r[elem_attr-1] == 1;
         if (no_dimk || elem_attr_eq_1) { mark_rw[i] = 1.0; }
      });

      domain_integs[k]->AssembleFull(fes, marks, *lf);
   }
}

} // namespace mfem
