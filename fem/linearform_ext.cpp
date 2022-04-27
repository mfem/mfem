// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "linearform.hpp"
#include "../general/forall.hpp"

namespace mfem
{

LinearFormExtension::LinearFormExtension(LinearForm *lf): lf(lf) { Update(); }

void LinearFormExtension::Assemble()
{
   const FiniteElementSpace &fes = *lf->FESpace();
   MFEM_VERIFY(lf->SupportsDevice(), "Not supported.");
   MFEM_VERIFY(lf->Size() == fes.GetVSize(), "LinearForm size does not "
               "match the number of vector dofs!");

   const Array<Array<int>*> &domain_integs_marker = *lf->GetDLFI_Marker();
   const int mesh_attributes_size = fes.GetMesh()->attributes.Size();
   const Array<LinearFormIntegrator*> &domain_integs = *lf->GetDLFI();

   for (int k = 0; k < domain_integs.Size(); ++k)
   {
      // Get the markers for this integrator
      const Array<int> *domain_integs_marker_k = domain_integs_marker[k];

      // check if there are markers for this integrator
      const bool has_markers_k = domain_integs_marker_k != nullptr;

      if (has_markers_k)
      {
         // Element attribute marker should be of length mesh->attributes
         MFEM_VERIFY(mesh_attributes_size == domain_integs_marker_k->Size(),
                     "invalid element marker for domain linear form "
                     "integrator #" << k << ", counting from zero");
      }

      // if there are no markers, just use the whole linear form (1)
      if (!has_markers_k) { markers.HostReadWrite(); markers = 1; }
      else
      {
         // scan the attributes to set the markers to 0 or 1
         const int NE = fes.GetNE();
         const auto attr = attributes.Read();
         const auto dimk = domain_integs_marker_k->Read();
         auto markers_w = markers.Write();
         MFEM_FORALL(e, NE, markers_w[e] = dimk[attr[e]-1] == 1;);
      }

      // Assemble the linear form
      b = 0.0;
      domain_integs[k]->AssembleDevice(fes, markers, b);
      elem_restrict_lex->MultTranspose(b, *lf);
   }
}

void LinearFormExtension::Update()
{
   const FiniteElementSpace &fes = *lf->FESpace();
   const Mesh &mesh = *fes.GetMesh();
   const int NE = fes.GetNE();

   MFEM_VERIFY(lf->Size() == fes.GetVSize(), "");

   markers.SetSize(NE);
   //markers.UseDevice(true);

   // Gather the attributes on the host from all the elements
   attributes.SetSize(NE);
   for (int i = 0; i < NE; ++i) { attributes[i] = mesh.GetAttribute(i); }

   constexpr ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   elem_restrict_lex = fes.GetElementRestriction(ordering);
   MFEM_VERIFY(elem_restrict_lex, "Element restriction not available");
   b.SetSize(elem_restrict_lex->Height(), Device::GetMemoryType());
   b.UseDevice(true);
}

} // namespace mfem
