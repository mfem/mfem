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

   markers.SetSize(ne);
   attributes.SetSize(ne);

   // Fill the attributes on the host
   for (int i = 0; i < ne; ++i) { attributes[i] = mesh.GetAttribute(i); }
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
      // get the markers for this integrator
      const Array<int> *domain_integs_marker_k = domain_integs_marker[k];

      if (domain_integs_marker_k != nullptr)
      {
         // Element attribute marker should be of length mesh->attributes
         MFEM_VERIFY(mesh_attributes_size == domain_integs_marker_k->Size(),
                     "invalid element marker for domain linear form "
                     "integrator #" << k << ", counting from zero");
      }

      // check if there are markers for this integrator
      const bool has_markers_k = domain_integs_marker_k != nullptr;

      // if there are no markers, just use the whole linear form (1)
      if (!has_markers_k) { markers = 1; }
      else
      {
         // otherwise, scan the attributes to set the markers to 0 or 1
         const int NE = fes.GetNE();
         const auto attr = attributes.Read();
         const auto dimk = domain_integs_marker_k->Read();
         auto markers_w = markers.Write();
         MFEM_FORALL(e, NE, markers_w[e] = dimk[attr[e]-1] == 1;);
      }

      domain_integs[k]->AssembleFull(fes, markers, *lf);
   }
}

} // namespace mfem
