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
   const int mesh_attributes_max = fes.GetMesh()->attributes.Size() ?
                                   fes.GetMesh()->attributes.Max() : 0;
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
         MFEM_VERIFY(mesh_attributes_max == domain_integs_marker_k->Size(),
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
      if (k == 0) { elem_restrict_lex->MultTranspose(b, *lf); }
      else { elem_restrict_lex->AddMultTranspose(b, *lf); }
   }

   const Array<Array<int>*> &boundary_integs_marker = lf->boundary_integs_marker;
   const int bdr_attributes_max = fes.GetMesh()->bdr_attributes.Size() ?
                                  fes.GetMesh()->bdr_attributes.Max() : 0;
   const Array<LinearFormIntegrator*> &boundary_integs = lf->boundary_integs;

   for (int k = 0; k < boundary_integs.Size(); ++k)
   {
      // Get the markers for this integrator
      const Array<int> *boundary_integs_marker_k = boundary_integs_marker[k];

      // check if there are markers for this integrator
      const bool has_markers_k = boundary_integs_marker_k != nullptr;

      if (has_markers_k)
      {
         // Element attribute marker should be of length mesh->attributes
         MFEM_VERIFY(bdr_attributes_max == boundary_integs_marker_k->Size(),
                     "invalid boundary marker for boundary linear form "
                     "integrator #" << k << ", counting from zero");
      }

      // if there are no markers, just use the whole linear form (1)
      if (!has_markers_k) { bdr_markers.HostReadWrite(); bdr_markers = 1; }
      else
      {
         // scan the attributes to set the markers to 0 or 1
         const int NBE = bdr_attributes.Size();
         const auto attr = bdr_attributes.Read();
         const auto attr_markers = boundary_integs_marker_k->Read();
         auto markers_w = bdr_markers.Write();
         MFEM_FORALL(e, NBE, markers_w[e] = attr_markers[attr[e]-1] == 1;);
      }

      // Assemble the linear form
      bdr_b = 0.0;
      boundary_integs[k]->AssembleDevice(fes, bdr_markers, bdr_b);
      bdr_restrict_lex->AddMultTranspose(bdr_b, *lf);
   }
}

void LinearFormExtension::Update()
{
   const FiniteElementSpace &fes = *lf->FESpace();
   const Mesh &mesh = *fes.GetMesh();
   constexpr ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;

   MFEM_VERIFY(lf->Size() == fes.GetVSize(), "");

   if (lf->domain_integs.Size() > 0)
   {
      const int NE = fes.GetNE();
      markers.SetSize(NE);
      //markers.UseDevice(true);

      // Gather the attributes on the host from all the elements
      attributes.SetSize(NE);
      for (int i = 0; i < NE; ++i) { attributes[i] = mesh.GetAttribute(i); }

      elem_restrict_lex = fes.GetElementRestriction(ordering);
      MFEM_VERIFY(elem_restrict_lex, "Element restriction not available");
      b.SetSize(elem_restrict_lex->Height(), Device::GetMemoryType());
      b.UseDevice(true);
   }

   if (lf->boundary_integs.Size() > 0)
   {
      const int nf_bdr = fes.GetNFbyType(FaceType::Boundary);
      bdr_markers.SetSize(nf_bdr);
      // bdr_markers.UseDevice(true);

      // The face restriction will give us "face E-vectors" on the boundary that
      // are numbered in the order of the faces of mesh. This numbering will be
      // different than the numbering of the boundary elements. We compute
      // mappings so that the array `bdr_attributes[i]` gives the boundary
      // attribute of the `i`th boundary face in the mesh face order.
      std::unordered_map<int,int> f_to_be;
      for (int i = 0; i < mesh.GetNBE(); ++i)
      {
         const int f = mesh.GetBdrElementEdgeIndex(i);
         f_to_be[f] = i;
      }
      MFEM_VERIFY(size_t(nf_bdr) == f_to_be.size(), "Incompatible sizes");
      bdr_attributes.SetSize(nf_bdr);
      int f_ind = 0;
      for (int f = 0; f < mesh.GetNumFaces(); ++f)
      {
         if (f_to_be.find(f) != f_to_be.end())
         {
            const int be = f_to_be[f];
            bdr_attributes[f_ind] = mesh.GetBdrAttribute(be);
            ++f_ind;
         }
      }

      bdr_restrict_lex =
         dynamic_cast<const FaceRestriction*>(
            fes.GetFaceRestriction(ordering, FaceType::Boundary,
                                   L2FaceValues::SingleValued));
      MFEM_VERIFY(bdr_restrict_lex, "Face restriction not available");
      bdr_b.SetSize(bdr_restrict_lex->Height(), Device::GetMemoryType());
      bdr_b.UseDevice(true);
   }
}

} // namespace mfem
