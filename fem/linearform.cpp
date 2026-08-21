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

// Implementation of class LinearForm

#include "fem.hpp"

#include <limits>
#include <memory>
#include <vector>

namespace mfem
{
namespace
{

/** @brief Return the reference-space tolerance used for delta containment.

    The delta-center classification is performed in reference coordinates, so
    this tolerance is independent of the physical size of the mesh element. The
    small multiple of machine epsilon leaves room for inverse-map and shape
    evaluation roundoff without widening the reference element by a
    mesh-scale-dependent amount. */
constexpr real_t DeltaReferenceTol()
{
   return 100.0 * std::numeric_limits<real_t>::epsilon();
}

/** @brief Return true if @a ip is in the reference element of type @a geom. */
bool ReferenceContains(Geometry::Type geom, const IntegrationPoint &ip)
{
   return Geometry::CheckPoint(geom, ip, DeltaReferenceTol());
}

/** @brief Return local vertices of the mesh entity containing @a ip.

    @param geom Element geometry used to select the linear nodal element.
    @param ip Reference-space point in the element.
    @param active Output array, overwritten with the local vertex numbers whose
    linear shape values are nonzero above a small reference-space tolerance. */
void ActiveEntityVertices(Geometry::Type geom, const IntegrationPoint &ip,
                          Array<int> &active)
{
   // Local vertices with nonzero linear shape values identify the mesh entity
   // containing the point.
   constexpr real_t entity_tol = DeltaReferenceTol();

   static const LinearFECollection nodal_fec;
   const FiniteElement &fe = *nodal_fec.FiniteElementForGeometry(geom);
   Vector shape(fe.GetDof());
   fe.CalcShape(ip, shape);
   active.SetSize(0);
   for (int i = 0; i < shape.Size(); i++)
   {
      if (shape(i) > entity_tol) { active.Append(i); }
   }
}

} // namespace

LinearForm::LinearForm(FiniteElementSpace *f, LinearForm *lf)
   : Vector(f->GetVSize())
{
   ext = nullptr;
   extern_lfs = 1;
   fast_assembly = false;
   fes = f;

   // Linear forms are stored on the device
   UseDevice(true);

   // Copy the pointers to the integrators and the corresponding marker arrays
   domain_integs = lf->domain_integs;
   domain_integs_marker = lf->domain_integs_marker;

   domain_delta_integs = lf->domain_delta_integs;
   domain_delta_integs_marker = lf->domain_delta_integs_marker;

   boundary_integs = lf->boundary_integs;
   boundary_integs_marker = lf->boundary_integs_marker;

   boundary_face_integs = lf->boundary_face_integs;
   boundary_face_integs_marker = lf->boundary_face_integs_marker;

   interior_face_integs = lf->interior_face_integs;
}

void LinearForm::AddDomainIntegrator(LinearFormIntegrator *lfi)
{
   DeltaLFIntegrator *maybe_delta =
      dynamic_cast<DeltaLFIntegrator *>(lfi);
   if (!maybe_delta || !maybe_delta->IsDelta())
   {
      domain_integs.Append(lfi);
      domain_integs_marker.Append(NULL);
   }
   else
   {
      domain_delta_integs.Append(maybe_delta);
      domain_delta_integs_marker.Append(NULL);
   }
}

void LinearForm::AddDomainIntegrator(LinearFormIntegrator *lfi,
                                     Array<int> &elem_marker)
{
   DeltaLFIntegrator *maybe_delta =
      dynamic_cast<DeltaLFIntegrator *>(lfi);
   if (!maybe_delta || !maybe_delta->IsDelta())
   {
      domain_integs.Append(lfi);
      domain_integs_marker.Append(&elem_marker);
   }
   else
   {
      domain_delta_integs.Append(maybe_delta);
      domain_delta_integs_marker.Append(&elem_marker);
   }
}

void LinearForm::AddBoundaryIntegrator (LinearFormIntegrator * lfi)
{
   boundary_integs.Append (lfi);
   boundary_integs_marker.Append(NULL); // NULL -> all attributes are active
}

void LinearForm::AddBoundaryIntegrator (LinearFormIntegrator * lfi,
                                        Array<int> &bdr_attr_marker)
{
   boundary_integs.Append (lfi);
   boundary_integs_marker.Append(&bdr_attr_marker);
}

void LinearForm::AddBdrFaceIntegrator (LinearFormIntegrator * lfi)
{
   boundary_face_integs.Append(lfi);
   // NULL -> all attributes are active
   boundary_face_integs_marker.Append(NULL);
}

void LinearForm::AddBdrFaceIntegrator(LinearFormIntegrator *lfi,
                                      Array<int> &bdr_attr_marker)
{
   boundary_face_integs.Append(lfi);
   boundary_face_integs_marker.Append(&bdr_attr_marker);
}

void LinearForm::AddInteriorFaceIntegrator(LinearFormIntegrator *lfi)
{
   interior_face_integs.Append(lfi);
}

bool LinearForm::SupportsDevice() const
{
   // return false for NURBS meshes, so we don’t convert it to non-NURBS
   // through Assemble, AssembleDevice, GetGeometricFactors and EnsureNodes
   const Mesh &mesh = *fes->GetMesh();
   if (mesh.NURBSext != nullptr) { return false; }

   // scan integrators to verify that all can use device assembly
   auto IntegratorsSupportDevice = [](const Array<LinearFormIntegrator*> &integ)
   {
      for (int k = 0; k < integ.Size(); k++)
      {
         if (!integ[k]->SupportsDevice()) { return false; }
      }
      return true;
   };

   if (!IntegratorsSupportDevice(domain_integs)) { return false; }
   if (!IntegratorsSupportDevice(boundary_integs)) { return false; }
   if (boundary_face_integs.Size() > 0 || interior_face_integs.Size() > 0 ||
       domain_delta_integs.Size() > 0) { return false; }

   if (boundary_integs.Size() > 0)
   {
      // Make sure there are no boundary faces that are not boundary elements
      if (fes->GetNFbyType(FaceType::Boundary) != fes->GetNBE())
      {
         return false;
      }
      // Make sure every boundary element corresponds to a boundary face
      for (int be = 0; be < fes->GetNBE(); ++be)
      {
         const int f = mesh.GetBdrElementFaceIndex(be);
         const auto face_info = mesh.GetFaceInformation(f);
         if (!face_info.IsBoundary())
         {
            return false;
         }
      }
   }

   // no support for elements with varying polynomial orders
   if (fes->IsVariableOrder()) { return false; }

   // no support for 1D and embedded meshes
   const int mesh_dim = mesh.Dimension();
   if (mesh_dim == 1 || mesh_dim != mesh.SpaceDimension()) { return false; }

   // tensor-product finite element space only
   if (!UsesTensorBasis(*fes)) { return false; }

   return true;
}

void LinearForm::UseFastAssembly(bool use_fa)
{
   fast_assembly = use_fa;

   if (fast_assembly && SupportsDevice() && !ext)
   {
      ext = new LinearFormExtension(this);
   }
}

void LinearForm::Assemble()
{
   Array<int> vdofs;
   ElementTransformation *eltrans;
   Vector elemvect;

   Vector::operator=(0.0);

   // The above operation is executed on device because of UseDevice().
   // The first use of AddElementVector() below will move it back to host
   // because both 'vdofs' and 'elemvect' are on host.

   if (fast_assembly && ext) { return ext->Assemble(); }

   if (domain_integs.Size())
   {
      for (int k = 0; k < domain_integs.Size(); k++)
      {
         if (domain_integs_marker[k] != NULL)
         {
            MFEM_VERIFY(domain_integs_marker[k]->Size() ==
                        (fes->GetMesh()->attributes.Size() ?
                         fes->GetMesh()->attributes.Max() : 0),
                        "invalid element marker for domain linear form "
                        "integrator #" << k << ", counting from zero");
         }
      }

      DofTransformation doftrans;
      for (int i = 0; i < fes -> GetNE(); i++)
      {
         int elem_attr = fes->GetMesh()->GetAttribute(i);
         for (int k = 0; k < domain_integs.Size(); k++)
         {
            const Array<int> * const markers = domain_integs_marker[k];
            if (markers) { markers->HostRead(); }
            if ( markers == NULL || (*markers)[elem_attr-1] == 1 )
            {
               fes -> GetElementVDofs (i, vdofs, doftrans);
               eltrans = fes -> GetElementTransformation (i);
               domain_integs[k]->AssembleRHSElementVect(*fes->GetFE(i),
                                                        *eltrans, elemvect);
               doftrans.TransformDual(elemvect);
               AddElementVector (vdofs, elemvect);
            }
         }
      }
   }
   AssembleDelta();

   if (boundary_integs.Size())
   {
      Mesh *mesh = fes->GetMesh();

      // Which boundary attributes need to be processed?
      Array<int> bdr_attr_marker(mesh->bdr_attributes.Size() ?
                                 mesh->bdr_attributes.Max() : 0);
      bdr_attr_marker = 0;
      for (int k = 0; k < boundary_integs.Size(); k++)
      {
         if (boundary_integs_marker[k] == NULL)
         {
            bdr_attr_marker = 1;
            break;
         }
         Array<int> &bdr_marker = *boundary_integs_marker[k];
         MFEM_ASSERT(bdr_marker.Size() == bdr_attr_marker.Size(),
                     "invalid boundary marker for boundary integrator #"
                     << k << ", counting from zero");
         for (int i = 0; i < bdr_attr_marker.Size(); i++)
         {
            bdr_attr_marker[i] |= bdr_marker[i];
         }
      }

      DofTransformation doftrans;
      for (int i = 0; i < fes -> GetNBE(); i++)
      {
         const int bdr_attr = mesh->GetBdrAttribute(i);
         if (bdr_attr_marker[bdr_attr-1] == 0) { continue; }
         fes -> GetBdrElementVDofs (i, vdofs, doftrans);
         eltrans = fes -> GetBdrElementTransformation (i);
         for (int k=0; k < boundary_integs.Size(); k++)
         {
            if (boundary_integs_marker[k] &&
                (*boundary_integs_marker[k])[bdr_attr-1] == 0) { continue; }

            boundary_integs[k]->AssembleRHSElementVect(*fes->GetBE(i),
                                                       *eltrans, elemvect);
            doftrans.TransformDual(elemvect);
            AddElementVector (vdofs, elemvect);
         }
      }
   }
   if (boundary_face_integs.Size())
   {
      FaceElementTransformations *tr;
      Mesh *mesh = fes->GetMesh();

      // Which boundary attributes need to be processed?
      Array<int> bdr_attr_marker(mesh->bdr_attributes.Size() ?
                                 mesh->bdr_attributes.Max() : 0);
      bdr_attr_marker = 0;
      for (int k = 0; k < boundary_face_integs.Size(); k++)
      {
         if (boundary_face_integs_marker[k] == NULL)
         {
            bdr_attr_marker = 1;
            break;
         }
         Array<int> &bdr_marker = *boundary_face_integs_marker[k];
         MFEM_ASSERT(bdr_marker.Size() == bdr_attr_marker.Size(),
                     "invalid boundary marker for boundary face integrator #"
                     << k << ", counting from zero");
         for (int i = 0; i < bdr_attr_marker.Size(); i++)
         {
            bdr_attr_marker[i] |= bdr_marker[i];
         }
      }

      for (int i = 0; i < mesh->GetNBE(); i++)
      {
         const int bdr_attr = mesh->GetBdrAttribute(i);
         if (bdr_attr_marker[bdr_attr-1] == 0) { continue; }

         tr = mesh->GetBdrFaceTransformations(i);
         if (tr != NULL)
         {
            mfem::DofTransformation doftrans;
            fes -> GetElementVDofs (tr -> Elem1No, vdofs, doftrans);
            for (int k = 0; k < boundary_face_integs.Size(); k++)
            {
               if (boundary_face_integs_marker[k] &&
                   (*boundary_face_integs_marker[k])[bdr_attr-1] == 0)
               { continue; }

               boundary_face_integs[k]->
               AssembleRHSElementVect(*fes->GetFE(tr->Elem1No),
                                      *tr, elemvect);
               doftrans.TransformDual(elemvect);
               AddElementVector (vdofs, elemvect);
            }
         }
      }
   }

   if (interior_face_integs.Size())
   {
      Mesh *mesh = fes->GetMesh();

      for (int k = 0; k < interior_face_integs.Size(); k++)
      {
         for (int i = 0; i < mesh->GetNumFaces(); i++)
         {
            FaceElementTransformations *tr = NULL;
            tr = mesh->GetInteriorFaceTransformations (i);
            if (tr != NULL)
            {
               fes -> GetElementVDofs (tr -> Elem1No, vdofs);
               Array<int> vdofs2;
               fes -> GetElementVDofs (tr -> Elem2No, vdofs2);
               vdofs.Append(vdofs2);
               interior_face_integs[k]->
               AssembleRHSElementVect(*fes->GetFE(tr->Elem1No),
                                      *fes->GetFE(tr->Elem2No),
                                      *tr, elemvect);
               AddElementVector (vdofs, elemvect);
            }
         }
      }
   }
}

void LinearForm::Update()
{
   SetSize(fes->GetVSize());
   if (ext) { ext->Update(); }
}

void LinearForm::Update(FiniteElementSpace *f, Vector &v, int v_offset)
{
   MFEM_ASSERT(v.Size() >= v_offset + f->GetVSize(), "");
   fes = f;
   v.UseDevice(true);
   this->Vector::MakeRef(v, v_offset, fes->GetVSize());
   if (ext) { ext->Update(); }
}

void LinearForm::MakeRef(FiniteElementSpace *f, Vector &v, int v_offset)
{
   Update(f, v, v_offset);
}

void LinearForm::FindDeltaCenters(DenseMatrix &centers, Array<int> &elem_ids,
                                  Array<IntegrationPoint> &ips, bool warn)
{
   fes->GetMesh()->FindPoints(centers, elem_ids, ips, warn);
}

bool LinearForm::DeltaElementMarked(int i, int elem) const
{
   const Array<int> * const marker = domain_delta_integs_marker[i];
   if (!marker) { return true; }

   Mesh *mesh = fes->GetMesh();
   MFEM_VERIFY(marker->Size() == (mesh->attributes.Size() ?
                                  mesh->attributes.Max() : 0),
               "invalid element marker for domain delta linear form "
               "integrator #" << i << ", counting from zero");
   marker->HostRead();
   return (*marker)[mesh->GetAttribute(elem)-1] == 1;
}

void LinearForm::ComputeDeltaLocations(
   std::vector<std::vector<DeltaElement>> &locations)
{
   const int ndi = domain_delta_integs.Size();
   MFEM_ASSERT(ndi > 0, "");
   Mesh *mesh = fes->GetMesh();
   const int sdim = mesh->SpaceDimension();

   auto has_location = [](const std::vector<DeltaElement> &locs, int elem)
   {
      for (const DeltaElement &loc : locs)
      {
         if (loc.elem == elem) { return true; }
      }
      return false;
   };

   auto add_location = [&has_location](std::vector<DeltaElement> &locs,
                                       int elem, const IntegrationPoint &ip)
   {
      if (!has_location(locs, elem)) { locs.push_back({elem, ip}); }
   };

   DenseMatrix centers(sdim, ndi);
   {
      Vector center;
      for (int i = 0; i < ndi; i++)
      {
         centers.GetColumnReference(i, center);
         domain_delta_integs[i]->GetDeltaCenter(center);
         MFEM_VERIFY(center.Size() == sdim,
                     "Point dim " << center.Size() <<
                     " does not match space dim " << sdim);
      }
   }
   // Find an element containing each center (see FindDeltaCenters).
   Array<int> anchor_elem_id;
   Array<IntegrationPoint> anchor_ip;
   FindDeltaCenters(centers, anchor_elem_id, anchor_ip, false);

   locations.assign(ndi, std::vector<DeltaElement> {});

   // On conforming meshes, containing elements share all vertices of the
   // active entity. This avoids relying on TransformResult::Inside for
   // boundary points.
   if (mesh->Conforming())
   {
      std::unique_ptr<Table> vte(mesh->GetVertexToElementTable());

      InverseElementTransformation inv_tr;
      inv_tr.SetSolverType(InverseElementTransformation::Newton);
      Vector pt(sdim);
      Array<int> anchor_verts, cand_verts, active_local, entity_verts;

      for (int i = 0; i < ndi; i++)
      {
         const int e0 = anchor_elem_id[i];
         if (e0 < 0) { continue; }

         for (int d = 0; d < sdim; d++) { pt(d) = centers(d, i); }

         auto &locs = locations[i];
         add_location(locs, e0, anchor_ip[i]);

         mesh->GetElementVertices(e0, anchor_verts);
         ActiveEntityVertices(mesh->GetElementBaseGeometry(e0), anchor_ip[i],
                              active_local);

         // Mesh-vertex ids of the entity the center sits on.
         entity_verts.SetSize(0);
         for (int a : active_local) { entity_verts.Append(anchor_verts[a]); }

         MFEM_VERIFY(entity_verts.Size() > 0,
                     "could not identify delta center entity");

         // If the active entity is the whole element, the center is an
         // interior point and the anchor is the only containing element.
         // Otherwise, inspect neighbors sharing all vertices of the
         // lower-dimensional active entity.
         if (entity_verts.Size() < anchor_verts.Size())
         {
            // Candidates: elements incident to the first entity vertex.
            const int v0 = entity_verts[0];
            const int n_adj = vte->RowSize(v0);
            const int *adj = vte->GetRow(v0);
            for (int k = 0; k < n_adj; k++)
            {
               const int en = adj[k];
               if (en == e0 || has_location(locs, en)) { continue; }
               mesh->GetElementVertices(en, cand_verts);
               bool shares_all = true;
               for (int ev : entity_verts)
               {
                  if (cand_verts.Find(ev) < 0) { shares_all = false; break; }
               }
               if (!shares_all) { continue; }
               // Topologically contains the active entity. Compute the
               // reference point for assembly and verify containment in
               // scale-independent reference space.
               inv_tr.SetTransformation(*fes->GetElementTransformation(en));
               IntegrationPoint ip_n;
               const int result = inv_tr.Transform(pt, ip_n);
               if (result == InverseElementTransformation::Unknown ||
                   !ReferenceContains(mesh->GetElementBaseGeometry(en), ip_n))
               {
                  continue;
               }
               add_location(locs, en, ip_n);
            }
         }
      }
   }
   else
   {
      // Use NCMesh closure nodes to bridge master/slave interfaces where a
      // hanging vertex is not a coarse element corner.
      const int ne = mesh->GetNE();
      InverseElementTransformation inv_tr;
      inv_tr.SetSolverType(InverseElementTransformation::Newton);
      Vector pt(sdim);
      Array<int> active_local, candidates;

      for (int i = 0; i < ndi; i++)
      {
         const int e0 = anchor_elem_id[i];
         if (e0 < 0) { continue; }

         for (int d = 0; d < sdim; d++) { pt(d) = centers(d, i); }
         ActiveEntityVertices(mesh->GetElementBaseGeometry(e0), anchor_ip[i],
                              active_local);

         auto &locs = locations[i];
         // Seed the anchor found by FindDeltaCenters. It is already known to
         // contain the center, so do not require the local inverse map below to
         // rediscover it.
         add_location(locs, e0, anchor_ip[i]);

         mesh->ncmesh->FindClosureElements(e0, active_local, candidates);
         for (int e : candidates)
         {
            if (e == e0 || e >= ne || has_location(locs, e)) { continue; }
            // The containment filter is applied in reference space: once the
            // physical center is mapped back to this candidate, the reference
            // element is straight-sided and its bounds are scale-independent.
            inv_tr.SetTransformation(*fes->GetElementTransformation(e));
            IntegrationPoint ip_e;
            const int result = inv_tr.Transform(pt, ip_e);
            if (result == InverseElementTransformation::Unknown ||
                !ReferenceContains(mesh->GetElementBaseGeometry(e), ip_e))
            {
               continue;
            }
            add_location(locs, e, ip_e);
         }
      }
   }
}

void LinearForm::SyncDeltaCounts(Array<int> &counts)
{
   MFEM_CONTRACT_VAR(counts);
}

void LinearForm::AssembleDelta()
{
   if (domain_delta_integs.Size() == 0) { return; }

   std::vector<std::vector<DeltaElement>> locations;
   ComputeDeltaLocations(locations);

   Array<int> counts(domain_delta_integs.Size());
   counts = 0;
   for (int i = 0; i < domain_delta_integs.Size(); i++)
   {
      for (const DeltaElement &loc : locations[i])
      {
         if (DeltaElementMarked(i, loc.elem)) { counts[i]++; }
      }
   }
   SyncDeltaCounts(counts);

   Array<int> vdofs;
   Vector elemvect;
   for (int i = 0; i < domain_delta_integs.Size(); i++)
   {
      const int n = counts[i];
      if (n <= 0) { continue; }

      const real_t weight = 1.0 / static_cast<real_t>(n);
      for (const DeltaElement &loc : locations[i])
      {
         if (!DeltaElementMarked(i, loc.elem)) { continue; }
         ElementTransformation &Trans = *fes->GetElementTransformation(loc.elem);
         Trans.SetIntPoint(&loc.ip);

         fes->GetElementVDofs(loc.elem, vdofs);
         domain_delta_integs[i]->AssembleDeltaElementVect(*fes->GetFE(loc.elem),
                                                          Trans, elemvect);
         elemvect *= weight;
         AddElementVector(vdofs, elemvect);
      }
   }
}

LinearForm & LinearForm::operator=(real_t value)
{
   Vector::operator=(value);
   return *this;
}

LinearForm & LinearForm::operator=(const Vector &v)
{
   MFEM_ASSERT(fes && v.Size() == fes->GetVSize(), "");
   Vector::operator=(v);
   return *this;
}

LinearForm::~LinearForm()
{
   if (!extern_lfs)
   {
      int k;
      for (k=0; k < domain_delta_integs.Size(); k++)
      { delete domain_delta_integs[k]; }
      for (k=0; k < domain_integs.Size(); k++) { delete domain_integs[k]; }
      for (k=0; k < boundary_integs.Size(); k++) { delete boundary_integs[k]; }
      for (k=0; k < boundary_face_integs.Size(); k++)
      { delete boundary_face_integs[k]; }
      for (k=0; k < interior_face_integs.Size(); k++)
      { delete interior_face_integs[k]; }
   }

   delete ext;
}

}
