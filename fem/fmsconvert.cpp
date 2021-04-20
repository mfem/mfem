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

#include "../config/config.hpp"

#ifdef MFEM_USE_FMS

#include "fmsconvert.hpp"
#include <climits>

using std::endl;

// #define DEBUG_FMS_MFEM 1
// #define DEBUG_MFEM_FMS 1

namespace mfem
{

static inline int
FmsBasisTypeToMfemBasis(FmsBasisType b)
{
   int retval = -1;
   switch (b)
   {
      case FMS_NODAL_GAUSS_OPEN:
         retval = mfem::BasisType::GaussLegendre;;
         break;
      case FMS_NODAL_GAUSS_CLOSED:
         retval = mfem::BasisType::GaussLobatto;;
         break;
      case FMS_POSITIVE:
         retval = mfem::BasisType::Positive;;
         break;
      case FMS_NODAL_UNIFORM_OPEN:
         retval = mfem::BasisType::OpenUniform;
         break;
      case FMS_NODAL_UNIFORM_CLOSED:
         retval = mfem::BasisType::ClosedUniform;
         break;
      case FMS_NODAL_CHEBYSHEV_OPEN:
      case FMS_NODAL_CHEBYSHEV_CLOSED:
         mfem::out <<
                   "FMS_NODAL_CHEBYSHEV_OPEN, FMS_NODAL_CHEBYSHEV_CLOSED need conversion to MFEM types."
                   << endl;
         break;
   }
   return retval;
}

/**
@brief Get the order and layout of the field.
*/
static int
FmsFieldGetOrderAndLayout(FmsField f, FmsInt *f_order, FmsLayoutType *f_layout)
{
   int err = 0;
   FmsFieldDescriptor fd;
   FmsLayoutType layout;
   FmsScalarType data_type;
   const void *data = nullptr;
   FmsInt order = 0;

   FmsFieldGet(f, &fd, NULL, &layout, &data_type,
               &data);

   FmsFieldDescriptorType f_fd_type;
   FmsFieldDescriptorGetType(fd, &f_fd_type);
   if (f_fd_type != FMS_FIXED_ORDER)
   {
      err = 1;
   }
   else
   {
      FmsFieldType field_type;
      FmsBasisType basis_type;
      FmsFieldDescriptorGetFixedOrder(fd, &field_type,
                                      &basis_type, &order);
   }

   *f_order = order;
   *f_layout = layout;

   return err;
}

/**
@brief This function converts an FmsField to an MFEM GridFunction.
@note I took some of the Pumi example code from the mesh conversion function
      that converted coordinates and am trying to make it more general.
      Coordinates are just another field so it seems like a good starting
      point. We still have to support a bunch of other function types, etc.
*/
template <typename DataType>
int
FmsFieldToGridFunction(FmsMesh fms_mesh, FmsField f, Mesh *mesh,
                       GridFunction &func, bool setFE)
{
   int err = 0;

   // NOTE: transplanted from the FmsMeshToMesh function
   //       We should do this work once and save it.
   //--------------------------------------------------
   FmsInt dim, n_vert, n_elem, n_bdr_elem, space_dim;

   // Find the first component that has coordinates - that will be the new mfem
   // mesh.
   FmsInt num_comp;
   FmsMeshGetNumComponents(fms_mesh, &num_comp);
   FmsComponent main_comp = NULL;
   FmsField coords = NULL;
   for (FmsInt comp_id = 0; comp_id < num_comp; comp_id++)
   {
      FmsComponent comp;
      FmsMeshGetComponent(fms_mesh, comp_id, &comp);
      FmsComponentGetCoordinates(comp, &coords);
      if (coords) { main_comp = comp; break; }
   }
   if (!main_comp) { return 1; }
   FmsComponentGetDimension(main_comp, &dim);
   FmsComponentGetNumEntities(main_comp, &n_elem);
   FmsInt n_ents[FMS_NUM_ENTITY_TYPES];
   FmsInt n_main_parts;
   FmsComponentGetNumParts(main_comp, &n_main_parts);
   for (FmsInt et = FMS_VERTEX; et < FMS_NUM_ENTITY_TYPES; et++)
   {
      n_ents[et] = 0;
      for (FmsInt part_id = 0; part_id < n_main_parts; part_id++)
      {
         FmsInt num_ents;
         FmsComponentGetPart(main_comp, part_id, (FmsEntityType)et, NULL, NULL,
                             NULL, NULL, &num_ents);
         n_ents[et] += num_ents;
      }
   }
   n_vert = n_ents[FMS_VERTEX];
   //--------------------------------------------------

   // Interrogate the field.
   FmsFieldDescriptor f_fd;
   FmsLayoutType f_layout;
   FmsScalarType f_data_type;
   const void *f_data;
   FmsFieldGet(f, &f_fd, &space_dim, &f_layout, &f_data_type,
               &f_data);
   // FmsFieldGet(coords, NULL, &space_dim, NULL, NULL, NULL);

   FmsInt f_num_dofs;
   FmsFieldDescriptorGetNumDofs(f_fd, &f_num_dofs);

   // Access FMS data through this typed pointer.
   auto src_data = reinterpret_cast<const DataType *>(f_data);

   FmsFieldDescriptorType f_fd_type;
   FmsFieldDescriptorGetType(f_fd, &f_fd_type);
   if (f_fd_type != FMS_FIXED_ORDER)
   {
      return 9;
   }
   FmsFieldType f_field_type;
   FmsBasisType f_basis_type;
   FmsInt f_order;
   FmsFieldDescriptorGetFixedOrder(f_fd, &f_field_type,
                                   &f_basis_type, &f_order);

   if (f_field_type != FMS_CONTINUOUS && f_field_type != FMS_DISCONTINUOUS &&
       f_field_type != FMS_HDIV)
   {
      return 10;
   }

   int btype = FmsBasisTypeToMfemBasis(f_basis_type);
   if (btype < 0)
   {
      mfem::out << "\tInvalid BasisType: " << BasisType::Name(btype) << std::endl;
      return 11;
   }

   //------------------------------------------------------------------
   if (setFE)
   {
      // We could assemble a name based on fe_coll.hpp rules and pass to
      // FiniteElementCollection::New()

      mfem::FiniteElementCollection *fec = nullptr;
      switch (f_field_type)
      {
         case FMS_DISCONTINUOUS:
            fec = new L2_FECollection(f_order, dim, btype);
            break;
         case FMS_CONTINUOUS:
            fec = new H1_FECollection(f_order, dim, btype);
            break;
         case FMS_HDIV:
            fec = new RT_FECollection(f_order, dim);
            break;
      }

      int ordering = (f_layout == FMS_BY_VDIM) ? Ordering::byVDIM : Ordering::byNODES;
      auto fes = new FiniteElementSpace(mesh, fec, space_dim, ordering);
      func.SetSpace(fes);
   }
   //------------------------------------------------------------------
   const FmsInt nstride = (f_layout == FMS_BY_VDIM) ? space_dim : 1;
   const FmsInt vstride = (f_layout == FMS_BY_VDIM) ? 1 : f_num_dofs;

   // Data reordering to store the data into func.
   if ((FmsInt)(func.Size()) != f_num_dofs*space_dim)
   {
      return 12;
   }

   mfem::FiniteElementSpace *fes = func.FESpace();
   const int vdim = fes->GetVDim();
   const mfem::FiniteElementCollection *fec = fes->FEColl();
   const int vert_dofs = fec->DofForGeometry(mfem::Geometry::POINT);
   const int edge_dofs = fec->DofForGeometry(mfem::Geometry::SEGMENT);
   const int tri_dofs = fec->DofForGeometry(mfem::Geometry::TRIANGLE);
   const int quad_dofs = fec->DofForGeometry(mfem::Geometry::SQUARE);
   const int tet_dofs = fec->DofForGeometry(mfem::Geometry::TETRAHEDRON);
   const int hex_dofs = fec->DofForGeometry(mfem::Geometry::CUBE);
   int ent_dofs[FMS_NUM_ENTITY_TYPES];
   ent_dofs[FMS_VERTEX] = vert_dofs;
   ent_dofs[FMS_EDGE] = edge_dofs;
   ent_dofs[FMS_TRIANGLE] = tri_dofs;
   ent_dofs[FMS_QUADRILATERAL] = quad_dofs;
   ent_dofs[FMS_TETRAHEDRON] = tet_dofs;
   ent_dofs[FMS_HEXAHEDRON] = hex_dofs;
   FmsInt fms_dof_offset = 0;
   int mfem_ent_cnt[4] = {0,0,0,0}; // mfem entity counters, by dimension
   int mfem_last_vert_cnt = 0;
   mfem::HashTable<mfem::Hashed2> mfem_edge;
   mfem::HashTable<mfem::Hashed4> mfem_face;
   if (dim >= 2 && edge_dofs > 0)
   {
      mfem::Array<int> ev;
      for (int i = 0; i < mesh->GetNEdges(); i++)
      {
         mesh->GetEdgeVertices(i, ev);
         int id = mfem_edge.GetId(ev[0], ev[1]);
         if (id != i) { return 13; }
      }
   }
   if (dim >= 3 &&
       ((n_ents[FMS_TRIANGLE] > 0 && tri_dofs > 0) ||
        (n_ents[FMS_QUADRILATERAL] > 0 && quad_dofs > 0)))
   {
      mfem::Array<int> fv;
      for (int i = 0; i < mesh->GetNFaces(); i++)
      {
         mesh->GetFaceVertices(i, fv);
         if (fv.Size() == 3) { fv.Append(INT_MAX); }
         // HashTable uses the smallest 3 of the 4 indices to hash Hashed4
         int id = mfem_face.GetId(fv[0], fv[1], fv[2], fv[3]);
         if (id != i) { return 14; }
      }
   }

   // Loop over all parts of the main component
   for (FmsInt part_id = 0; part_id < n_main_parts; part_id++)
   {
      // Loop over all entity types in the part
      for (FmsInt et = FMS_VERTEX; et < FMS_NUM_ENTITY_TYPES; et++)
      {
         FmsDomain domain;
         FmsIntType ent_id_type;
         const void *ents;
         const FmsOrientation *ents_ori;
         FmsInt num_ents;
         FmsComponentGetPart(main_comp, part_id, (FmsEntityType)et, &domain,
                             &ent_id_type, &ents, &ents_ori, &num_ents);
         if (num_ents == 0) { continue; }
         if (ent_dofs[et] == 0)
         {
            if (et == FMS_VERTEX) { mfem_last_vert_cnt = mfem_ent_cnt[et]; }
            mfem_ent_cnt[FmsEntityDim[et]] += num_ents;
            continue;
         }

         if (ents != NULL &&
             (ent_id_type != FMS_INT32 && ent_id_type != FMS_UINT32))
         {
            return 15;
         }
         if (ents_ori != NULL)
         {
            return 16;
         }

         if (et == FMS_VERTEX)
         {
            const int mfem_dof_offset = mfem_ent_cnt[0]*vert_dofs;
            for (FmsInt i = 0; i < num_ents*vert_dofs; i++)
            {
               for (int j = 0; j < vdim; j++)
               {
                  const int idx = i*nstride+j*vstride;
                  func(mfem_dof_offset*nstride+idx) =
                     static_cast<double>(src_data[fms_dof_offset*nstride+idx]);
               }
            }
            fms_dof_offset += num_ents*vert_dofs;
            mfem_last_vert_cnt = mfem_ent_cnt[et];
            mfem_ent_cnt[0] += num_ents;
            continue;
         }
         mfem::Array<int> dofs;
         if (FmsEntityDim[et] == dim)
         {
            for (FmsInt e = 0; e < num_ents; e++)
            {
               fes->GetElementInteriorDofs(mfem_ent_cnt[dim]+e, dofs);
               for (int i = 0; i < ent_dofs[et]; i++, fms_dof_offset++)
               {
                  for (int j = 0; j < vdim; j++)
                  {
                     func(fes->DofToVDof(dofs[i],j)) =
                        static_cast<double>(src_data[fms_dof_offset*nstride+j*vstride]);
                  }
               }
            }
            mfem_ent_cnt[dim] += num_ents;
            continue;
         }
         const FmsInt nv = FmsEntityNumVerts[et];
         mfem::Array<int> ents_verts(num_ents*nv), m_ev;
         const int *ei = (const int *)ents;
         if (ents == NULL)
         {
            FmsDomainGetEntitiesVerts(domain, (FmsEntityType)et, NULL, FMS_INT32,
                                      0, ents_verts.GetData(), num_ents);
         }
         else
         {
            for (FmsInt i = 0; i < num_ents; i++)
            {
               FmsDomainGetEntitiesVerts(domain, (FmsEntityType)et, NULL,
                                         FMS_INT32, ei[i], &ents_verts[i*nv], 1);
            }
         }
         for (int i = 0; i < ents_verts.Size(); i++)
         {
            ents_verts[i] += mfem_last_vert_cnt;
         }
         const int *perm;
         switch ((FmsEntityType)et)
         {
            case FMS_EDGE:
            {
               for (FmsInt part_ent_id = 0; part_ent_id < num_ents; part_ent_id++)
               {
                  const int *ev = &ents_verts[2*part_ent_id];
                  int mfem_edge_id = mfem_edge.FindId(ev[0], ev[1]);
                  if (mfem_edge_id < 0)
                  {
                     return 17;
                  }
                  mesh->GetEdgeVertices(mfem_edge_id, m_ev);
                  int ori = (ev[0] == m_ev[0]) ? 0 : 1;
                  perm = fec->DofOrderForOrientation(mfem::Geometry::SEGMENT, ori);
                  fes->GetEdgeInteriorDofs(mfem_edge_id, dofs);
                  for (int i = 0; i < edge_dofs; i++)
                  {
                     for (int j = 0; j < vdim; j++)
                     {
                        func(fes->DofToVDof(dofs[i],j)) =
                           static_cast<double>(src_data[(fms_dof_offset+perm[i])*nstride+j*vstride]);
                     }
                  }
                  fms_dof_offset += edge_dofs;
               }
               break;
            }
            case FMS_TRIANGLE:
            {
               for (FmsInt part_ent_id = 0; part_ent_id < num_ents; part_ent_id++)
               {
                  const int *tv = &ents_verts[3*part_ent_id];
                  int mfem_face_id = mfem_face.FindId(tv[0], tv[1], tv[2], INT_MAX);
                  if (mfem_face_id < 0)
                  {
                     return 18;
                  }
                  mesh->GetFaceVertices(mfem_face_id, m_ev);
                  int ori = 0;
                  while (tv[ori] != m_ev[0]) { ori++; }
                  ori = (tv[(ori+1)%3] == m_ev[1]) ? 2*ori : 2*ori+1;
                  perm = fec->DofOrderForOrientation(mfem::Geometry::TRIANGLE, ori);
                  fes->GetFaceInteriorDofs(mfem_face_id, dofs);
                  for (int i = 0; i < tri_dofs; i++)
                  {
                     for (int j = 0; j < vdim; j++)
                     {
                        func(fes->DofToVDof(dofs[i],j)) =
                           static_cast<double>(src_data[(fms_dof_offset+perm[i])*nstride+j*vstride]);
                     }
                  }
                  fms_dof_offset += tri_dofs;
               }
               break;
            }
            case FMS_QUADRILATERAL:
            {
               for (FmsInt part_ent_id = 0; part_ent_id < num_ents; part_ent_id++)
               {
                  const int *qv = &ents_verts[4*part_ent_id];
                  int mfem_face_id = mfem_face.FindId(qv[0], qv[1], qv[2], qv[3]);
                  if (mfem_face_id < 0) { return 19; }
                  mesh->GetFaceVertices(mfem_face_id, m_ev);
                  int ori = 0;
                  while (qv[ori] != m_ev[0]) { ori++; }
                  ori = (qv[(ori+1)%4] == m_ev[1]) ? 2*ori : 2*ori+1;
                  perm = fec->DofOrderForOrientation(mfem::Geometry::SQUARE, ori);
                  fes->GetFaceInteriorDofs(mfem_face_id, dofs);
                  for (int i = 0; i < quad_dofs; i++)
                  {
                     for (int j = 0; j < vdim; j++)
                     {
                        func(fes->DofToVDof(dofs[i],j)) =
                           static_cast<double>(src_data[(fms_dof_offset+perm[i])*nstride+j*vstride]);
                     }
                  }
                  fms_dof_offset += quad_dofs;
               }
               break;
            }
            default: break;
         }
         mfem_ent_cnt[FmsEntityDim[et]] += num_ents;
      }
   }

   return err;
}

int
FmsMeshToMesh(FmsMesh fms_mesh, Mesh **mfem_mesh)
{
   FmsInt dim, n_vert, n_elem, n_bdr_elem, space_dim;

   // Find the first component that has coordinates - that will be the new mfem
   // mesh.
   FmsInt num_comp;
   FmsMeshGetNumComponents(fms_mesh, &num_comp);
   FmsComponent main_comp = NULL;
   FmsField coords = NULL;
   for (FmsInt comp_id = 0; comp_id < num_comp; comp_id++)
   {
      FmsComponent comp;
      FmsMeshGetComponent(fms_mesh, comp_id, &comp);
      FmsComponentGetCoordinates(comp, &coords);
      if (coords) { main_comp = comp; break; }
   }
   if (!main_comp) { return 1; }
   FmsComponentGetDimension(main_comp, &dim);
   FmsComponentGetNumEntities(main_comp, &n_elem);
   FmsInt n_ents[FMS_NUM_ENTITY_TYPES];
   FmsInt n_main_parts;
   FmsComponentGetNumParts(main_comp, &n_main_parts);

#define RENUMBER_ENTITIES
#ifdef RENUMBER_ENTITIES
   // I noticed that to get domains working right, since they appear to be
   // defined in a local vertex numbering scheme, we have to offset the vertex
   // ids that MFEM makes for shapes to move them to the coordinates in the
   // current domain.

   // However, parts would just be a set of element ids in the current domain
   // and it does not seem appropriate to offset the points in that case.
   // Should domains be treated specially?
   int *verts_per_part = new int[n_main_parts];
#endif

   // Sum the counts for each entity type across parts.
   for (FmsInt et = FMS_VERTEX; et < FMS_NUM_ENTITY_TYPES; et++)
   {
      n_ents[et] = 0;
      for (FmsInt part_id = 0; part_id < n_main_parts; part_id++)
      {
         FmsInt num_ents;
         FmsComponentGetPart(main_comp, part_id, (FmsEntityType)et, NULL, NULL,
                             NULL, NULL, &num_ents);
         n_ents[et] += num_ents;
#ifdef RENUMBER_ENTITIES
         if (et == FMS_VERTEX)
         {
            verts_per_part[part_id] = num_ents;
         }
#endif
      }
   }
   n_vert = n_ents[FMS_VERTEX];

#ifdef RENUMBER_ENTITIES
   int *verts_start = new int[n_main_parts];
   verts_start[0] = 0;
   for (int i = 1; i < n_main_parts; ++i)
   {
      verts_start[i] = verts_start[i-1] + verts_per_part[i-1];
   }
#endif

   // The first related component of dimension dim-1 will be the boundary of the
   // new mfem mesh.
   FmsComponent bdr_comp = NULL;
   FmsInt num_rel_comps;
   const FmsInt *rel_comp_ids;
   FmsComponentGetRelations(main_comp, &rel_comp_ids, &num_rel_comps);
   for (FmsInt i = 0; i < num_rel_comps; i++)
   {
      FmsComponent comp;
      FmsMeshGetComponent(fms_mesh, rel_comp_ids[i], &comp);
      FmsInt comp_dim;
      FmsComponentGetDimension(comp, &comp_dim);
      if (comp_dim == dim-1) { bdr_comp = comp; break; }
   }
   if (bdr_comp)
   {
      FmsComponentGetNumEntities(bdr_comp, &n_bdr_elem);
   }
   else
   {
      n_bdr_elem = 0;
   }

   FmsFieldGet(coords, NULL, &space_dim, NULL, NULL, NULL);
   int err = 0;
   Mesh *mesh = nullptr;
   mesh = new Mesh(dim, n_vert, n_elem, n_bdr_elem, space_dim);

   // Element tags
   FmsInt num_tags;
   FmsMeshGetNumTags(fms_mesh, &num_tags);
   FmsTag elem_tag = NULL, bdr_tag = NULL;
   for (FmsInt tag_id = 0; tag_id < num_tags; tag_id++)
   {
      FmsTag tag;
      FmsMeshGetTag(fms_mesh, tag_id, &tag);
      FmsComponent comp;
      FmsTagGetComponent(tag, &comp);
      if (!elem_tag && comp == main_comp)
      {
#if DEBUG_FMS_MFEM
         const char *tn = NULL;
         FmsTagGetName(tag, &tn);
         mfem::out << "Found element tag " << tn << std::endl;
#endif
         elem_tag = tag;
      }
      else if (!bdr_tag && comp == bdr_comp)
      {
#if DEBUG_FMS_MFEM
         const char *tn = NULL;
         FmsTagGetName(tag, &tn);
         mfem::out << "Found boundary tag " << tn << std::endl;
#endif
         bdr_tag = tag;
      }
   }
   FmsIntType attr_type;
   const void *v_attr, *v_bdr_attr;
   mfem::Array<int> attr, bdr_attr;
   FmsInt num_attr;
   // Element attributes
   if (elem_tag)
   {
      FmsTagGet(elem_tag, &attr_type, &v_attr, &num_attr);
      if (attr_type == FMS_UINT8)
      {
         mfem::Array<uint8_t> at((uint8_t*)v_attr, num_attr);
         attr = at;
      }
      else if (attr_type == FMS_INT32 || attr_type == FMS_UINT32)
      {
         attr.MakeRef((int*)v_attr, num_attr);
      }
      else
      {
         err = 1; // "attribute type not supported!"
         goto func_exit;
      }
   }
   // Boundary attributes
   if (bdr_tag)
   {
      FmsTagGet(bdr_tag, &attr_type, &v_bdr_attr, &num_attr);
      if (attr_type == FMS_UINT8)
      {
         mfem::Array<uint8_t> at((uint8_t*)v_bdr_attr, num_attr);
         bdr_attr = at;
      }
      else if (attr_type == FMS_INT32 || attr_type == FMS_UINT32)
      {
         bdr_attr.MakeRef((int*)v_bdr_attr, num_attr);
      }
      else
      {
         err = 2; // "bdr attribute type not supported!"
         goto func_exit;
      }
   }

   // Add elements
   for (FmsInt part_id = 0; part_id < n_main_parts; part_id++)
   {
      for (int et = FMS_VERTEX; et < FMS_NUM_ENTITY_TYPES; et++)
      {
         if (FmsEntityDim[et] != dim) { continue; }

         FmsDomain domain;
         FmsIntType elem_id_type;
         const void *elem_ids;
         const FmsOrientation *elem_ori;
         FmsInt num_elems;
         FmsComponentGetPart(main_comp, part_id, (FmsEntityType)et, &domain,
                             &elem_id_type, &elem_ids, &elem_ori, &num_elems);

         if (num_elems == 0) { continue; }

         if (elem_ids != NULL &&
             (elem_id_type != FMS_INT32 && elem_id_type != FMS_UINT32))
         {
            err = 3; goto func_exit;
         }
         if (elem_ori != NULL)
         {
            err = 4; goto func_exit;
         }

         const FmsInt nv = FmsEntityNumVerts[et];
         mfem::Array<int> ents_verts(num_elems*nv);
         if (elem_ids == NULL)
         {
            FmsDomainGetEntitiesVerts(domain, (FmsEntityType)et, NULL, FMS_INT32,
                                      0, ents_verts.GetData(), num_elems);
         }
         else
         {
            const int *ei = (const int *)elem_ids;
            for (FmsInt i = 0; i < num_elems; i++)
            {
               FmsDomainGetEntitiesVerts(domain, (FmsEntityType)et, NULL, FMS_INT32,
                                         ei[i], &ents_verts[i*nv], 1);
            }
         }
         const int elem_offset = mesh->GetNE();
         switch ((FmsEntityType)et)
         {
            case FMS_EDGE:
               err = 5;
               goto func_exit;
               break;
            case FMS_TRIANGLE:
#ifdef RENUMBER_ENTITIES
               // The domain vertices/edges were defined in local ordering. We
               // now have a set of triangle vertices defined in terms of local
               // vertex numbers. Renumber them to a global numbering.
               for (FmsInt i = 0; i < num_elems*3; i++)
               {
                  ents_verts[i] += verts_start[part_id];
               }
#endif

               for (FmsInt i = 0; i < num_elems; i++)
               {
                  mesh->AddTriangle(
                     &ents_verts[3*i], elem_tag ? attr[elem_offset+i] : 1);
               }
               break;
            case FMS_QUADRILATERAL:
#ifdef RENUMBER_ENTITIES
               for (FmsInt i = 0; i < num_elems*4; i++)
               {
                  ents_verts[i] += verts_start[part_id];
               }
#endif
               for (FmsInt i = 0; i < num_elems; i++)
               {
                  mesh->AddQuad(&ents_verts[4*i], elem_tag ? attr[elem_offset+i] : 1);
               }
               break;
            case FMS_TETRAHEDRON:
#ifdef RENUMBER_ENTITIES
               for (FmsInt i = 0; i < num_elems*4; i++)
               {
                  ents_verts[i] += verts_start[part_id];
               }
#endif
               for (FmsInt i = 0; i < num_elems; i++)
               {
                  mesh->AddTet(&ents_verts[4*i], elem_tag ? attr[elem_offset+i] : 1);
               }
               break;

            // TODO: What about wedges and pyramids?


            case FMS_HEXAHEDRON:
#ifdef RENUMBER_ENTITIES
               for (FmsInt i = 0; i < num_elems*8; i++)
               {
                  ents_verts[i] += verts_start[part_id];
               }

#endif
               for (FmsInt i = 0; i < num_elems; i++)
               {
                  const int *hex_verts = &ents_verts[8*i];
#if 0
                  const int reorder[8] = {0, 1, 2, 3, 5, 4, 6, 7};
                  const int new_verts[8] = {hex_verts[reorder[0]],
                                            hex_verts[reorder[1]],
                                            hex_verts[reorder[2]],
                                            hex_verts[reorder[3]],
                                            hex_verts[reorder[4]],
                                            hex_verts[reorder[5]],
                                            hex_verts[reorder[6]],
                                            hex_verts[reorder[7]]
                                           };
                  hex_verts = new_verts;
#endif
                  mesh->AddHex(hex_verts, elem_tag ? attr[elem_offset+i] : 1);
               }
               break;
            default:
               break;
         }
      }
   }

   // Add boundary elements
   if (bdr_comp && n_bdr_elem > 0)
   {
      FmsInt n_bdr_parts;
      FmsComponentGetNumParts(bdr_comp, &n_bdr_parts);

      for (FmsInt part_id = 0; part_id < n_bdr_parts; part_id++)
      {
         for (int et = FMS_VERTEX; et < FMS_NUM_ENTITY_TYPES; et++)
         {
            if (FmsEntityDim[et] != dim-1) { continue; }

            FmsDomain domain;
            FmsIntType elem_id_type;
            const void *elem_ids;
            const FmsOrientation *elem_ori;
            FmsInt num_elems;
            FmsComponentGetPart(bdr_comp, part_id, (FmsEntityType)et, &domain,
                                &elem_id_type, &elem_ids, &elem_ori, &num_elems);
            if (num_elems == 0) { continue; }

            if (elem_ids != NULL &&
                (elem_id_type != FMS_INT32 && elem_id_type != FMS_UINT32))
            {
               err = 6; goto func_exit;
            }
            if (elem_ori != NULL)
            {
               err = 7; goto func_exit;
            }

            const FmsInt nv = FmsEntityNumVerts[et];
            mfem::Array<int> ents_verts(num_elems*nv);
            if (elem_ids == NULL)
            {
               FmsDomainGetEntitiesVerts(domain, (FmsEntityType)et, NULL, FMS_INT32,
                                         0, ents_verts.GetData(), num_elems);
            }
            else
            {
               const int *ei = (const int *)elem_ids;
               for (FmsInt i = 0; i < num_elems; i++)
               {
                  FmsDomainGetEntitiesVerts(domain, (FmsEntityType)et, NULL,
                                            FMS_INT32, ei[i], &ents_verts[i*nv], 1);
               }
            }
            const int elem_offset = mesh->GetNBE();
            switch ((FmsEntityType)et)
            {
               case FMS_EDGE:
                  for (FmsInt i = 0; i < num_elems; i++)
                  {
                     mesh->AddBdrSegment(
                        &ents_verts[2*i], bdr_tag ? bdr_attr[elem_offset+i] : 1);
                  }
                  break;
               case FMS_TRIANGLE:
                  for (FmsInt i = 0; i < num_elems; i++)
                  {
                     mesh->AddBdrTriangle(
                        &ents_verts[3*i], bdr_tag ? bdr_attr[elem_offset+i] : 1);
                  }
                  break;
               case FMS_QUADRILATERAL:
                  for (FmsInt i = 0; i < num_elems; i++)
                  {
                     mesh->AddBdrQuad(
                        &ents_verts[4*i], bdr_tag ? bdr_attr[elem_offset+i] : 1);
                  }
                  break;
               default:
                  break;
            }
         }
      }
   }

#ifdef RENUMBER_ENTITIES
   delete [] verts_per_part;
   delete [] verts_start;
#endif

   // Transfer coordinates
   {
      // Set the vertex coordinates to zero
      const double origin[3] = {0.,0.,0.};
      for (FmsInt vi = 0; vi < n_vert; vi++)
      {
         mesh->AddVertex(origin);
      }

      // Finalize the mesh topology
      mesh->FinalizeTopology();

      FmsFieldDescriptor coords_fd = NULL;
      FmsLayoutType coords_layout;
      FmsFieldGet(coords, &coords_fd, NULL, &coords_layout, NULL, NULL);
      if (!coords_fd)
      {
         mfem::err << "Error reading the FMS mesh coords' FieldDescriptor." << std::endl;
         err = 8;
         goto func_exit;
      }
      FmsInt coords_order = 0;
      FmsBasisType coords_btype = FMS_NODAL_GAUSS_CLOSED;
      FmsFieldType coords_ftype = FMS_CONTINUOUS;
      FmsFieldDescriptorGetFixedOrder(coords_fd, &coords_ftype, &coords_btype,
                                      &coords_order);
      // Maybe this is extra but it seems mesh->SetCurvature assumes
      // btype=1. Maybe protects us against corrupt data.
      if (coords_btype != FMS_NODAL_GAUSS_CLOSED)
      {
         mfem::err << "Error reading FMS mesh coords." << std::endl;
         err = 9;
         goto func_exit;
      }

      // Switch to mfem::Mesh with nodes (interpolates the linear coordinates)
      const bool discont = false;
      mesh->SetCurvature(coords_order, (coords_ftype == FMS_DISCONTINUOUS), space_dim,
                         (coords_layout == FMS_BY_VDIM) ?
                         mfem::Ordering::byVDIM : mfem::Ordering::byNODES);

      // Finalize mesh construction
      mesh->Finalize();

      // Set the high-order mesh nodes
      mfem::GridFunction &nodes = *mesh->GetNodes();
      int ce = FmsFieldToGridFunction<double>(fms_mesh, coords, mesh, nodes, false);
   }

func_exit:

   if (err)
   {
      delete mesh;
   }
   else
   {
      *mfem_mesh = mesh;
   }
   return err;
}

bool
BasisTypeToFmsBasisType(int bt, FmsBasisType &btype)
{
   bool retval = false;
   switch (bt)
   {
      case mfem::BasisType::GaussLegendre:
         // mfem::out << "mfem::BasisType::GaussLegendre -> FMS_NODAL_GAUSS_OPEN" << endl;
         btype = FMS_NODAL_GAUSS_OPEN;
         retval = true;
         break;
      case mfem::BasisType::GaussLobatto:
         // mfem::out << "mfem::BasisType::GaussLobato -> FMS_NODAL_GAUSS_CLOSED" << endl;
         btype = FMS_NODAL_GAUSS_CLOSED;
         retval = true;
         break;
      case mfem::BasisType::Positive:
         // mfem::out << "mfem::BasisType::Positive -> FMS_POSITIVE" << endl;
         btype = FMS_POSITIVE;
         retval = true;
         break;
      case mfem::BasisType::OpenUniform:
         // mfem::out << "mfem::BasisType::OpenUniform -> ?" << endl;
         btype = FMS_NODAL_UNIFORM_OPEN;
         retval = true;
         break;
      case mfem::BasisType::ClosedUniform:
         // mfem::out << "mfem::BasisType::ClosedUniform -> ?" << endl;
         btype = FMS_NODAL_UNIFORM_CLOSED;
         retval = true;
         break;
      case mfem::BasisType::OpenHalfUniform:
         // mfem::out << "mfem::BasisType::OpenHalfUniform -> ?" << endl;
         break;
      case mfem::BasisType::Serendipity:
         // mfem::out << "mfem::BasisType::Serendipity -> ?" << endl;
         break;
      case mfem::BasisType::ClosedGL:
         // mfem::out << "mfem::BasisType::ClosedGL -> ?" << endl;
         break;

   }
   /*
       Which MFEM types map to:?
          FMS_NODAL_CHEBYSHEV_OPEN,
          FMS_NODAL_CHEBYSHEV_CLOSED,
   */

   return retval;
}

/**
@note We add the FMS field descriptor and field in here so we can only do it
      after successfully validating the inputs (handling certain grid function
      types, etc.)
*/
int
GridFunctionToFmsField(FmsDataCollection dc, FmsComponent comp,
                       const std::string &fd_name, const std::string &field_name, const Mesh *mesh,
                       const GridFunction *gf,
                       FmsField *outfield)
{
   if (!dc) { return 1; }
   if (!comp) { return 2; }
   if (!mesh) { return 3; }
   if (!gf) { return 4; }
   if (!outfield) { return 5; }

   double *c = gf->GetData();
   int s = gf->Size();

   const mfem::FiniteElementSpace *fespace = gf->FESpace();
   const mfem::FiniteElementCollection *fecoll = fespace->FEColl();

#ifdef DEBUG_MFEM_FMS
   mfem::out << "Adding FMS field for " << field_name << "..." << endl;
#endif

   /* Q: No getter for the basis, do different kinds of FECollection have
         implied basis? There are two subclasses that actually have the getter,
         maybe those aren't implied? */
   FmsInt order = 1;
   int vdim = 1;
   FmsFieldType ftype = FMS_CONTINUOUS;
   FmsBasisType btype = FMS_NODAL_GAUSS_CLOSED;
   switch (fecoll->GetContType())
   {
      case mfem::FiniteElementCollection::CONTINUOUS:
      {
         ftype = FMS_CONTINUOUS;
         order = static_cast<FmsInt>(fespace->GetOrder(0));
         vdim = gf->VectorDim();
         auto fec = dynamic_cast<const mfem::H1_FECollection *>(fecoll);
         if (fec != nullptr)
         {
            if (!BasisTypeToFmsBasisType(fec->GetBasisType(), btype))
            {
               mfem::err << "Error converting MFEM basis type to FMS for FMS_CONTINUOUS." <<
                         std::endl;
               return 6;
            }
         }
         break;
      }
      case mfem::FiniteElementCollection::DISCONTINUOUS:
      {
         ftype = FMS_DISCONTINUOUS;
         order = static_cast<FmsInt>(fespace->GetOrder(0));
         vdim = gf->VectorDim();
         auto fec = dynamic_cast<const mfem::L2_FECollection *>(fecoll);
         if (fec != nullptr)
         {
            if (!BasisTypeToFmsBasisType(fec->GetBasisType(), btype))
            {
               mfem::err << "Error converting MFEM basis type to FMS for FMS_DISCONTINUOUS." <<
                         std::endl;
               return 7;
            }
         }
         break;
      }
      case mfem::FiniteElementCollection::TANGENTIAL:
      {
         mfem::out << "Warning, unsupported ContType (TANGENTIAL) for " << field_name <<
                   ". Using FMS_CONTINUOUS." << std::endl;
         break;
      }
      case mfem::FiniteElementCollection::NORMAL:
      {
         ftype = FMS_HDIV;
         // This RT_FECollection type seems to arise from "RT" fields such as "RT_3D_P1".
         // Checking fe_coll.hpp, this contains verbiage about H_DIV so we assign it the
         // FMS type FMS_HDIV.

         // I've seen RT_3D_P1 return the wrong order so get it from the name.
         int idim, iorder;
         if (sscanf(fecoll->Name(), "RT_%dD_P%d", &idim, &iorder) == 2)
         {
            order = (FmsInt)iorder;
         }
         else
         {
            order = static_cast<FmsInt>(fespace->GetOrder(0));
         }

         // Get the vdim from the fespace since the grid function is returning
         // 3 but we need it to be what was read from the file so we can pass the
         // right vdim to the FMS field descriptor to compute the expected number of dofs.
         vdim = fespace->GetVDim();
         break;
      }
      default:
         mfem::out << "Warning, unsupported ContType for field " << field_name <<
                   ". Using FMS_CONTINUOUS." << std::endl;
         ftype = FMS_CONTINUOUS;
         break;
   }

   // Now that we're not failing, create the fd and field.
   FmsFieldDescriptor fd = NULL;
   FmsField f = NULL;
   FmsDataCollectionAddFieldDescriptor(dc, fd_name.c_str(), &fd);
   FmsDataCollectionAddField(dc, field_name.c_str(), &f);
   *outfield = f;

   /* Q: Why is order defined on a per element basis? */
   FmsFieldDescriptorSetComponent(fd, comp);
   FmsFieldDescriptorSetFixedOrder(fd, ftype, btype, order);

   FmsInt ndofs;
   FmsFieldDescriptorGetNumDofs(fd, &ndofs);

   const char *name = NULL;
   FmsFieldGetName(f, &name);
   FmsLayoutType layout = fespace->GetOrdering() == mfem::Ordering::byVDIM ?
                          FMS_BY_VDIM : FMS_BY_NODES;

#ifdef DEBUG_MFEM_FMS
   switch (ftype)
   {
      case FMS_CONTINUOUS:
         mfem::out << "\tFMS_CONTINUOUS" << std::endl;
         break;
      case FMS_DISCONTINUOUS:
         mfem::out << "\tFMS_DISCONTINUOUS" << std::endl;
         break;
      case FMS_HDIV:
         mfem::out << "\tFMS_HDIV" << std::endl;
         break;
   }
   mfem::out << "\tField is order " << order << " with vdim " << vdim <<
             " and nDoFs " << ndofs << std::endl;
   mfem::out << "\tgf->size() " << gf->Size() << " ndofs * vdim " << ndofs * vdim
             << std::endl;
   mfem::out << "\tlayout " << layout << " (0 = BY_NODES, 1 = BY_VDIM)" <<
             std::endl;
#endif

   if (FmsFieldSet(f, fd, vdim, layout, FMS_DOUBLE, c))
   {
      mfem::err << "Error setting field " << field_name << " in FMS." << std::endl;
      return 8;
   }
   return 0;
}

bool
MfemMetaDataToFmsMetaData(DataCollection *mdc, FmsDataCollection fdc)
{
   if (!mdc) { return false; }
   if (!fdc) { return false; }

   int *cycle = NULL, *timestep = NULL;
   double *time = NULL;
   FmsMetaData top_level = NULL;
   FmsMetaData *cycle_time_timestep = NULL;
   int mdata_err = 0;
   mdata_err = FmsDataCollectionAttachMetaData(fdc, &top_level);
   if (!top_level || mdata_err)
   {
      mfem::err << "Failed to attach metadata to the FmsDataCollection" << std::endl;
      return 40;
   }

   mdata_err = FmsMetaDataSetMetaData(top_level, "MetaData", 3,
                                      &cycle_time_timestep);
   if (!cycle_time_timestep || mdata_err)
   {
      mfem::err << "Failed to acquire FmsMetaData array" << std::endl;
      return false;
   }

   if (!cycle_time_timestep[0])
   {
      mfem::err << "The MetaData pointer for cycle is NULL" << std::endl;
      return false;
   }
   mdata_err = FmsMetaDataSetIntegers(cycle_time_timestep[0], "cycle", FMS_INT32,
                                      1, (void**)&cycle);
   if (!cycle || mdata_err)
   {
      mfem::err << "The data pointer for cycle is NULL" << std::endl;
      return false;
   }
   *cycle = mdc->GetCycle();

   if (!cycle_time_timestep[1])
   {
      mfem::err << "The FmsMetaData pointer for time is NULL" << std::endl;
      return false;
   }
   mdata_err = FmsMetaDataSetScalars(cycle_time_timestep[1], "time", FMS_DOUBLE, 1,
                                     (void**)&time);
   if (!time || mdata_err)
   {
      mfem::err << "The data pointer for time is NULL." << std::endl;
      return false;
   }
   *time = mdc->GetTime();

   if (!cycle_time_timestep[2])
   {
      mfem::err << "The FmsMetData pointer for timestep is NULL" << std::endl;
      return false;
   }
   FmsMetaDataSetIntegers(cycle_time_timestep[2], "timestep", FMS_INT32, 1,
                          (void**)&timestep);
   if (!timestep || mdata_err)
   {
      mfem::err << "The data pointer for timestep is NULL" << std::endl;
      return false;
   }
   *timestep = mdc->GetTimeStep();

   return true;
}

//---------------------------------------------------------------------------
bool
FmsMetaDataGetInteger(FmsMetaData mdata, const std::string &key,
                      std::vector<int> &values)
{
   if (!mdata) { false; }

   bool retval = false;
   FmsMetaDataType type;
   FmsIntType int_type;
   FmsInt i, size;
   FmsMetaData *children = nullptr;
   const void *data = nullptr;
   const char *mdata_name = nullptr;
   if (FmsMetaDataGetType(mdata, &type) == 0)
   {
      switch (type)
      {
         case FMS_INTEGER:
            if (FmsMetaDataGetIntegers(mdata, &mdata_name, &int_type, &size, &data) == 0)
            {
               if (strcasecmp(key.c_str(), mdata_name) == 0)
               {
                  retval = true;

                  // Interpret the integers and store them in the std::vector<int>
                  switch (int_type)
                  {
                     case FMS_INT8:
                        for (i = 0; i < size; i++)
                        {
                           values.push_back(static_cast<int>(reinterpret_cast<const int8_t*>(data)[i]));
                        }
                        break;
                     case FMS_INT16:
                        for (i = 0; i < size; i++)
                        {
                           values.push_back(static_cast<int>(reinterpret_cast<const int16_t*>(data)[i]));
                        }
                        break;
                     case FMS_INT32:
                        for (i = 0; i < size; i++)
                        {
                           values.push_back(static_cast<int>(reinterpret_cast<const int32_t*>(data)[i]));
                        }
                        break;
                     case FMS_INT64:
                        for (i = 0; i < size; i++)
                        {
                           values.push_back(static_cast<int>(reinterpret_cast<const int64_t*>(data)[i]));
                        }
                        break;
                     case FMS_UINT8:
                        for (i = 0; i < size; i++)
                        {
                           values.push_back(static_cast<int>(reinterpret_cast<const uint8_t*>(data)[i]));
                        }
                        break;
                     case FMS_UINT16:
                        for (i = 0; i < size; i++)
                        {
                           values.push_back(static_cast<int>(reinterpret_cast<const uint16_t*>(data)[i]));
                        }
                        break;
                     case FMS_UINT32:
                        for (i = 0; i < size; i++)
                        {
                           values.push_back(static_cast<int>(reinterpret_cast<const uint32_t*>(data)[i]));
                        }
                        break;
                     case FMS_UINT64:
                        for (i = 0; i < size; i++)
                        {
                           values.push_back(static_cast<int>(reinterpret_cast<const uint64_t*>(data)[i]));
                        }
                        break;
                     default:
                        retval = false;
                        break;
                  }
               }
            }
            break;
         case FMS_META_DATA:
            if (FmsMetaDataGetMetaData(mdata, &mdata_name, &size, &children) == 0)
            {
               // Recurse to look for the key we want.
               for (i = 0; i < size && !retval; i++)
               {
                  retval = FmsMetaDataGetInteger(children[i], key, values);
               }
            }
            break;
      }
   }

   return retval;
}

//---------------------------------------------------------------------------
bool
FmsMetaDataGetScalar(FmsMetaData mdata, const std::string &key,
                     std::vector<double> &values)
{
   if (!mdata) { false; }

   bool retval = false;
   FmsMetaDataType type;
   FmsScalarType scal_type;
   FmsInt i, size;
   FmsMetaData *children = nullptr;
   const void *data = nullptr;
   const char *mdata_name = nullptr;
   if (FmsMetaDataGetType(mdata, &type) == 0)
   {
      switch (type)
      {
         case FMS_SCALAR:
            if (FmsMetaDataGetScalars(mdata, &mdata_name, &scal_type, &size, &data) == 0)
            {
               if (strcasecmp(key.c_str(), mdata_name) == 0)
               {
                  retval = true;

                  // Interpret the integers and store them in the std::vector<int>
                  switch (scal_type)
                  {
                     case FMS_FLOAT:
                        for (i = 0; i < size; i++)
                        {
                           values.push_back(static_cast<double>(reinterpret_cast<const float*>(data)[i]));
                        }
                        break;
                     case FMS_DOUBLE:
                        for (i = 0; i < size; i++)
                        {
                           values.push_back(reinterpret_cast<const double*>(data)[i]);
                        }
                        break;
                     default:
                        retval = false;
                        break;
                  }
               }
            }
            break;
         case FMS_META_DATA:
            if (FmsMetaDataGetMetaData(mdata, &mdata_name, &size, &children) == 0)
            {
               // Recurse to look for the key we want.
               for (i = 0; i < size && !retval; i++)
               {
                  retval = FmsMetaDataGetScalar(children[i], key, values);
               }
            }
            break;
      }
   }

   return retval;
}

//---------------------------------------------------------------------------
bool
FmsMetaDataGetString(FmsMetaData mdata, const std::string &key,
                     std::string &value)
{
   if (!mdata) { false; }

   bool retval = false;
   FmsMetaDataType type;
   FmsInt i, size;
   FmsMetaData *children = nullptr;
   const char *mdata_name = nullptr;
   const char *str_value = nullptr;

   if (FmsMetaDataGetType(mdata, &type) == 0)
   {
      switch (type)
      {
         case FMS_STRING:
            if (FmsMetaDataGetString(mdata, &mdata_name, &str_value) == 0)
            {
               if (strcasecmp(key.c_str(), mdata_name) == 0)
               {
                  retval = true;
                  value = str_value;
               }
            }
            break;
         case FMS_META_DATA:
            if (FmsMetaDataGetMetaData(mdata, &mdata_name, &size, &children) == 0)
            {
               // Recurse to look for the key we want.
               for (i = 0; i < size && !retval; i++)
               {
                  retval = FmsMetaDataGetString(children[i], key, value);
               }
            }
            break;
      }
   }

   return retval;
}

/* -------------------------------------------------------------------------- */
/* FMS to MFEM conversion function */
/* -------------------------------------------------------------------------- */

int FmsDataCollectionToDataCollection(FmsDataCollection dc,
                                      DataCollection **mfem_dc)
{
   int retval = 0;
   FmsMesh fms_mesh;
   FmsDataCollectionGetMesh(dc, &fms_mesh);

   // NOTE: The MFEM data collection has a single Mesh. Mesh has a constructor
   //       to take multiple Mesh objects but it appears to glue them together.
   Mesh *mesh = nullptr;
   int err = FmsMeshToMesh(fms_mesh, &mesh);
   if (err == 0)
   {
      std::string collection_name("collection");
      const char *cn = nullptr;
      FmsDataCollectionGetName(dc, &cn);
      if (cn != nullptr)
      {
         collection_name = cn;
      }

      // Make a data collection that contains the mesh.
      DataCollection *mdc = new DataCollection(collection_name, mesh);
      mdc->SetOwnData(true);

      // Now do fields, etc. and add them to mdc.
      FmsField *fields = nullptr;
      FmsInt num_fields = 0;
      if (FmsDataCollectionGetFields(dc, &fields, &num_fields) == 0)
      {
         for (FmsInt i = 0; i < num_fields; ++i)
         {
            const char *name = nullptr;
            FmsFieldGetName(fields[i], &name);

            GridFunction *gf = new GridFunction;

            // Get the data type.
            FmsFieldDescriptor f_fd;
            FmsLayoutType f_layout;
            FmsScalarType f_data_type;
            const void *f_data;
            FmsFieldGet(fields[i], &f_fd, NULL, &f_layout, &f_data_type,
                        &f_data);

            // Interpret the field according to its data type.
            int err = 1;
            switch (f_data_type)
            {
               case FMS_FLOAT:
                  err = FmsFieldToGridFunction<float>(fms_mesh, fields[i], mesh, *gf, true);
                  break;
               case FMS_DOUBLE:
                  err = FmsFieldToGridFunction<double>(fms_mesh, fields[i], mesh, *gf, true);
                  break;
               case FMS_COMPLEX_FLOAT:
               case FMS_COMPLEX_DOUBLE:
                  // Does MFEM support complex?
                  break;
            }

            if (err == 0)
            {
               mdc->RegisterField(name, gf);
            }
            else
            {
               mfem::out << "There was an error converting " << name << " code: " << err <<
                         std::endl;
               delete gf;
            }

            const char *fname = NULL;
            FmsFieldGetName(fields[i], &fname);
         }
      }

      // If we have metadata in FMS, pass what we can through to MFEM.
      FmsMetaData mdata = NULL;
      FmsDataCollectionGetMetaData(dc, &mdata);
      if (mdata)
      {
         std::vector<int> ivalues;
         std::vector<double> dvalues;
         std::string svalue;
         if (FmsMetaDataGetInteger(mdata, "cycle", ivalues))
         {
            if (!ivalues.empty())
            {
               mdc->SetCycle(ivalues[0]);
            }
         }
         if (FmsMetaDataGetScalar(mdata, "time", dvalues))
         {
            if (!dvalues.empty())
            {
               mdc->SetTime(dvalues[0]);
            }
         }
         if (FmsMetaDataGetScalar(mdata, "timestep", dvalues))
         {
            if (!dvalues.empty())
            {
               mdc->SetTimeStep(dvalues[0]);
            }
         }
      }

      *mfem_dc = mdc;
   }
   else
   {
      mfem::out << "FmsDataCollectionToDataCollection: mesh failed to convert. err="
                << err
                << endl;

      retval = 1;
   }

#if DEBUG_FMS_MFEM
   if (*mfem_dc)
   {
      VisItDataCollection visit_dc(std::string("DEBUG_DC"), mesh);
      visit_dc.SetOwnData(false);
      const auto &fields = (*mfem_dc)->GetFieldMap();
      for (const auto &field : fields)
      {
         visit_dc.RegisterField(field.first, field.second);
      }
      visit_dc.Save();
   }
#endif

   return retval;
}



/* -------------------------------------------------------------------------- */
/* MFEM to FMS conversion function */
/* -------------------------------------------------------------------------- */

int
MeshToFmsMesh(const Mesh *mmesh, FmsMesh *fmesh, FmsComponent *volume)
{
   if (!mmesh) { return 1; }
   if (!fmesh) { return 2; }
   if (!volume) { return 3; }

   int err = 0;
   const int num_verticies = mmesh->GetNV();
   const int num_edges = mmesh->GetNEdges();
   const int num_faces = mmesh->GetNFaces();
   const int num_elements = mmesh->GetNE();

#ifdef DEBUG_MFEM_FMS
   mfem::out << "nverts: " << num_verticies << std::endl;
   mfem::out << "nedges: " << num_edges << std::endl;
   mfem::out << "nfaces: " << num_faces << std::endl;
   mfem::out << "nele: " << num_elements << std::endl;
#endif

   FmsMeshConstruct(fmesh);
   FmsMeshSetPartitionId(*fmesh, 0, 1);

   FmsDomain *domains = NULL;
   FmsMeshAddDomains(*fmesh, "Domain", 1, &domains);
   FmsDomainSetNumVertices(domains[0], num_verticies);

   const int edge_reorder[2] = {1, 0};
   const int quad_reorder[4] = {0,1,2,3};
   const int tet_reorder[4] = {3,2,1,0};
   const int hex_reorder[6] = {0,5,1,3,4,2};
   const int *reorder[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
   reorder[FMS_EDGE] = edge_reorder;
   reorder[FMS_QUADRILATERAL] = quad_reorder;
   reorder[FMS_TETRAHEDRON] = tet_reorder;
   reorder[FMS_HEXAHEDRON] = hex_reorder;

   const mfem::Table *edges = mmesh->GetEdgeVertexTable();
   if (!edges)
   {
      mfem::err << "Error, mesh has no edges." << std::endl;
      return 1;
   }
   mfem::Table *faces = mmesh->GetFaceEdgeTable();
   if (!faces && num_faces > 0)
   {
      mfem::err <<
                "Error, mesh contains faces but the \"GetFaceEdgeTable\" returned NULL." <<
                std::endl;
      return 2;
   }

   // Build edges
   std::vector<int> edge_verts(edges->Size() * 2);
   for (int i = 0; i < edges->Size(); i++)
   {
      mfem::Array<int> nids;
      edges->GetRow(i, nids);
      for (int j = 0; j < 2; j++)
      {
         edge_verts[i*2 + j] = nids[j];
      }
   }

   // TODO: Move this code to after the for loop so edges can be added at top level entities
   FmsDomainSetNumEntities(domains[0], FMS_EDGE, FMS_INT32, edge_verts.size() / 2);
   FmsDomainAddEntities(domains[0], FMS_EDGE, reorder, FMS_INT32,
                        edge_verts.data(), edge_verts.size() / 2);
#ifdef DEBUG_MFEM_FMS
   mfem::out << "EDGES: ";
   for (int i = 0; i < edge_verts.size(); i++)
   {
      if (i % 2 == 0) { mfem::out << std::endl << "\t" << i/2 << ": "; }
      mfem::out << edge_verts[i] << " ";
   }
   mfem::out << std::endl;
#endif

   // Build faces
   if (faces)
   {
      // TODO: Support Triangles and Quads, and move this code after the for
      // loop so these can be added as top level entities
      int rowsize = faces->RowSize(0);
      std::vector<int> face_edges(faces->Size() * rowsize);
      for (int i = 0; i < faces->Size(); i++)
      {
         mfem::Array<int> eids;
         faces->GetRow(i, eids);
         for (int j = 0; j < rowsize; j++)
         {
            face_edges[i*rowsize + j] = eids[j];
         }
      }
      FmsEntityType ent_type = (rowsize == 3) ? FMS_TRIANGLE : FMS_QUADRILATERAL;
      FmsDomainSetNumEntities(domains[0], ent_type, FMS_INT32,
                              face_edges.size() / rowsize);
      FmsDomainAddEntities(domains[0], ent_type, NULL, FMS_INT32, face_edges.data(),
                           face_edges.size() / rowsize);
#ifdef DEBUG_MFEM_FMS
      mfem::out << "FACES: ";
      for (int i = 0; i < face_edges.size(); i++)
      {
         if (i % rowsize == 0) { mfem::out << std::endl << "\t" << i/rowsize << ": "; }
         mfem::out << "(" << edge_verts[face_edges[i]*2] << ", " <<
                   edge_verts[face_edges[i]*2+1] << ") ";
      }
      mfem::out << std::endl;
#endif
   }

   // Add top level elements
   std::vector<int> tags;
   std::vector<int> tris;
   std::vector<int> quads;
   std::vector<int> tets;
   std::vector<int> hexes;
   for (int i = 0; i < num_elements; i++)
   {
      auto etype = mmesh->GetElementType(i);
      tags.push_back(mmesh->GetAttribute(i));
      switch (etype)
      {
         case mfem::Element::POINT:
         {
            // TODO: ?
            break;
         }
         case mfem::Element::SEGMENT:
         {
            // TODO: ?
            break;
         }
         case mfem::Element::TRIANGLE:
         {
            mfem::Array<int> eids, oris;
            mmesh->GetElementEdges(i, eids, oris);
            for (int e = 0; e < 3; e++)
            {
               tris.push_back(eids[e]);
            }
            break;
         }
         case mfem::Element::QUADRILATERAL:
         {
            mfem::Array<int> eids, oris;
            mmesh->GetElementEdges(i, eids, oris);
            for (int e = 0; e < 4; e++)
            {
               quads.push_back(eids[e]);
            }
            break;
         }
         case mfem::Element::TETRAHEDRON:
         {
            mfem::Array<int> fids, oris;
            mmesh->GetElementFaces(i, fids, oris);
            for (int f = 0; f < 4; f++)
            {
               tets.push_back(fids[f]);
            }
            break;
         }
         case mfem::Element::HEXAHEDRON:
         {
            mfem::Array<int> fids, oris;
            mmesh->GetElementFaces(i, fids, oris);
            for (int f = 0; f < 6; f++)
            {
               hexes.push_back(fids[f]);
            }
            break;
         }
         default:
            mfem::err << "Error, element not implemented." << std::endl;
            return 3;
      }
   }

   if (tris.size())
   {
      FmsDomainSetNumEntities(domains[0], FMS_TRIANGLE, FMS_INT32, tris.size() / 3);
      FmsDomainAddEntities(domains[0], FMS_TRIANGLE, reorder, FMS_INT32, tris.data(),
                           tris.size() / 3);
#ifdef DEBUG_MFEM_FMS
      mfem::out << "TRIS: ";
      for (int i = 0; i < tris.size(); i++)
      {
         if (i % 3 == 0) { mfem::out << std::endl << "\t" << i/3 << ": "; }
         mfem::out << tris[i] << " ";
      }
      mfem::out << std::endl;
#endif
   }

   if (quads.size())
   {
      // TODO: Not quite right either, if there are hexes and quads then this
      // will overwrite the faces that made up the hexes
      FmsDomainSetNumEntities(domains[0], FMS_QUADRILATERAL, FMS_INT32,
                              quads.size() / 4);
      FmsDomainAddEntities(domains[0], FMS_QUADRILATERAL, reorder, FMS_INT32,
                           quads.data(), quads.size() / 4);
#ifdef DEBUG_MFEM_FMS
      mfem::out << "QUADS: ";
      for (int i = 0; i < quads.size(); i++)
      {
         if (i % 4 == 0) { mfem::out << std::endl << "\t" << i/4 << ": "; }
         mfem::out << quads[i] << " ";
      }
      mfem::out << std::endl;
#endif
   }

   if (tets.size())
   {
      FmsDomainSetNumEntities(domains[0], FMS_TETRAHEDRON, FMS_INT32,
                              tets.size() / 4);
      FmsDomainAddEntities(domains[0], FMS_TETRAHEDRON, reorder, FMS_INT32,
                           tets.data(), tets.size() / 4);
#ifdef DEBUG_MFEM_FMS
      mfem::out << "TETS: ";
      for (int i = 0; i < tets.size(); i++)
      {
         if (i % 4 == 0) { mfem::out << std::endl << "\t" << i/4 << ": "; }
         mfem::out << tets[i] << " ";
      }
      mfem::out << std::endl;
#endif
   }

   if (hexes.size())
   {
      FmsDomainSetNumEntities(domains[0], FMS_HEXAHEDRON, FMS_INT32,
                              hexes.size() / 6);
      FmsDomainAddEntities(domains[0], FMS_HEXAHEDRON, reorder, FMS_INT32,
                           hexes.data(), hexes.size() / 6);
#ifdef DEBUG_MFEM_FMS
      mfem::out << "HEXES: ";
      for (int i = 0; i < hexes.size(); i++)
      {
         if (i % 6 == 0) { mfem::out << std::endl << "\t" << i/6 << ": "; }
         mfem::out << hexes[i] << " ";
      }
      mfem::out << std::endl;
#endif
   }

   err = FmsMeshFinalize(*fmesh);
   if (err)
   {
      mfem::err << "FmsMeshFinalize returned error code " << err << std::endl;
      return 4;
   }

   err = FmsMeshValidate(*fmesh);
   if (err)
   {
      mfem::err << "FmsMeshValidate returned error code " << err << std::endl;
      return 5;
   }

   FmsComponent v = NULL;
   FmsMeshAddComponent(*fmesh, "volume", &v);
   FmsComponentAddDomain(v, domains[0]);

   FmsTag tag;
   FmsMeshAddTag(*fmesh, "element_attribute", &tag);
   FmsTagSetComponent(tag, v);
   FmsTagSet(tag, FMS_INT32, FMS_INT32, tags.data(), tags.size());

   // Add boundary component
   std::vector<int> bdr_eles[FMS_NUM_ENTITY_TYPES];
   std::vector<int> bdr_attributes;
   const int NBE = mmesh->GetNBE();
   for (int i = 0; i < NBE; i++)
   {
      const Element::Type betype = mmesh->GetBdrElementType(i);
      bdr_attributes.push_back(mmesh->GetBdrAttribute(i));
      switch (betype)
      {
         case Element::POINT:
            bdr_eles[FMS_VERTEX].push_back(mmesh->GetBdrElementEdgeIndex(i));
            break;
         case Element::SEGMENT:
            bdr_eles[FMS_EDGE].push_back(mmesh->GetBdrElementEdgeIndex(i));
            break;
         case Element::TRIANGLE:
            bdr_eles[FMS_TRIANGLE].push_back(mmesh->GetBdrElementEdgeIndex(i));
            break;
         case Element::QUADRILATERAL:
            bdr_eles[FMS_QUADRILATERAL].push_back(mmesh->GetBdrElementEdgeIndex(i));
            break;
         case Element::TETRAHEDRON:
            bdr_eles[FMS_TETRAHEDRON].push_back(mmesh->GetBdrElementEdgeIndex(i));
            break;
         case Element::HEXAHEDRON:
            bdr_eles[FMS_HEXAHEDRON].push_back(mmesh->GetBdrElementEdgeIndex(i));
            break;
         default:
            MFEM_WARNING("Unsupported boundary element " << betype << " at boundary index "
                         << i);
            break;
      }
   }

   if (NBE)
   {
      FmsComponent boundary = NULL;
      FmsMeshAddComponent(*fmesh, "boundary", &boundary);
      FmsInt part_id;
      FmsComponentAddPart(boundary, domains[0], &part_id);
      for (int i = FMS_NUM_ENTITY_TYPES - 1; i > 0; i--)
      {
         if (bdr_eles[i].size())
         {
            FmsComponentAddPartEntities(boundary, part_id, (FmsEntityType)i,
                                        FMS_INT32, FMS_INT32, FMS_INT32, NULL,
                                        bdr_eles[i].data(),
                                        NULL, bdr_eles[i].size());
            break;
         }
      }
      FmsComponentAddRelation(v, 1);
      FmsTag boundary_tag = NULL;
      FmsMeshAddTag(*fmesh, "boundary_attribute", &boundary_tag);
      FmsTagSetComponent(boundary_tag, boundary);
      FmsTagSet(boundary_tag, FMS_INT32, FMS_INT32, bdr_attributes.data(),
                bdr_attributes.size());
   }
   *volume = v;
   return 0;
}

int
DataCollectionToFmsDataCollection(DataCollection *mfem_dc,
                                  FmsDataCollection *dc)
{
   // TODO: Write me
   int err = 0;
   const Mesh *mmesh = mfem_dc->GetMesh();

   FmsMesh fmesh = NULL;
   FmsComponent volume = NULL;
   err = MeshToFmsMesh(mmesh, &fmesh, &volume);
   if (!fmesh || !volume || err)
   {
      mfem::err << "Error converting mesh topology from MFEM to FMS" << std::endl;
      if (fmesh)
      {
         FmsMeshDestroy(&fmesh);
      }
      return 1;
   }

   err = FmsDataCollectionCreate(fmesh, mfem_dc->GetCollectionName().c_str(), dc);
   if (!*dc || err)
   {
      mfem::err << "There was an error creating the FMS data collection." <<
                std::endl;
      FmsMeshDestroy(&fmesh);
      if (*dc)
      {
         FmsDataCollectionDestroy(dc);
      }
      return 3;
   }

   // Add the coordinates field to the data collection
   const mfem::GridFunction *mcoords = mmesh->GetNodes();
   if (mcoords)
   {
      FmsField fcoords = NULL;
      err  = GridFunctionToFmsField(*dc, volume, "CoordsDescriptor", "Coords", mmesh,
                                    mcoords, &fcoords);
      err |= FmsComponentSetCoordinates(volume, fcoords);
   }
   else
   {
      // Sometimes the nodes are stored as just a vector of vertex coordinates
      mfem::Vector mverts;
      mmesh->GetVertices(mverts);
      FmsFieldDescriptor fdcoords = NULL;
      FmsField fcoords = NULL;
      err  = FmsDataCollectionAddFieldDescriptor(*dc, "CoordsDescriptor", &fdcoords);
      err |= FmsFieldDescriptorSetComponent(fdcoords, volume);
      err |= FmsFieldDescriptorSetFixedOrder(fdcoords, FMS_CONTINUOUS,
                                             FMS_NODAL_GAUSS_CLOSED, 1);
      err |= FmsDataCollectionAddField(*dc, "Coords", &fcoords);
      err |= FmsFieldSet(fcoords, fdcoords, mmesh->SpaceDimension(), FMS_BY_NODES,
                         FMS_DOUBLE, mverts);
      err |= FmsComponentSetCoordinates(volume, fcoords);
   }

   if (err)
   {
      mfem::err << "There was an error setting the mesh coordinates." << std::endl;
      FmsMeshDestroy(&fmesh);
      FmsDataCollectionDestroy(dc);
      return 4;
   }

   const auto &fields = mfem_dc->GetFieldMap();
   for (const auto &pair : fields)
   {
      std::string fd_name(pair.first + "Descriptor");
      FmsField field;
      err = GridFunctionToFmsField(*dc, volume, fd_name, pair.first.c_str(), mmesh,
                                   pair.second, &field);
      if (err)
      {
         mfem::err << "WARNING: There was an error adding the " << pair.first <<
                   " field. Continuing..." << std::endl;
      }
   }

   // /* TODO:
   // const auto &qfields = mfem_dc->GetQFieldMap();
   // for(const auto &pair : qfields) {
   //   FmsFieldDescriptor fd = NULL;
   //   FmsField f = NULL;
   //   std::string fd_name(pair.first + "Collection");
   //   FmsDataCollectionAddFieldDescriptor(*dc, fd_name.c_str(), &fd);
   //   FmsDataCollectionAddField(*dc, pair.first.c_str(), &f);
   //   GridFunctionToFmsField(*dc, fd, f, volume, pair.second); // TODO: Volume isn't always going to be correct
   // } */

   MfemMetaDataToFmsMetaData(mfem_dc, *dc);

   return 0;
}

} // namespace mfem
#endif
