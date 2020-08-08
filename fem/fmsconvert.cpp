// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "fmsconvert.hpp"
#include <unordered_map>
#include <climits>
using std::cout;
using std::endl;
namespace mfem
{

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
    if (f_fd_type != FMS_FIXED_ORDER) {
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
FmsFieldToGridFunction(FmsMesh fms_mesh, FmsField f, Mesh *mesh, GridFunction &func, bool setFE)
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
  for (FmsInt comp_id = 0; comp_id < num_comp; comp_id++) {
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
  for (FmsInt et = FMS_VERTEX; et < FMS_NUM_ENTITY_TYPES; et++) {
    n_ents[et] = 0;
    for (FmsInt part_id = 0; part_id < n_main_parts; part_id++) {
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

//    FmsFieldGet(coords, NULL, &space_dim, NULL, NULL, NULL);


    FmsInt f_num_dofs;
    FmsFieldDescriptorGetNumDofs(f_fd, &f_num_dofs);

    // Access FMS data through this typed pointer.
    auto src_data = reinterpret_cast<const DataType *>(f_data);

    FmsFieldDescriptorType f_fd_type;
    FmsFieldDescriptorGetType(f_fd, &f_fd_type);
    if (f_fd_type != FMS_FIXED_ORDER) {
      return 9;
    }
    FmsFieldType f_field_type;
    FmsBasisType f_basis_type;
    FmsInt f_order;
    FmsFieldDescriptorGetFixedOrder(f_fd, &f_field_type,
                                    &f_basis_type, &f_order);
    if (f_field_type != FMS_CONTINUOUS) {
      return 10;
    }
    if (f_basis_type != FMS_NODAL_GAUSS_CLOSED) {
      return 11;
    }

//------------------------------------------------------------------
    if(setFE)
    {
// We could assemble a name based on fe_coll.hpp rules and pass to FiniteElementCollection::New()
        auto fec = new H1_FECollection(f_order, dim);
        int ordering = (f_layout == FMS_BY_VDIM) ? Ordering::byVDIM : Ordering::byNODES;        
        auto fes = new FiniteElementSpace(mesh, fec, space_dim, ordering);
        func.SetSpace(fes);
cout << "\tFESpace=" << (void*)func.FESpace() << endl;

    }
//------------------------------------------------------------------
    const FmsInt nstride = (f_layout == FMS_BY_VDIM) ? space_dim : 1;
    const FmsInt vstride = (f_layout == FMS_BY_VDIM) ? 1 : f_num_dofs;


    // Data reordering to store the data into func.
cout << "func.Size()=" << func.Size() << endl;
cout << "f_num_dofs=" << f_num_dofs << endl;
cout << "space_dim=" << space_dim << endl;

    if ((FmsInt)(func.Size()) != f_num_dofs*space_dim) {
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
    if (dim >= 2 && edge_dofs > 0) {
      mfem::Array<int> ev;
      for (int i = 0; i < mesh->GetNEdges(); i++) {
        mesh->GetEdgeVertices(i, ev);
        int id = mfem_edge.GetId(ev[0], ev[1]);
        if (id != i) { return 13; }
      }
    }
    if (dim >= 3 &&
        ((n_ents[FMS_TRIANGLE] > 0 && tri_dofs > 0) ||
         (n_ents[FMS_QUADRILATERAL] > 0 && quad_dofs > 0))) {
      mfem::Array<int> fv;
      for (int i = 0; i < mesh->GetNFaces(); i++) {
        mesh->GetFaceVertices(i, fv);
        if (fv.Size() == 3) { fv.Append(INT_MAX); }
        // HashTable uses the smallest 3 of the 4 indices to hash Hashed4
        int id = mfem_face.GetId(fv[0], fv[1], fv[2], fv[3]);
        if (id != i) { return 14; }
      }
    }

    // Loop over all parts of the main component
    for (FmsInt part_id = 0; part_id < n_main_parts; part_id++) {
      // Loop over all entity types in the part
      for (FmsInt et = FMS_VERTEX; et < FMS_NUM_ENTITY_TYPES; et++) {
        FmsDomain domain;
        FmsIntType ent_id_type;
        const void *ents;
        const FmsOrientation *ents_ori;
        FmsInt num_ents;
        FmsComponentGetPart(main_comp, part_id, (FmsEntityType)et, &domain,
                            &ent_id_type, &ents, &ents_ori, &num_ents);
        if (num_ents == 0) { continue; }
        if (ent_dofs[et] == 0) {
          if (et == FMS_VERTEX) { mfem_last_vert_cnt = mfem_ent_cnt[et]; }
          mfem_ent_cnt[FmsEntityDim[et]] += num_ents;
          continue;
        }

        if (ents != NULL &&
            (ent_id_type != FMS_INT32 && ent_id_type != FMS_UINT32)) {
          return 15;
        }
        if (ents_ori != NULL) {
          return 16;
        }

        if (et == FMS_VERTEX) {
          const int mfem_dof_offset = mfem_ent_cnt[0]*vert_dofs;
          for (FmsInt i = 0; i < num_ents*vert_dofs; i++) {
            for (int j = 0; j < vdim; j++) {
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
        if (FmsEntityDim[et] == dim) {
          for (FmsInt e = 0; e < num_ents; e++) {
            fes->GetElementInteriorDofs(mfem_ent_cnt[dim]+e, dofs);
            for (int i = 0; i < ent_dofs[et]; i++, fms_dof_offset++) {
              for (int j = 0; j < vdim; j++) {
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
        if (ents == NULL) {
          FmsDomainGetEntitiesVerts(domain, (FmsEntityType)et, NULL, FMS_INT32,
                                    0, ents_verts.GetData(), num_ents);
        } else {
          for (FmsInt i = 0; i < num_ents; i++) {
            FmsDomainGetEntitiesVerts(domain, (FmsEntityType)et, NULL,
                                      FMS_INT32, ei[i], &ents_verts[i*nv], 1);
          }
        }
        for (int i = 0; i < ents_verts.Size(); i++) {
          ents_verts[i] += mfem_last_vert_cnt;
        }
        const int *perm;
        switch ((FmsEntityType)et) {
        case FMS_EDGE: {
          for (FmsInt part_ent_id = 0; part_ent_id < num_ents; part_ent_id++) {
            const int *ev = &ents_verts[2*part_ent_id];
            int mfem_edge_id = mfem_edge.FindId(ev[0], ev[1]);
            if (mfem_edge_id < 0) {
              return 17;
            }
            mesh->GetEdgeVertices(mfem_edge_id, m_ev);
            int ori = (ev[0] == m_ev[0]) ? 0 : 1;
            perm = fec->DofOrderForOrientation(mfem::Geometry::SEGMENT, ori);
            fes->GetEdgeInteriorDofs(mfem_edge_id, dofs);
            for (int i = 0; i < edge_dofs; i++) {
              for (int j = 0; j < vdim; j++) {
                func(fes->DofToVDof(dofs[i],j)) =
                  static_cast<double>(src_data[(fms_dof_offset+perm[i])*nstride+j*vstride]);
              }
            }
            fms_dof_offset += edge_dofs;
          }
          break;
        }
        case FMS_TRIANGLE: {
          for (FmsInt part_ent_id = 0; part_ent_id < num_ents; part_ent_id++) {
            const int *tv = &ents_verts[3*part_ent_id];
            int mfem_face_id = mfem_face.FindId(tv[0], tv[1], tv[2], INT_MAX);
            if (mfem_face_id < 0) {
              return 18;
            }
            mesh->GetFaceVertices(mfem_face_id, m_ev);
            int ori = 0;
            while (tv[ori] != m_ev[0]) { ori++; }
            ori = (tv[(ori+1)%3] == m_ev[1]) ? 2*ori : 2*ori+1;
            perm = fec->DofOrderForOrientation(mfem::Geometry::TRIANGLE, ori);
            fes->GetFaceInteriorDofs(mfem_face_id, dofs);
            for (int i = 0; i < tri_dofs; i++) {
              for (int j = 0; j < vdim; j++) {
                func(fes->DofToVDof(dofs[i],j)) =
                  static_cast<double>(src_data[(fms_dof_offset+perm[i])*nstride+j*vstride]);
              }
            }
            fms_dof_offset += tri_dofs;
          }
          break;
        }
        case FMS_QUADRILATERAL: {
          for (FmsInt part_ent_id = 0; part_ent_id < num_ents; part_ent_id++) {
            const int *qv = &ents_verts[4*part_ent_id];
            int mfem_face_id = mfem_face.FindId(qv[0], qv[1], qv[2], qv[3]);
            if (mfem_face_id < 0) { return 19; }
            mesh->GetFaceVertices(mfem_face_id, m_ev);
            int ori = 0;
            while (qv[ori] != m_ev[0]) { ori++; }
            ori = (qv[(ori+1)%4] == m_ev[1]) ? 2*ori : 2*ori+1;
            perm = fec->DofOrderForOrientation(mfem::Geometry::SQUARE, ori);
            fes->GetFaceInteriorDofs(mfem_face_id, dofs);
            for (int i = 0; i < quad_dofs; i++) {
              for (int j = 0; j < vdim; j++) {
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
cout << "FmsMeshToMesh: num_comp=" << num_comp << endl;
  FmsComponent main_comp = NULL;
  FmsField coords = NULL;
  for (FmsInt comp_id = 0; comp_id < num_comp; comp_id++) {
    FmsComponent comp;
    FmsMeshGetComponent(fms_mesh, comp_id, &comp);
    FmsComponentGetCoordinates(comp, &coords);
    if (coords) { cout << "comp " << comp_id << " has coordinates." << endl; main_comp = comp; break; }
  }
  if (!main_comp) { return 1; }
  FmsComponentGetDimension(main_comp, &dim);
  FmsComponentGetNumEntities(main_comp, &n_elem);
  FmsInt n_ents[FMS_NUM_ENTITY_TYPES];
  FmsInt n_main_parts;
  FmsComponentGetNumParts(main_comp, &n_main_parts);
cout << "FmsMeshToMesh: n_main_parts=" << n_main_parts << endl;

#define RENUMBER_ENTITIES
#ifdef RENUMBER_ENTITIES
// I noticed that to get domains working right, since they appear to be 
// defined in a local vertex numbering scheme, we have to offset the 
// vertex ids that MFEM makes for shapes to move them to the coordinates
// in the current domain.

// However, parts would just be a set of element ids in the current domain
// and it does not seem appropriate to offset the points in that case.
// Should domains be treated specially?
  int *verts_per_part = new int[n_main_parts];
#endif

  // Sum the counts for each entity type across parts.
  for (FmsInt et = FMS_VERTEX; et < FMS_NUM_ENTITY_TYPES; et++) {
    n_ents[et] = 0;
cout << "et=" << et << endl;
    for (FmsInt part_id = 0; part_id < n_main_parts; part_id++) {
      FmsInt num_ents;
      FmsComponentGetPart(main_comp, part_id, (FmsEntityType)et, NULL, NULL,
                          NULL, NULL, &num_ents);
cout << "\t" << part_id << ": num_ents=" << num_ents << endl;
      n_ents[et] += num_ents;
#ifdef RENUMBER_ENTITIES
      if(et == FMS_VERTEX)
          verts_per_part[part_id] = num_ents;
#endif
    }
  }
  n_vert = n_ents[FMS_VERTEX];

#ifdef RENUMBER_ENTITIES
  int *verts_start = new int[n_main_parts];
  verts_start[0] = 0;
  for(int i = 1; i < n_main_parts;++i)
     verts_start[i] = verts_start[i-1] + verts_per_part[i-1];

  cout << "verts_per_part = {";
  for(int i = 0; i < n_main_parts;++i)
      cout << verts_per_part[i] << ", ";
  cout << "}" << endl;
  cout << "verts_start = {";
  for(int i = 0; i < n_main_parts;++i)
      cout << verts_start[i] << ", ";
  cout << "}" << endl;
#endif

  // The first related component of dimension dim-1 will be the boundary of the
  // new mfem mesh.
  FmsComponent bdr_comp = NULL;
  FmsInt num_rel_comps;
  const FmsInt *rel_comp_ids;
  FmsComponentGetRelations(main_comp, &rel_comp_ids, &num_rel_comps);
  for (FmsInt i = 0; i < num_rel_comps; i++) {
    FmsComponent comp;
    FmsMeshGetComponent(fms_mesh, rel_comp_ids[i], &comp);
    FmsInt comp_dim;
    FmsComponentGetDimension(comp, &comp_dim);
    if (comp_dim == dim-1) { bdr_comp = comp; break; }
  }
  if (bdr_comp) {
    FmsComponentGetNumEntities(bdr_comp, &n_bdr_elem);
  } else {
    n_bdr_elem = 0;
  }

  FmsFieldGet(coords, NULL, &space_dim, NULL, NULL, NULL);
#if 1
cout << "dim=" << dim << endl;
cout << "n_vert=" << n_vert << endl;
cout << "n_elem=" << n_elem << endl;
cout << "n_bdr_elem=" << n_bdr_elem << endl;
cout << "space_dim=" << space_dim << endl;
for (FmsInt et = FMS_VERTEX; et < FMS_NUM_ENTITY_TYPES; et++)
    cout << "n_ents[" << et << "]=" << n_ents[et] << endl;
#endif
  int err = 0;
  Mesh *mesh = nullptr;
  mesh = new Mesh(dim, n_vert, n_elem, n_bdr_elem, space_dim);

  // Element tags
  FmsInt num_tags;
  FmsMeshGetNumTags(fms_mesh, &num_tags);
  FmsTag elem_tag = NULL, bdr_tag = NULL;
  for (FmsInt tag_id = 0; tag_id < num_tags; tag_id++) {
    FmsTag tag;
    FmsMeshGetTag(fms_mesh, tag_id, &tag);
    FmsComponent comp;
    FmsTagGetComponent(tag, &comp);
    if (!elem_tag && comp == main_comp) {
      elem_tag = tag;
    } else if (!bdr_tag && comp == bdr_comp) {
      bdr_tag = tag;
    }
  }
  FmsIntType attr_type;
  const void *v_attr, *v_bdr_attr;
  mfem::Array<int> attr, bdr_attr;
  FmsInt num_attr;
  // Element attributes
  if (elem_tag) {
    FmsTagGet(elem_tag, &attr_type, &v_attr, &num_attr);
    if (attr_type == FMS_UINT8) {
      mfem::Array<uint8_t> at((uint8_t*)v_attr, num_attr);
      attr = at;
    } else if (attr_type == FMS_INT32 || attr_type == FMS_UINT32) {
      attr.MakeRef((int*)v_attr, num_attr);
    } else {
      err = 1; // "attribute type not supported!"
      goto func_exit;
    }
  }
  // Boundary attributes
  if (bdr_tag) {
    FmsTagGet(bdr_tag, &attr_type, &v_bdr_attr, &num_attr);
    if (attr_type == FMS_UINT8) {
      mfem::Array<uint8_t> at((uint8_t*)v_bdr_attr, num_attr);
      bdr_attr = at;
    } else if (attr_type == FMS_INT32 || attr_type == FMS_UINT32) {
      bdr_attr.MakeRef((int*)v_bdr_attr, num_attr);
    } else {
      err = 2; // "bdr attribute type not supported!"
      goto func_exit;
    }
  }

  // Add elements
cout << "n_main_parts=" << n_main_parts << endl;
  for (FmsInt part_id = 0; part_id < n_main_parts; part_id++) {
cout << "part " << part_id << ":" << endl;
    for (int et = FMS_VERTEX; et < FMS_NUM_ENTITY_TYPES; et++) {
      if (FmsEntityDim[et] != dim) { continue; }

      FmsDomain domain;
      FmsIntType elem_id_type;
      const void *elem_ids;
      const FmsOrientation *elem_ori;
      FmsInt num_elems;
      FmsComponentGetPart(main_comp, part_id, (FmsEntityType)et, &domain,
                          &elem_id_type, &elem_ids, &elem_ori, &num_elems);
cout << "Getting component part " << part_id << "'s entities et=" << et << ". num_elems=" << num_elems << endl;
      if (num_elems == 0) { continue; }

      if (elem_ids != NULL &&
          (elem_id_type != FMS_INT32 && elem_id_type != FMS_UINT32)) {
        err = 3; goto func_exit;
      }
      if (elem_ori != NULL) {
        err = 4; goto func_exit;
      }

      const FmsInt nv = FmsEntityNumVerts[et];
      mfem::Array<int> ents_verts(num_elems*nv);
      if (elem_ids == NULL) {
        FmsDomainGetEntitiesVerts(domain, (FmsEntityType)et, NULL, FMS_INT32,
                                  0, ents_verts.GetData(), num_elems);
      } else {
        const int *ei = (const int *)elem_ids;
        for (FmsInt i = 0; i < num_elems; i++) {
          FmsDomainGetEntitiesVerts(domain, (FmsEntityType)et, NULL, FMS_INT32,
                                    ei[i], &ents_verts[i*nv], 1);
        }
      }
      const int elem_offset = mesh->GetNE();
      switch ((FmsEntityType)et) {
      case FMS_EDGE:
        err = 5;
        goto func_exit;
        break;
      case FMS_TRIANGLE:
#ifdef RENUMBER_ENTITIES
        // The domain vertices/edges were defined in local ordering. We now
        // have a set of triangle vertices defined in terms of local vertex
        // numbers. Renumber them to a global numbering.
        for (FmsInt i = 0; i < num_elems*3; i++)
            ents_verts[i] += verts_start[part_id];
#endif

        for (FmsInt i = 0; i < num_elems; i++) {
#if 1
          cout << "\tAddTriangle: {"
               << ents_verts[3*i+0] << ", "
               << ents_verts[3*i+1] << ", "
               << ents_verts[3*i+2] << "}, tag=" << (elem_tag ? attr[elem_offset+i] : 1) << endl;
#endif
          mesh->AddTriangle(
            &ents_verts[3*i], elem_tag ? attr[elem_offset+i] : 1);
        }
        break;
      case FMS_QUADRILATERAL:
#ifdef RENUMBER_ENTITIES
        for (FmsInt i = 0; i < num_elems*4; i++)
            ents_verts[i] += verts_start[part_id];
#endif
        for (FmsInt i = 0; i < num_elems; i++) {
          mesh->AddQuad(&ents_verts[4*i], elem_tag ? attr[elem_offset+i] : 1);
        }
        break;
      case FMS_TETRAHEDRON:
#ifdef RENUMBER_ENTITIES
        for (FmsInt i = 0; i < num_elems*4; i++)
            ents_verts[i] += verts_start[part_id];
#endif
        for (FmsInt i = 0; i < num_elems; i++) {
          mesh->AddTet(&ents_verts[4*i], elem_tag ? attr[elem_offset+i] : 1);
        }
        break;

      // TODO: What about wedges and pyramids?


      case FMS_HEXAHEDRON:
#ifdef RENUMBER_ENTITIES
        for (FmsInt i = 0; i < num_elems*8; i++)
            ents_verts[i] += verts_start[part_id];
#endif
        for (FmsInt i = 0; i < num_elems; i++) {
          mesh->AddHex(&ents_verts[8*i], elem_tag ? attr[elem_offset+i] : 1);
        }
        break;
      default:
        break;
      }
    }
  }

  // Add boundary elements
  if (bdr_comp && n_bdr_elem > 0) {
    FmsInt n_bdr_parts;
    FmsComponentGetNumParts(bdr_comp, &n_bdr_parts);

    for (FmsInt part_id = 0; part_id < n_bdr_parts; part_id++) {
      for (int et = FMS_VERTEX; et < FMS_NUM_ENTITY_TYPES; et++) {
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
            (elem_id_type != FMS_INT32 && elem_id_type != FMS_UINT32)) {
          err = 6; goto func_exit;
        }
        if (elem_ori != NULL) {
          err = 7; goto func_exit;
        }

        const FmsInt nv = FmsEntityNumVerts[et];
        mfem::Array<int> ents_verts(num_elems*nv);
        if (elem_ids == NULL) {
          FmsDomainGetEntitiesVerts(domain, (FmsEntityType)et, NULL, FMS_INT32,
                                    0, ents_verts.GetData(), num_elems);
        } else {
          const int *ei = (const int *)elem_ids;
          for (FmsInt i = 0; i < num_elems; i++) {
            FmsDomainGetEntitiesVerts(domain, (FmsEntityType)et, NULL,
                                      FMS_INT32, ei[i], &ents_verts[i*nv], 1);
          }
        }
        const int elem_offset = mesh->GetNBE();
        switch ((FmsEntityType)et) {
        case FMS_EDGE:
          for (FmsInt i = 0; i < num_elems; i++) {
            mesh->AddBdrSegment(
              &ents_verts[2*i], bdr_tag ? bdr_attr[elem_offset+i] : 1);
          }
          break;
        case FMS_TRIANGLE:
          for (FmsInt i = 0; i < num_elems; i++) {
            mesh->AddBdrTriangle(
              &ents_verts[3*i], bdr_tag ? bdr_attr[elem_offset+i] : 1);
          }
          break;
        case FMS_QUADRILATERAL:
          for (FmsInt i = 0; i < num_elems; i++) {
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

  // Finalize the mesh topology
  // FIXME: mfem::Mesh::FinalizeCheck() assumes all vertices are added
  // mesh.FinalizeTopology();

  // Transfer coordinates
  {
cout << "n_vert=" << n_vert << endl;
    // Set the vertex coordinates to zero
    const double origin[3] = {0.,0.,0.};
    for (FmsInt vi = 0; vi < n_vert; vi++) {
      mesh->AddVertex(origin);
    }

    // Finalize the mesh topology
    mesh->FinalizeTopology();

    FmsInt coords_order = 0;
    FmsLayoutType coords_layout;
    FmsFieldGetOrderAndLayout(coords, &coords_order, &coords_layout);
cout << "coords_order=" << coords_order << endl;
cout << "coords_layout=" << coords_layout << endl;

    // Switch to mfem::Mesh with nodes (interpolates the linear coordinates)
    const bool discont = false;
    mesh->SetCurvature(coords_order, discont, space_dim,
                       (coords_layout == FMS_BY_VDIM) ?
                       mfem::Ordering::byVDIM : mfem::Ordering::byNODES);

    // Finalize mesh construction
    mesh->Finalize();

    // Set the high-order mesh nodes
    mfem::GridFunction &nodes = *mesh->GetNodes();
    int ce = FmsFieldToGridFunction<double>(fms_mesh, coords, mesh, nodes, false);
cout << "FmsFieldToGridFunction (for coords) returned " << ce << endl;

  }

func_exit:

  if (err) {
    delete mesh;
  } else {
     *mfem_mesh = mesh;
  }
  return err;
}

int
GridFunctionToFmsField(FmsDataCollection dc, FmsFieldDescriptor fd, FmsField f, FmsComponent comp, const GridFunction *gf) {
  if(!dc) return 1;
  if(!fd) return 2;
  if(!f) return 3;
  if(!comp) return 4;
  if(!gf) return 5;

  double *c = gf->GetData();
  int s = gf->Size();
  
  std::cout << "Coords.Size() " << s << std::endl;
  const mfem::FiniteElementSpace *fespace = gf->FESpace();
  const mfem::FiniteElementCollection *fecoll = fespace->FEColl();

  FmsFieldType ftype;
  switch(fecoll->GetContType()) {
    case mfem::FiniteElementCollection::CONTINUOUS: {
      ftype = FMS_CONTINUOUS;
      break;
    }
    case mfem::FiniteElementCollection::DISCONTINUOUS: {
      ftype = FMS_DISCONTINUOUS;
      break;
    }
    case mfem::FiniteElementCollection::TANGENTIAL: {
      ftype = FMS_HCURL; // Q: Is this correct? I don't think so.
      break;
    }
    case mfem::FiniteElementCollection::NORMAL: {
      ftype = FMS_CONTINUOUS; // Q: Is this correct? I don't think so.
      break;
    }
    default: {
      break;
    }
  }

  FmsBasisType btype = FMS_NODAL_GAUSS_CLOSED;
  /* Q: No getter for the basis, do different kinds of FECollection have implied basis?
      There are two subclasses that actually have the getter, maybe those aren't implied?
  */

  /* Q: Why is order defined on a per element basis? */
  FmsInt order = static_cast<FmsInt>(fespace->GetOrder(0));
  FmsFieldDescriptorSetComponent(fd, comp);
  FmsFieldDescriptorSetFixedOrder(fd, ftype, FMS_NODAL_GAUSS_CLOSED, order);

  FmsInt ndofs;
  FmsFieldDescriptorGetNumDofs(fd, &ndofs);
  std::cout << "FD num dofs " << ndofs << std::endl;

  FmsInt vdim = static_cast<FmsInt>(gf->VectorDim());
  FmsFieldSet(f, fd, vdim, FMS_BY_VDIM, FMS_DOUBLE, c);


  return 0;
}

//---------------------------------------------------------------------------
bool
FmsMetaDataGetInteger(FmsMetaData mdata, const std::string &key, std::vector<int> &values)
{   
    if (!mdata) false;

    bool retval = false;
    FmsMetaDataType type;
    FmsIntType int_type;
    FmsInt i, size;
    FmsMetaData *children = nullptr;
    const void *data = nullptr;
    const char *mdata_name = nullptr;
    if(FmsMetaDataGetType(mdata, &type) == 0)
    {
        switch(type)
        {
        case FMS_INTEGER:
            if(FmsMetaDataGetIntegers(mdata, &mdata_name, &int_type, &size, &data) == 0)
            {
                if(strcasecmp(key.c_str(), mdata_name) == 0)
                {
                    retval = true;

                    // Interpret the integers and store them in the std::vector<int>
                    switch(int_type)
                    {
                    case FMS_INT8:
                        for(i = 0; i < size; i++)
                           values.push_back(static_cast<int>(reinterpret_cast<const int8_t*>(data)[i]));
                        break;
                    case FMS_INT16:
                        for(i = 0; i < size; i++)
                            values.push_back(static_cast<int>(reinterpret_cast<const int16_t*>(data)[i]));
                        break;
                    case FMS_INT32:
                        for(i = 0; i < size; i++)
                            values.push_back(static_cast<int>(reinterpret_cast<const int32_t*>(data)[i]));
                        break;
                    case FMS_INT64:
                        for(i = 0; i < size; i++)
                            values.push_back(static_cast<int>(reinterpret_cast<const int64_t*>(data)[i]));
                        break;
                    case FMS_UINT8:
                        for(i = 0; i < size; i++)
                            values.push_back(static_cast<int>(reinterpret_cast<const uint8_t*>(data)[i]));
                        break;
                    case FMS_UINT16:
                        for(i = 0; i < size; i++)
                            values.push_back(static_cast<int>(reinterpret_cast<const uint16_t*>(data)[i]));
                        break;
                    case FMS_UINT32:
                        for(i = 0; i < size; i++)
                            values.push_back(static_cast<int>(reinterpret_cast<const uint32_t*>(data)[i]));
                        break;
                    case FMS_UINT64:
                        for(i = 0; i < size; i++)
                            values.push_back(static_cast<int>(reinterpret_cast<const uint64_t*>(data)[i]));
                        break;
                    default:
                        retval = false;
                        break;
                    }
                }
            }
            break;
        case FMS_META_DATA:
            if(FmsMetaDataGetMetaData(mdata, &mdata_name, &size, &children) == 0)
            {
                // Recurse to look for the key we want.
                for(i = 0; i < size && !retval; i++)
                    retval = FmsMetaDataGetInteger(children[i], key, values);
            }
            break;
        }
    }

    return retval;
}

//---------------------------------------------------------------------------
bool
FmsMetaDataGetScalar(FmsMetaData mdata, const std::string &key, std::vector<double> &values)
{   
    if (!mdata) false;

    bool retval = false;
    FmsMetaDataType type;
    FmsScalarType scal_type;
    FmsInt i, size;
    FmsMetaData *children = nullptr;
    const void *data = nullptr;
    const char *mdata_name = nullptr;
    if(FmsMetaDataGetType(mdata, &type) == 0)
    {
        switch(type)
        {
        case FMS_SCALAR:
            if(FmsMetaDataGetScalars(mdata, &mdata_name, &scal_type, &size, &data) == 0)
            {
                if(strcasecmp(key.c_str(), mdata_name) == 0)
                {
                    retval = true;

                    // Interpret the integers and store them in the std::vector<int>
                    switch(scal_type)
                    {
                    case FMS_FLOAT:
                        for(i = 0; i < size; i++)
                           values.push_back(static_cast<double>(reinterpret_cast<const float*>(data)[i]));
                        break;
                    case FMS_DOUBLE:
                        for(i = 0; i < size; i++)
                            values.push_back(reinterpret_cast<const double*>(data)[i]);
                        break;
                    default:
                        retval = false;
                        break;
                    }
                }
            }
            break;
        case FMS_META_DATA:
            if(FmsMetaDataGetMetaData(mdata, &mdata_name, &size, &children) == 0)
            {
                // Recurse to look for the key we want.
                for(i = 0; i < size && !retval; i++)
                    retval = FmsMetaDataGetScalar(children[i], key, values);
            }
            break;
        }
    }

    return retval;
}

//---------------------------------------------------------------------------
bool
FmsMetaDataGetString(FmsMetaData mdata, const std::string &key, std::string &value)
{   
    if (!mdata) false;

    bool retval = false;
    FmsMetaDataType type;
    FmsInt i, size;
    FmsMetaData *children = nullptr;
    const char *mdata_name = nullptr;
    const char *str_value = nullptr;

    if(FmsMetaDataGetType(mdata, &type) == 0)
    {
        switch(type)
        {
        case FMS_STRING:
            if(FmsMetaDataGetString(mdata, &mdata_name, &str_value) == 0)
            {
                if(strcasecmp(key.c_str(), mdata_name) == 0)
                {
                    retval = true;
                    value = str_value;
                }
            }
            break;
        case FMS_META_DATA:
            if(FmsMetaDataGetMetaData(mdata, &mdata_name, &size, &children) == 0)
            {
                // Recurse to look for the key we want.
                for(i = 0; i < size && !retval; i++)
                    retval = FmsMetaDataGetString(children[i], key, value);
            }
            break;
        }
    }

    return retval;
}

/* -------------------------------------------------------------------------- */
/* FMS to MFEM conversion function */
/* -------------------------------------------------------------------------- */

int FmsDataCollectionToDataCollection(FmsDataCollection dc, DataCollection **mfem_dc)
{
  int retval = 0;
  FmsMesh fms_mesh;
  FmsDataCollectionGetMesh(dc, &fms_mesh);

  // NOTE: The MFEM data collection has a single Mesh. Mesh has a constructor
  //       to take multiple Mesh objects but it appears to glue them together.
  Mesh *mesh = nullptr;
  int err = FmsMeshToMesh(fms_mesh, &mesh);
  if(err == 0)
  {
     std::string collection_name("collection");
     char *cn = nullptr;
     FmsDataCollectionGetName(dc, &cn);
     if(cn != nullptr)
         collection_name = cn;

     // Make a data collection that contains the mesh.
     DataCollection *mdc = new DataCollection(collection_name, mesh);
     mdc->SetOwnData(true);

     // TODO: Now do fields, etc. and add them to mdc.
#if 1
     FmsField *fields = nullptr;
     FmsInt num_fields = 0;
     if(FmsDataCollectionGetFields(dc, &fields, &num_fields) == 0)
     {
         for(FmsInt i = 0; i < num_fields; ++i)
         {
             const char *name = nullptr;
             FmsFieldGetName(fields[i], &name);
cout << "FmsDataCollectionToDataCollection: convert " << name << endl;

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
             switch(f_data_type)
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

             if(err == 0)
             {
                 mdc->RegisterField(name, gf);
             }
             else
             {
                 delete gf;
             }
         }
     }
#endif

     // If we have metadata in FMS, pass what we can through to MFEM.
     FmsMetaData mdata = NULL;
     FmsDataCollectionGetMetaData(dc, &mdata);
     if(mdata)
     {
         std::vector<int> ivalues;
         std::vector<double> dvalues;
         std::string svalue;
         if(FmsMetaDataGetInteger(mdata, "cycle", ivalues))
         {
             if(!ivalues.empty())
                 mdc->SetCycle(ivalues[0]);
         }
         if(FmsMetaDataGetScalar(mdata, "time", dvalues))
         {
             if(!dvalues.empty())
                 mdc->SetTime(dvalues[0]);
         }
         if(FmsMetaDataGetScalar(mdata, "timestep", dvalues))
         {
             if(!dvalues.empty())
                 mdc->SetTimeStep(dvalues[0]);
         }
     }

     *mfem_dc = mdc;
  }
  else
  {
cout << "FmsDataCollectionToDataCollection: mesh failed to convert. err=" << err << endl;

      retval = 1;
  }

  return retval;
}



/* -------------------------------------------------------------------------- */
/* MFEM to FMS conversion function */
/* -------------------------------------------------------------------------- */

int
MeshToFmsMesh(const Mesh *mfem_mesh, FmsMesh *outmesh)
{
#if 0
// PURPOSE: This function will reformat the MFEM mesh into an FMS mesh, making FMS calls to do so.
// NOTE: See conduitdatacollection for how we need to traverse the MFEM data...

// When I see things in MFEM that look applicable, I'm sticking them here.

    FmsMesh mesh;
    FmsMeshConstruct(mesh);

    // NOTE: I do not think MFEM's DataCollection has the concept of multiple Mesh
    FmsDomain *domains;
    FmsMeshAddDomains(mesh, "domain", 1, &domains);
    FmsDomainSetNumVertices(domains[0], 12);


    *outmesh = mesh;
#endif
    // TODO: Write me.
    return 1;
}

int
DataCollectionToFmsDataCollection(DataCollection *mfem_dc, FmsDataCollection *dc)
{
  // TODO: Write me

  const Mesh *mmesh = mfem_dc->GetMesh();
  const int num_verticies = mmesh->GetNV();
  const int num_edges = mmesh->GetNEdges();
  const int num_faces = mmesh->GetNFaces();
  const int num_elements = mmesh->GetNE();

  std::cout << "nverts: " << num_verticies << std::endl;
  std::cout << "nedges: " << num_edges << std::endl;
  std::cout << "nfaces: " << num_faces << std::endl;
  std::cout << "nele: " << num_elements << std::endl;

  FmsMesh fmesh = NULL;
  FmsMeshConstruct(&fmesh);
  FmsMeshSetPartitionId(fmesh, 0, 1);

  FmsDomain *domains = NULL;
  FmsMeshAddDomains(fmesh, "Domain", 1, &domains);
  FmsDomainSetNumVertices(domains[0], num_verticies);

  mfem::HashTable<mfem::Hashed2> edge_table;
  mfem::HashTable<mfem::Hashed4> face_table;
  std::vector<int> tri_edges;
  std::vector<int> quad_edges;
  std::vector<int> tet_faces;
  std::vector<int> hex_faces;
  for(int i = 0; i < num_elements; i++) {
    const mfem::Element *ele = mmesh->GetElement(i);
    if(!ele) return 1;
    auto ele_type = ele->GetType();
    const int *as_verts = ele->GetVertices();
    switch(ele_type) {
      case mfem::Element::POINT: {
        
        break;
      }
      case mfem::Element::SEGMENT: {
        edge_table.GetId(as_verts[0], as_verts[1]);
        break;
      }
      case mfem::Element::TRIANGLE: {
        int edgeids[3]; /* = {
          edge_table.GetId(as_verts[0], as_verts[1]),
          edge_table.GetId(as_verts[1], as_verts[2]),
          edge_table.GetId(as_verts[2], as_verts[0])
        };*/
        for(int i = 0; i < 3; i++) {
          const int *vertidxs = ele->GetEdgeVertices(i);
          edgeids[i] = edge_table.GetId(as_verts[vertidxs[0]], as_verts[vertidxs[1]]);
        }
        tri_edges.push_back(edgeids[0]);
        tri_edges.push_back(edgeids[1]);
        tri_edges.push_back(edgeids[2]);
        face_table.GetId(edgeids[0], edgeids[1], edgeids[2]);
        break;
      }
      case mfem::Element::QUADRILATERAL: {
        int edgeids[4]; /* = {
          edge_table.GetId(as_verts[0], as_verts[1]),
          edge_table.GetId(as_verts[1], as_verts[2]),
          edge_table.GetId(as_verts[2], as_verts[3]),
          edge_table.GetId(as_verts[3], as_verts[0])
        }; */
        for(int i = 0; i < 4; i++) {
          const int *vertidxs = ele->GetEdgeVertices(i);
          edgeids[i] = edge_table.GetId(as_verts[vertidxs[0]], as_verts[vertidxs[1]]);
        }
        quad_edges.push_back(edgeids[0]);
        quad_edges.push_back(edgeids[1]);
        quad_edges.push_back(edgeids[2]);
        quad_edges.push_back(edgeids[3]);
        face_table.GetId(edgeids[0], edgeids[1], edgeids[2], edgeids[3]);
        break;
      }
      case mfem::Element::TETRAHEDRON: {
        // Have to build the edges then build the faces
        int edgeids[6];
        for(int i = 0; i < 6; i++) {
          const int *vertidxs = ele->GetEdgeVertices(i);
          edgeids[i] = edge_table.GetId(as_verts[vertidxs[0]], as_verts[vertidxs[1]]);
        }

        // Build the 4 triangles that make up the tet
        int tris[4][3] = {
          {edgeids[0], edgeids[2], edgeids[1]}, // acb
          {edgeids[0], edgeids[4], edgeids[3]}, // aed
          {edgeids[2], edgeids[3], edgeids[5]}, // cdf
          {edgeids[1], edgeids[5], edgeids[4]}  // bfe
        };
        tri_edges.push_back(tris[0][0]); tri_edges.push_back(tris[0][1]); tri_edges.push_back(tris[0][2]);
        tri_edges.push_back(tris[1][0]); tri_edges.push_back(tris[1][1]); tri_edges.push_back(tris[1][2]);
        tri_edges.push_back(tris[2][0]); tri_edges.push_back(tris[2][1]); tri_edges.push_back(tris[2][2]);
        tri_edges.push_back(tris[3][0]); tri_edges.push_back(tris[3][1]); tri_edges.push_back(tris[3][2]);
        
        // Finally create the tet
        int faceids[4];
        faceids[0] = face_table.GetId(tris[0][0], tris[0][1], tris[0][2]); 
        faceids[1] = face_table.GetId(tris[1][0], tris[1][1], tris[1][2]); 
        faceids[2] = face_table.GetId(tris[2][0], tris[2][1], tris[2][2]); 
        faceids[3] = face_table.GetId(tris[3][0], tris[3][1], tris[3][2]); 
        tet_faces.push_back(faceids[0]);
        tet_faces.push_back(faceids[1]);
        tet_faces.push_back(faceids[2]);
        tet_faces.push_back(faceids[3]);
        break;
      }
      case mfem::Element::HEXAHEDRON: {
        // Build the edges
        int edgeids[12] = {
          // Bottom
          edge_table.GetId(as_verts[0], as_verts[1]),
          edge_table.GetId(as_verts[1], as_verts[2]),
          edge_table.GetId(as_verts[2], as_verts[3]),
          edge_table.GetId(as_verts[3], as_verts[0]),
          // Top
          edge_table.GetId(as_verts[4], as_verts[5]),
          edge_table.GetId(as_verts[5], as_verts[6]),
          edge_table.GetId(as_verts[6], as_verts[7]),
          edge_table.GetId(as_verts[7], as_verts[4]),
          // Sides
          edge_table.GetId(as_verts[0], as_verts[4]),
          edge_table.GetId(as_verts[1], as_verts[5]),
          edge_table.GetId(as_verts[2], as_verts[6]),
          edge_table.GetId(as_verts[3], as_verts[7]),
        };
        // for(int i = 0; i < 12; i++) {
        //   const int *vertidxs = ele->GetEdgeVertices(i);
        //   edgeids[i] = edge_table.GetId(as_verts[vertidxs[0]], as_verts[vertidxs[1]]);
        // }
        // Build the 6 quads that make up the hex
        int quads[6][4] = {
          {edgeids[0], edgeids[9], edgeids[4], edgeids[8]}, // ajei
          {edgeids[2], edgeids[11], edgeids[6], edgeids[10]}, // clgk
          {edgeids[0], edgeids[3], edgeids[2], edgeids[1]}, // adcb
          {edgeids[4], edgeids[5], edgeids[6], edgeids[7]}, // efgh
          {edgeids[3], edgeids[8], edgeids[7], edgeids[11]}, // dihl
          {edgeids[1], edgeids[10], edgeids[5], edgeids[9]} // gkfj
        };
        for(int j = 0; j < 6; j++) {
          // Dont add this quad to the vector if it's already in the face table
          if(face_table.FindId(quads[j][0], quads[j][1], quads[j][2], quads[j][3]) > 0) continue;
          for(int k = 0; k < 4; k++) {
            quad_edges.push_back(quads[j][k]);
          }
        }
        int faceids[6] = {
          face_table.GetId(quads[0][0], quads[0][1], quads[0][2], quads[0][3]), 
          face_table.GetId(quads[1][0], quads[1][1], quads[1][2], quads[1][3]),
          face_table.GetId(quads[2][0], quads[2][1], quads[2][2], quads[2][3]),
          face_table.GetId(quads[3][0], quads[3][1], quads[3][2], quads[3][3]),
          face_table.GetId(quads[4][0], quads[4][1], quads[4][2], quads[4][3]),
          face_table.GetId(quads[5][0], quads[5][1], quads[5][2], quads[5][3])
        };
        for(int j = 0; j < 6; j++) {
          hex_faces.push_back(faceids[j]);
        }
        break;
      }
      default: {
        // ERROR: Not implemented
        return 2;
      }
    }
  }

  if(edge_table.Size()) {
    std::vector<int> edge_verts;
    edge_verts.reserve(edge_table.Size() * 2 + 1);
    for(int i = 0; i < edge_table.Size(); i++) {
      const auto &edge = edge_table[i];
      edge_verts.push_back(edge.p1);
      edge_verts.push_back(edge.p2);
    }
    std::cout << "EDGES:";
    for(int i = 0; i < edge_verts.size(); i++) {
      if(i%2==0) std::cout << std::endl << "\t" << i/2 << ": ";
      std::cout << edge_verts[i] << " ";
    }
    std::cout << std::endl;
    
    const FmsInt nedges = edge_verts.size() / 2u;
    FmsDomainSetNumEntities(domains[0], FMS_EDGE, FMS_INT32, nedges);
    FmsDomainAddEntities(domains[0], FMS_EDGE, NULL, FMS_INT32, edge_verts.data(), nedges);
  }
  else {
    // ERROR?
    return 3;
  }

  if(tri_edges.size()) {
    std::cout << "TRIANGLES:";
    for(int i = 0; i < tri_edges.size(); i++) {
      if(i%3==0) std::cout << std::endl << "\t" << i/3 << ": ";
      std::cout << tri_edges[i] << " ";
    }
    std::cout << std::endl;

    const FmsInt ntris = tri_edges.size() / 3u;
    FmsDomainSetNumEntities(domains[0], FMS_TRIANGLE, FMS_INT32, ntris);
    FmsDomainAddEntities(domains[0], FMS_TRIANGLE, NULL, FMS_INT32, tri_edges.data(), ntris);
  }

  if(quad_edges.size()) {
      std::cout << "QUADS:";
    for(int i = 0; i < quad_edges.size(); i++) {
      if(i%4==0) std::cout << std::endl << "\t" << i/4 << ": ";
      std::cout << quad_edges[i] << " ";
    }
    std::cout << std::endl;

    const FmsInt nquads = quad_edges.size() / 4u;
    FmsDomainSetNumEntities(domains[0], FMS_QUADRILATERAL, FMS_INT32, nquads);
    FmsDomainAddEntities(domains[0], FMS_QUADRILATERAL, NULL, FMS_INT32, quad_edges.data(), nquads);
  }

  /* Q: All the edges are stored in one array so the IDs are implied (the index in the array).
    3D elements refer to faces by their face ID, faces are not stored in one array because they are stored
    between TRI and QUAD arrays. How does FMS know that it should lookup ID=5 from triangles or quads - does this even matter?
  */
  if(tet_faces.size()) {
    std::cout << "TETS:";
    for(int i = 0; i < tet_faces.size(); i++) {
      if(i%4==0) std::cout << std::endl << "\t" << i/4 << ": ";
      std::cout << tet_faces[i] << " ";
    }
    std::cout << std::endl;

    const FmsInt ntets = tet_faces.size() / 4u;
    FmsDomainSetNumEntities(domains[0], FMS_TETRAHEDRON, FMS_INT32, ntets);
    FmsDomainAddEntities(domains[0], FMS_TETRAHEDRON, NULL, FMS_INT32, tet_faces.data(), ntets);
  }

  if(hex_faces.size()) {
    std::cout << "HEXES:";
    for(int i = 0; i < hex_faces.size(); i++) {
      if(i%6==0) std::cout << std::endl << "\t" << i/6 << ": ";
      std::cout << hex_faces[i] << " ";
    }
    std::cout << std::endl;

    const FmsInt nhexes = hex_faces.size() / 6u;
    FmsDomainSetNumEntities(domains[0], FMS_HEXAHEDRON, FMS_INT32, nhexes);
    FmsDomainAddEntities(domains[0], FMS_HEXAHEDRON, NULL, FMS_INT32, hex_faces.data(), nhexes);
  }

  // if(wed_faces.size()) {
  //   const FmsInt nweds = wed_faces.size() / 5u;
  //   FmsDomainSetNumEntities(domains[0], FMS_WEDGE, FMS_INT32, nweds);
  //   FmsDomainAddEntities(domains[0], FMS_WEDGE, NULL, FMS_INT32, wed_faces.data(), nweds);
  // }

  FmsComponent volume;
  FmsMeshAddComponent(fmesh, "volume", &volume);
  FmsComponentAddDomain(volume, domains[0]);

  // TODO: Add boundaries

  FmsMeshFinalize(fmesh);
  FmsMeshValidate(fmesh);

  FmsDataCollectionCreate(fmesh, mfem_dc->GetCollectionName().c_str(), dc);

  // Add the coordinates field to the data collection
  const mfem::GridFunction *mcoords = mmesh->GetNodes();
  if(mcoords) {
    FmsFieldDescriptor fcoords_fd = NULL;
    FmsField fcoords = NULL;
    FmsDataCollectionAddFieldDescriptor(*dc, "CoordsDescriptor", &fcoords_fd);
    FmsDataCollectionAddField(*dc, "Coords", &fcoords);
    GridFunctionToFmsField(*dc, fcoords_fd, fcoords, volume, mcoords);
    FmsComponentSetCoordinates(volume, fcoords);
  }
  else {
    // ERROR?
  }

  const auto &fields = mfem_dc->GetFieldMap();
  for(const auto &pair : fields) {
    FmsFieldDescriptor fd = NULL;
    FmsField f = NULL;
    std::string fd_name(pair.first + "Collection");
    FmsDataCollectionAddFieldDescriptor(*dc, fd_name.c_str(), &fd);
    FmsDataCollectionAddField(*dc, pair.first.c_str(), &f);
    GridFunctionToFmsField(*dc, fd, f, volume, pair.second); // TODO: Volume isn't always going to be correct
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

  return 0;
}

} // end namespace mfem

