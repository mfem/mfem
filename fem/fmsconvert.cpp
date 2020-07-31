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
  for (FmsInt part_id = 0; part_id < n_main_parts; part_id++) {
    for (int et = FMS_VERTEX; et < FMS_NUM_ENTITY_TYPES; et++) {
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
        for (FmsInt i = 0; i < num_elems; i++) {
          mesh->AddTriangle(
            &ents_verts[3*i], elem_tag ? attr[elem_offset+i] : 1);
        }
        break;
      case FMS_QUADRILATERAL:
        for (FmsInt i = 0; i < num_elems; i++) {
          mesh->AddQuad(&ents_verts[4*i], elem_tag ? attr[elem_offset+i] : 1);
        }
        break;
      case FMS_TETRAHEDRON:
        for (FmsInt i = 0; i < num_elems; i++) {
          mesh->AddTet(&ents_verts[4*i], elem_tag ? attr[elem_offset+i] : 1);
        }
        break;
      case FMS_HEXAHEDRON:
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

//---------------------------------------------------------------------------
#if 0
   GridFunction *res = NULL;
   mfem::FiniteElementCollection *fec = FiniteElementCollection::New(
                                           fec_name.c_str());
   mfem::FiniteElementSpace *fes = new FiniteElementSpace(mesh,
                                                          fec,
                                                          vdim,
                                                          ordering);

   if (zero_copy)
   {
      res = new GridFunction(fes,const_cast<double*>(vals_ptr));
   }
   else
   {
      // copy case, this constructor will alloc the space for the GF data
      res = new GridFunction(fes);
      // create an mfem vector that wraps the conduit data
      Vector vals_vec(const_cast<double*>(vals_ptr),fes->GetVSize());
      // copy values into the result
      (*res) = vals_vec;
   }
#endif

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
DataCollectionToFmsDataCollection(const DataCollection *mfem_dc, FmsDataCollection *dc)
{
    // TODO: Write me.

    


    *dc = nullptr;
    return 1;
}

} // end namespace mfem

