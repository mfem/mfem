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

#include "ncsubmesh.hpp"

#include <unordered_map>
#include "submesh_utils.hpp"
#include "submesh.hpp"

namespace mfem
{

using namespace SubMeshUtils;

NCSubMesh::NCSubMesh(SubMesh& submesh, const NCMesh &parent, From from,
                     const Array<int> &attributes)
   : NCMesh(), parent_(&parent)
{
   Dim = submesh.Dimension();
   spaceDim = submesh.SpaceDimension();
   MyRank = 0;
   Iso = true;
   Legacy = false;

   if (from == From::Domain)
   {
      SubMeshUtils::ConstructVolumeTree(*this, attributes);
   }
   else if (from == From::Boundary)
   {
      SubMeshUtils::ConstructFaceTree(*this, attributes);
   }

   // Loop over all nodes, and reparent based on the node relations of the
   // parent
   for (int i = 0; i < parent_node_ids_.Size(); i++)
   {
      const auto &parent_node = parent.nodes[parent_node_ids_[i]];
      const int submesh_p1 = parent_to_submesh_node_ids_[parent_node.p1];
      const int submesh_p2 = parent_to_submesh_node_ids_[parent_node.p2];
      nodes.Reparent(i, submesh_p1, submesh_p2);
   }

   nodes.UpdateUnused();
   for (int i = 0; i < elements.Size(); i++)
   {
      if (elements[i].IsLeaf())
      {
         // Register all faces
         RegisterFaces(i);
      }
   }

   InitRootElements();
   InitRootState(root_state.Size());
   InitGeomFlags();
   Update(); // Fills in secondary information based off of elements, nodes and faces.

   // If parent has coordinates defined, copy the relevant portion
   if (parent.coordinates.Size() > 0)
   {
      coordinates.SetSize(3*parent_node_ids_.Size());
      parent.tmp_vertex = new TmpVertex[parent.nodes.NumIds()];
      for (int n = 0; n < parent_node_ids_.Size(); n++)
      {
         std::memcpy(&coordinates[3*n], parent.CalcVertexPos(parent_node_ids_[n]),
                     3*sizeof(real_t));
      }
      delete [] parent.tmp_vertex;
   }

   // The element indexing was changed as part of generation of leaf elements.
   // We need to update the map.
   if (from == From::Domain)
   {
      // The element indexing was changed as part of generation of leaf
      // elements. We need to update the map.
      submesh.parent_to_submesh_element_ids_ = -1;
      for (int i = 0; i < submesh.parent_element_ids_.Size(); i++)
      {
         submesh.parent_element_ids_[i] =
            parent.elements[parent_element_ids_[leaf_elements[i]]].index;
         submesh.parent_to_submesh_element_ids_[submesh.parent_element_ids_[i]] = i;
      }
   }
   else
   {
      submesh.parent_to_submesh_element_ids_ = -1;
      // parent elements are BOUNDARY elements, need to map face index to be.
      const auto &parent_face_to_be = submesh.GetParent()->GetFaceToBdrElMap();
      MFEM_ASSERT(NElements == submesh.GetNE(), "!");
      auto new_parent_to_submesh_element_ids = submesh.parent_to_submesh_element_ids_;
      Array<int> new_parent_element_ids;
      new_parent_element_ids.Reserve(submesh.parent_element_ids_.Size());
      for (int i = 0; i < submesh.parent_element_ids_.Size(); i++)
      {
         new_parent_element_ids.Append(
            parent_face_to_be[parent.faces[parent_element_ids_[leaf_elements[i]]].index]);
         new_parent_to_submesh_element_ids[new_parent_element_ids[i]] = i;
      }

      MFEM_ASSERT(new_parent_element_ids.Size() == submesh.parent_element_ids_.Size(),
                  "!");
#ifdef MFEM_DEBUG
      for (auto x : new_parent_element_ids)
      {
         MFEM_ASSERT(std::find(submesh.parent_element_ids_.begin(),
                               submesh.parent_element_ids_.end(), x)
                     != submesh.parent_element_ids_.end(),
                     x << " not found in submesh.parent_element_ids_");
      }
      for (auto x : submesh.parent_element_ids_)
      {
         MFEM_ASSERT(std::find(new_parent_element_ids.begin(),
                               new_parent_element_ids.end(), x)
                     != new_parent_element_ids.end(), x << " not found in new_parent_element_ids_");
      }
#endif
      submesh.parent_element_ids_ = std::move(new_parent_element_ids);
      submesh.parent_to_submesh_element_ids_ =
         std::move(new_parent_to_submesh_element_ids);
   }
}

} // namespace mfem
