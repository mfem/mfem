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

#ifndef MFEM_PNCSUBMESH
#define MFEM_PNCSUBMESH

#include "../../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "../pncmesh.hpp"
#include "psubmesh.hpp"
#include "submesh_utils.hpp"
#include <unordered_map>

namespace mfem
{

/**
 * @brief Class representing a Parallel Nonconformal SubMesh. This is only used
 * by ParSubMesh.
 */
class ParNCSubMesh : public ParNCMesh
{
   friend class ParSubMesh; ///< Only ParSubMesh can use methods in this class
public:
   using From = SubMesh::From; ///< Convenience type alias
   /**
   * @brief Check if NCMesh @a m is a ParNCSubMesh.
   *
   * @param m The input Mesh
   */
   static bool IsParNCSubMesh(const NCMesh *m)
   {
      return dynamic_cast<const ParNCSubMesh *>(m) != nullptr;
   }
   /// Get the parent ParNCMesh object
   const ParNCMesh* GetParent() const
   {
      return parent_;
   }

protected:
   /// protected constructor
   ParNCSubMesh(ParSubMesh& submesh, const ParNCMesh &parent, From from,
                const Array<int> &attributes);

   /// The parent ParNCMesh. Not owned.
   const ParNCMesh *parent_;

   /// Mapping from submesh element nc ids (index of the array), to the parent
   /// element ids. If from a boundary, these map to faces in the parent.
   Array<int> parent_element_ids_;

   /// Mapping from ParNCSubMesh node ids (index of the array), to the parent
   /// NCMesh node ids.
   Array<int> parent_node_ids_;

   /// Mapping from parent NCMesh node ids to submesh NCMesh node ids.
   // Inverse map of parent_node_ids_.
   std::unordered_map<int, int> parent_to_submesh_node_ids_;

   /// Mapping from parent NCMesh element ids to submesh NCMesh element ids.
   // Inverse map of parent_element_ids_.
   std::unordered_map<int, int> parent_to_submesh_element_ids_;

   // Helper friend methods for construction.
   friend void SubMeshUtils::ConstructFaceTree<ParNCSubMesh>
   (ParNCSubMesh &submesh, const Array<int> &attributes);
   friend void SubMeshUtils::ConstructVolumeTree<ParNCSubMesh>
   (ParNCSubMesh &submesh, const Array<int> &attributes);

   /**
    * @brief Accessor for parent nodes
    * @details Required to bypass access protection in parent class.
    *
    * @return const HashTable<Node>&
    */
   const HashTable<Node> &ParentNodes() const { return parent_->nodes; }

   /**
    * @brief Accessor for parent faces
    * @details Required to bypass access protection in parent class.
    *
    * @return const HashTable<Face>&
    */
   const HashTable<Face> &ParentFaces() const { return parent_->faces; }
};

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_PNCSUBMESH
