// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_NCSUBMESH
#define MFEM_NCSUBMESH

#include "../ncmesh.hpp"
#include "submesh.hpp"
#include "submesh_utils.hpp"
#include <unordered_map>

namespace mfem
{

/**
 * @brief Class representing a Nonconformal SubMesh. This is only used by
 * SubMesh.
 */
class NCSubMesh : public NCMesh
{
   friend class SubMesh; ///< Only SubMesh can use methods in this class
public:
   using From = SubMesh::From; ///< Convenience type alias
   /// Get the parent NCMesh object
   const NCMesh* GetParent() const
   {
      return parent_;
   }

   /**
   * @brief Check if NCMesh @a m is a NCSubMesh.
   *
   * @param m The input NCMesh
   */
   static bool IsNCSubMesh(const NCMesh *m)
   {
      return dynamic_cast<const NCSubMesh *>(m) != nullptr;
   }
private:

   /// Private constructor
   NCSubMesh(SubMesh& submesh, const NCMesh &parent, From from,
             const Array<int> &attributes);

   /// The parent NCMesh. Not owned.
   const NCMesh *parent_;

   /// Mapping from submesh element nc ids (index of the array), to the parent
   /// element ids. If from a boundary, these map to faces in the parent.
   Array<int> parent_element_ids_;

   /// Mapping from NCSubMesh node ids (index of the array), to the parent
   /// NCMesh node ids.
   Array<int> parent_node_ids_;

   /// Mapping from parent NCMesh node ids to submesh NCMesh node ids.
   // Inverse map of parent_node_ids_.
   std::unordered_map<int, int> parent_to_submesh_node_ids_;

   /// Mapping from parent NCMesh element ids to submesh NCMesh element ids.
   // Inverse map of parent_element_ids_.
   std::unordered_map<int, int> parent_to_submesh_element_ids_;

   // Helper friend methods for construction.
   friend void SubMeshUtils::ConstructFaceTree<NCSubMesh>(NCSubMesh &submesh,
                                                          const Array<int> &attributes);
   friend void SubMeshUtils::ConstructVolumeTree<NCSubMesh>(NCSubMesh &submesh,
                                                            const Array<int> &attributes);

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

#endif // MFEM_NCSUBMESH
