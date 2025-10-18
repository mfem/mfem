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

#ifndef MFEM_PSUBMESH
#define MFEM_PSUBMESH

#include "../../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "ptransfermap.hpp"
#include "../pmesh.hpp"
#include "../../fem/pgridfunc.hpp"
#include "submesh.hpp"

namespace mfem
{

class ParNCSubMesh;

/**
 * @brief Subdomain representation of a topological parent in another ParMesh.
 *
 * ParSubMesh is a subdomain representation of a ParMesh defined on its parents
 * attributes. The current implementation creates either a domain or surface
 * subset of the parent Mesh and reuses the parallel distribution.
 *
 * The attributes are taken from the parent. That means if a volume is extracted
 * from a volume, it has the same domain attribute as the parent. Its new
 * boundary attributes are, for any boundary common to the parent and the new
 * submesh, the boundary attribute of the parent; and, for all new boundaries,
 * a single, generated, common attribute equal to one plus the largest boundary
 * attribute of the parent.
 *
 * If a surface is extracted from a volume, the boundary attribute from the
 * parent is assigned to be the new domain attribute. Its new boundary attribute
 * is a single, generated, common attribute equal to one plus the largest
 * boundary attribute of the parent.
 *
 * For more customized boundary attributes, the resulting ParSubMesh has to be
 * postprocessed.
 *
 * ParSubMesh maintains the parallel distribution of the elements on
 * corresponding processors.
 */

class ParSubMesh : public ParMesh
{
   friend class ParNCSubMesh;
public:
   using From = SubMesh::From; ///< Convenience type-alias.
   ParSubMesh() = delete;

   /**
    * @brief Create a domain ParSubMesh from its parent.
    *
    * The ParSubMesh object expects the parent ParMesh object to be valid for
    * the entire object lifetime. The @a domain_attributes have to mark exactly
    * one connected subset of the parent Mesh.
    *
    * @param[in] parent Parent ParMesh
    * @param[in] domain_attributes Domain attributes to extract
    */
   static ParSubMesh CreateFromDomain(const ParMesh &parent,
                                      const Array<int> &domain_attributes);

   /**
   * @brief Create a surface ParSubMesh from its parent.
   *
   * The ParSubMesh object expects the parent ParMesh object to be valid for the
   * entire object lifetime. The @a boundary_attributes have to mark exactly one
   * connected subset of the parent Mesh.
   *
   * @param[in] parent Parent ParMesh
   * @param[in] boundary_attributes Boundary attributes to extract
   */
   static ParSubMesh CreateFromBoundary(const ParMesh &parent,
                                        const Array<int> &boundary_attributes);

   /**
    * @brief Get the parent ParMesh object
    */
   const ParMesh* GetParent() const
   {
      return &parent_;
   }

   /**
    * @brief Get the From indicator.
    *
    * Indicates whether the ParSubMesh has been created from a domain or
    * surface.
    */
   SubMesh::From GetFrom() const
   {
      return from_;
   }

   /**
    * @brief Get the parent element id map.
    *
    * ParSubMesh element id (array index) to parent ParMesh element id.
    */
   const Array<int>& GetParentElementIDMap() const
   {
      return parent_element_ids_;
   }

   /**
    * @brief Get the parent vertex id map.
    *
    * ParSubMesh vertex id (array index) to parent ParMesh vertex id.
    */
   const Array<int>& GetParentVertexIDMap() const
   {
      return parent_vertex_ids_;
   }

   /**
    * @brief Get the parent edge id map
    *
    * Submesh edge id (array index) to parent Mesh edge id.
    */
   const Array<int>& GetParentEdgeIDMap() const
   {
      return parent_edge_ids_;
   }

   /**
    * @brief Get the parent face id map.
    *
    * ParSubMesh face id (array index) to parent ParMesh face id.
    */
   const Array<int>& GetParentFaceIDMap() const
   {
      return parent_face_ids_;
   }

   /**
    * @brief Get the relative face orientations
    *
    * ParSubMesh element id (array index) to parent ParMesh face orientation.
    */
   const Array<int>& GetParentFaceOrientations() const
   {
      return parent_face_ori_;
   }

   /**
    * @brief Get the submesh element corresponding to a parent element. -1 ==
    * not present.
    * @param pe The parent element id.
    * @return int
    */
   int GetSubMeshElementFromParent(int pe) const
   {
      return (pe == -1 || pe >= parent_to_submesh_element_ids_.Size())
             ? -1 : parent_to_submesh_element_ids_[pe];
   }

   /**
    * @brief Get the submesh vertex corresponding to a parent element. -1 == not
    * present.
    * @param pv The parent vertex id.
    * @return int
    */
   int GetSubMeshVertexFromParent(int pv) const
   {
      return (pv == -1 || pv >= parent_to_submesh_vertex_ids_.Size())
             ? -1 : parent_to_submesh_vertex_ids_[pv];
   }

   /**
    * @brief Get the submesh edge corresponding to a parent element. -1 == not
    * present.
    * @param pe The parent edge id.
    * @return int
    */
   int GetSubMeshEdgeFromParent(int pe) const
   {
      return (pe == -1 || pe >= parent_to_submesh_edge_ids_.Size())
             ? pe : parent_to_submesh_edge_ids_[pe];
   }

   /**
    * @brief Get the submesh face corresponding to a parent element. -1 == not
    * present.
    * @param pf The parent face id.
    * @return int
    */
   int GetSubMeshFaceFromParent(int pf) const
   {
      return (pf == -1 || pf >= parent_to_submesh_face_ids_.Size())
             ? pf : parent_to_submesh_face_ids_[pf];
   }

   /**
    * @brief Transfer the dofs of a ParGridFunction.
    *
    * The @a src ParGridFunction can either be defined on a ParMesh or a
    * ParSubMesh and is transferred appropriately.
    *
    * @note Either @a src or @a dst has to be defined on a ParSubMesh.
    *
    * @param[in] src
    * @param[out] dst
    */
   static void Transfer(const ParGridFunction &src, ParGridFunction &dst);

   /**
    * @brief Create a Transfer Map object.
    *
    * The @a src ParGridFunction can either be defined on a ParMesh or a
    * ParSubMesh and is transferred appropriately.
    *
    * @note Either @a src or @a dst has to be defined on a ParSubMesh.
    */
   static ParTransferMap CreateTransferMap(const ParGridFunction &src,
                                           const ParGridFunction &dst);

   /**
   * @brief Check if ParMesh @a m is a ParSubMesh.
   *
   * @param m The input ParMesh
   */
   static bool IsParSubMesh(const ParMesh *m)
   {
      return dynamic_cast<const ParSubMesh *>(m) != nullptr;
   }

private:
   ParSubMesh(const ParMesh &parent, SubMesh::From from,
              const Array<int> &attributes);

   /**
    * @brief Find shared vertices on the ParSubMesh.
    *
    * Uses the parent GroupCommunicator to determine shared vertices.
    * Collective. Limited to 32 ranks.
    *
    * Array of integer bitfields to indicate if rank X (bit location) has shared
    * vtx Y (array index).
    *
    * Example with 4 ranks and X shared vertices.
    * * R0-R3 indicate ranks 0 to 3
    * * v0-v3 indicate vertices 0 to 3
    * The array is used as follows (only relevant bits shown):
    *
    * rhvtx[0] = [0...0 1 0 1] Rank 0 and 2 have shared vertex 0
    * rhvtx[1] = [0...0 1 1 1] Rank 0, 1 and 2 have shared vertex 1
    * rhvtx[2] = [0...0 0 1 1] Rank 0 and 1 have shared vertex 2
    * rhvtx[3] = [0...1 0 1 0] Rank 1 and 3 have shared vertex 3. Corner case
    * which shows that a rank can contribute the shared vertex, but the adjacent
    * element or edge might not be included in the relevant SubMesh.
    *
    *  +--------------+--------------+...
    *  |              |v0            |
    *  |      R0      |      R2      |     R3
    *  |              |              |
    *  +--------------+--------------+...
    *  |              |v1            |
    *  |      R0      |      R1      |     R3
    *  |              |v2            |v3
    *  +--------------+--------------+...
    *
    * @param[out] rhvtx Encoding of which rank contains which vertex.
    */
   void FindSharedVerticesRanks(Array<int> &rhvtx);

   /**
    * @brief Find shared edges on the ParSubMesh.
    *
    * Uses the parent GroupCommunicator to determine shared edges. Collective.
    * Limited to groups containing less than 32 ranks.
    *
    * See FindSharedVerticesRanks for the encoding for @a rhe.
    *
    * @param[out] rhe Encoding of which rank contains which edge.
    */
   void FindSharedEdgesRanks(Array<int> &rhe);


   /**
    * @brief Find shared faces on the ParSubMesh.
    *
    * Uses the parent GroupCommunicator to determine shared faces. Collective.
    *
    * The encoded output arrays @a rhq and @a rht contain either 0, 1 or 2 for
    * each shared face.
    *
    * 0: Face might have been a shared face in the parent ParMesh, but is
    * not contained in the ParSubMesh.
    * 1: Face is contained in the ParSubMesh but only on one rank.
    * 2: Face is contained in the ParSubMesh and shared by two ranks. This
    * is the only feasible entity of a shared face in a ParSubMesh.
    *
    * @param[out] rhq Encoding of which rank contains which face quadrilateral.
    */
   void FindSharedFacesRanks(Array<int>& rht, Array<int> &rhq);

   /**
    * @brief Append shared vertices encoded in @a rhvtx to @a groups.
    *
    * @param[in,out] groups
    * @param[in,out] rhvtx Encoding of which rank contains which vertex. The
    * output is reused s.t. the array index i (the vertex id) is the associated
    * group.
    */
   void AppendSharedVerticesGroups(ListOfIntegerSets &groups, Array<int> &rhvtx);

   /**
    * @brief Append shared edges encoded in @a rhe to @a groups.
    *
    * @param[in,out] groups
    * @param[in,out] rhe Encoding of which rank contains which edge. The output
    * is reused s.t. the array index i (the edge id) is the associated group.
    */
   void AppendSharedEdgesGroups(ListOfIntegerSets &groups, Array<int> &rhe);

   /**
    * @brief Append shared faces encoded in @a rhq and @a rht to @a groups.
    *
    * @param[in,out] groups
    * @param[in,out] rht Encoding of which rank contains which face triangle.
    * The output is reused s.t. the array index i (the face triangle id) is the
    * associated group. "Rank Has Triangle"
    * @param[in,out] rhq Encoding of which rank contains which face
    * quadrilateral. The output is reused s.t. the array index i (the face
    * quadrilateral id) is the associated group. "Rank Has Quad"
    */
   void AppendSharedFacesGroups(ListOfIntegerSets &groups, Array<int>& rht,
                                Array<int> &rhq);

   /**
    * @brief Build vertex group.
    *
    * @param[in] ngroups Number of groups.
    * @param[in] rhvtx Encoding of which rank contains which vertex.
    * @param[in] nsverts Number of shared vertices.
    */
   void BuildVertexGroup(int ngroups, const Array<int>& rhvtx, int& nsverts);

   /**
    * @brief Build edge group.
    *
    * @param[in] ngroups Number of groups.
    * @param[in] rhe Encoding of which rank contains which edge.
    * @param[in] nsedges Number of shared edges.
    */
   void BuildEdgeGroup(int ngroups, const Array<int>& rhe, int& nsedges);

   /**
    * @brief Build face group.
    *
    * @param[in] ngroups Number of groups.
    * @param[in] rht Encoding of which rank contains which face triangle.
    * @param[in] nstrias Number of shared face triangles.
    * @param[in] rhq Encoding of which rank contains which face quadrilateral.
    * @param[in] nsquads Number of shared face quadrilaterals.
    */
   void BuildFaceGroup(int ngroups, const Array<int>& rht, int& nstrias,
                       const Array<int>& rhq, int& nsquads);

   /**
    * @brief Build the shared vertex to local vertex mapping.
    *
    * @param nsverts Number of shared vertices.
    * @param rhvtx Encoding of which rank contains which vertex.
    */
   void BuildSharedVerticesMapping(const int nsverts, const Array<int>& rhvtx);

   /**
   * @brief Build the shared edge to local edge mapping.
   *
   * @param[in] nsedges Number of shared edges.
   * @param[in] rhe Encoding of which rank contains which edge.
   */
   void BuildSharedEdgesMapping(const int nsedges, const Array<int>& rhe);

   /**
    * @brief Build the shared faces to local faces mapping.
    *
    * Shared faces are divided into triangles and quadrilaterals.
    *
    * @param[in] nstrias Number of shared face triangles.
    * @param[in] rht Encoding of which rank contains which face triangle.
    * @param[in] nsquads Number of shared face quadrilaterals.
    * @param[in] rhq Encoding of which rank contains which face quadrilateral.
    */
   void BuildSharedFacesMapping(const int nstrias, const Array<int>& rht,
                                const int nsquads, const Array<int>& rhq);


   std::unordered_map<int, int>
   FindGhostBoundaryElementAttributes() const;

   /// The parent Mesh
   const ParMesh &parent_;

   /// Optional nonconformal submesh. Managed via pncmesh pointer in base class.
   ParNCSubMesh *pncsubmesh_;

   /// Indicator from which part of the parent ParMesh the ParSubMesh is going
   /// to be created.
   SubMesh::From from_;

   /// Attributes on the parent ParMesh on which the ParSubMesh is created.
   /// Could either be domain or boundary attributes (determined by from_).
   Array<int> attributes_;

   /// Mapping from ParSubMesh element ids (index of the array), to the parent
   /// ParMesh element ids.
   Array<int> parent_element_ids_;

   /// Mapping from ParSubMesh vertex ids (index of the array), to the parent
   /// ParMesh vertex ids.
   Array<int> parent_vertex_ids_;

   /// Mapping from ParSubMesh edge ids (index of the array), to the parent
   /// ParMesh edge ids.
   Array<int> parent_edge_ids_;

   /// Mapping from ParSubMesh face ids (index of the array), to the parent
   /// ParMesh face ids.
   Array<int> parent_face_ids_;

   /// Mapping from SubMesh face ids (index of the array), to the orientation of
   /// the face relative to the parent face.
   Array<int> parent_face_ori_;

   /// Mapping from parent ParMesh element ids (index of the array), to the
   /// ParSubMesh element ids. Inverse map of parent_element_ids_.
   Array<int> parent_to_submesh_element_ids_;

   /// Mapping from parent ParMesh vertex ids (index of the array), to the
   /// ParSubMesh vertex ids. Inverse map of parent_vertex_ids_.
   Array<int> parent_to_submesh_vertex_ids_;

   /// Mapping from parent ParMesh edge ids (index of the array), to the
   /// ParSubMesh edge ids. Inverse map of parent_edge_ids_.
   Array<int> parent_to_submesh_edge_ids_;

   /// Mapping from parent ParMesh face ids (index of the array), to the
   /// ParSubMesh face ids. Inverse map of parent_face_ids_.
   Array<int> parent_to_submesh_face_ids_;
};

} // namespace mfem

#endif // MFEM_USE_MPI

#endif
