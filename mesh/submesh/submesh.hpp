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

#ifndef MFEM_SUBMESH
#define MFEM_SUBMESH

#include "../mesh.hpp"
#include "transfermap.hpp"

namespace mfem
{

class NCSubMesh;

/**
 * @brief Subdomain representation of a topological parent in another Mesh.
 *
 * SubMesh is a subdomain representation of a Mesh defined on its parents
 * attributes. The current implementation creates either a domain or surface
 * subset of the parents Mesh and reuses the parallel distribution.
 *
 * The attributes are taken from the parent. That means if a volume is extracted
 * from a volume, it has the same domain attribute as the parent. Its boundary
 * attributes are generated (there will be one boundary attribute 1 for all of
 * the boundaries).
 *
 * If a surface is extracted from a volume, the boundary attribute from the
 * parent is assigned to be the new domain attribute. Its boundary attributes
 * are generated (there will be one boundary attribute 1 for all of the
 * boundaries).
 *
 * For more customized boundary attributes, the resulting SubMesh has to be
 * postprocessed.
 */
class SubMesh : public Mesh
{
   friend class NCSubMesh;
public:
   /// Indicator from which part of the parent Mesh the SubMesh is created.
   enum class From
   {
      Domain,
      Boundary
   };

   SubMesh() = delete;
   SubMesh(SubMesh &&) = default;
   SubMesh &operator=(SubMesh &&) = default;

   /**
    * @brief Create a domain SubMesh from its parent.
    *
    * The SubMesh object expects the parent Mesh object to be valid for the
    * entire object lifetime. The @a domain_attributes have to mark exactly one
    * connected subset of the parent Mesh.
    *
    * @param[in] parent Parent Mesh
    * @param[in] domain_attributes Domain attributes to extract
    */
   static SubMesh CreateFromDomain(const Mesh &parent,
                                   const Array<int> &domain_attributes);

   /**
   * @brief Create a surface SubMesh from its parent.
   *
   * The SubMesh object expects the parent Mesh object to be valid for the
   * entire object lifetime. The @a boundary_attributes have to mark exactly one
   * connected subset of the parent Mesh.
   *
   * @param[in] parent Parent Mesh
   * @param[in] boundary_attributes Boundary attributes to extract

   */
   static SubMesh CreateFromBoundary(const Mesh &parent,
                                     const Array<int> &boundary_attributes);

   ///Get the parent Mesh object
   const Mesh* GetParent() const
   {
      return parent_;
   }

   /**
    * @brief Get the From indicator.
    *
    * Indicates whether the SubMesh has been created from a domain or surface.
    */
   From GetFrom() const
   {
      return from_;
   }

   /**
    * @brief Get the parent element id map.
    *
    * SubMesh element id (array index) to parent Mesh element id.
    */
   const Array<int>& GetParentElementIDMap() const
   {
      return parent_element_ids_;
   }

   /**
    * @brief Get the face id map
    *
    * SubMesh face id (array index) to parent Mesh face id.
    */
   const Array<int>& GetParentFaceIDMap() const
   {
      return parent_face_ids_;
   }

   /**
    * @brief Get the edge id map
    *
    * Submesh edge id (array index) to parent Mesh edge id.
    */
   const Array<int>& GetParentEdgeIDMap() const
   {
      return parent_edge_ids_;
   }

   /**
    * @brief Get the relative face orientations
    *
    * SubMesh element id (array index) to parent Mesh face orientation.
    */
   const Array<int>& GetParentFaceOrientations() const
   {
      return parent_face_ori_;
   }

   /**
    * @brief Get the parent vertex id map.
    *
    * SubMesh vertex id (array index) to parent Mesh vertex id.
    */
   const Array<int>& GetParentVertexIDMap() const
   {
      return parent_vertex_ids_;
   }

   /**
    * @brief Get the submesh element corresponding to a parent element. -1 ==
    * not present.
    * @param pe The parent element id.
    * @return int
    */
   int GetSubMeshElementFromParent(int pe) const
   {
      return pe == -1 ? pe : parent_to_submesh_element_ids_[pe];
   }
   /**
    * @brief Get the submesh vertex corresponding to a parent element. -1 == not
    * present.
    * @param pv The parent vertex id.
    * @return int
    */
   int GetSubMeshVertexFromParent(int pv) const
   {
      return pv == -1 ? pv : parent_to_submesh_vertex_ids_[pv];
   }
   /**
    * @brief Get the submesh edge corresponding to a parent element. -1 == not
    * present.
    * @param pe The parent edge id.
    * @return int
    */
   int GetSubMeshEdgeFromParent(int pe) const
   {
      return pe == -1 ? pe : parent_to_submesh_edge_ids_[pe];
   }
   /**
    * @brief Get the submesh face corresponding to a parent element. -1 == not
    * present.
    * @param pf The parent face id.
    * @return int
    */
   int GetSubMeshFaceFromParent(int pf) const
   {
      return pf == -1 ? pf : parent_to_submesh_face_ids_[pf];
   }

   /**
    * @brief Transfer the dofs of a GridFunction.
    *
    * The @a src GridFunction can either be defined on a Mesh or a SubMesh and
    * is transferred appropriately.
    *
    * @note Either @a src or @a dst has to be defined on a SubMesh.
    *
    * @param[in] src
    * @param[out] dst
    */
   static void Transfer(const GridFunction &src, GridFunction &dst);

   /**
    * @brief Create a Transfer Map object.
    *
    * The @a src GridFunction can either be defined on a Mesh or a SubMesh and
    * is transferred appropriately.
    *
    * @note Either @a src or @a dst has to be defined on a SubMesh.
    */
   static TransferMap CreateTransferMap(const GridFunction &src,
                                        const GridFunction &dst);

   /**
   * @brief Check if Mesh @a m is a SubMesh.
   *
   * @param m The input Mesh
   */
   static bool IsSubMesh(const Mesh *m)
   {
      return dynamic_cast<const SubMesh *>(m) != nullptr;
   }

private:
   /// Private constructor
   SubMesh(const Mesh &parent, From from, const Array<int> &attributes);

   /// The parent Mesh. Not owned.
   const Mesh *parent_;

   /// Optional nonconformal submesh. Managed via ncmesh pointer in base class.
   NCSubMesh *ncsubmesh_;

   /// Indicator from which part of the parent ParMesh the ParSubMesh is going
   /// to be created.
   From from_;

   /// Attributes on the parent ParMesh on which the ParSubMesh is created.
   /// Could either be domain or boundary attributes (determined by from_).
   Array<int> attributes_;

   /// Mapping from submesh element ids (index of the array), to the parent
   /// element ids.
   Array<int> parent_element_ids_;

   /// Mapping from submesh vertex ids (index of the array), to the parent
   /// vertex ids.
   Array<int> parent_vertex_ids_;

   /// Mapping from SubMesh edge ids (index of the array), to the parent Mesh
   /// face ids.
   Array<int> parent_edge_ids_;

   /// Mapping from SubMesh face ids (index of the array), to the parent Mesh
   /// face ids.
   Array<int> parent_face_ids_;

   /// Mapping from SubMesh face ids (index of the array), to the orientation of
   /// the face relative to the parent face.
   Array<int> parent_face_ori_;

   /// Mapping from parent Mesh vertex ids (index of the array), to the SubMesh
   /// vertex ids. Inverse map of parent_element_ids_.
   Array<int> parent_to_submesh_element_ids_;

   /// Mapping from parent Mesh vertex ids (index of the array), to the SubMesh
   /// vertex ids. Inverse map of parent_vertex_ids_.
   Array<int> parent_to_submesh_vertex_ids_;

   /// Mapping from parent Mesh edge ids (index of the array), to the SubMesh
   /// edge ids. Inverse map of parent_edge_ids_.
   Array<int> parent_to_submesh_edge_ids_;

   /// Mapping from parent Mesh face ids (index of the array), to the SubMesh
   /// face ids. Inverse map of parent_face_ids_.
   Array<int> parent_to_submesh_face_ids_;
};

} // namespace mfem

#endif
