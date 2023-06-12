// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
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
#include <unordered_map>

namespace mfem
{

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
public:
   /// Indicator from which part of the parent Mesh the SubMesh is created.
   enum From
   {
      Domain,
      Boundary
   };

   static const int GENERATED_ATTRIBUTE = 900;

   SubMesh() = delete;

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
                                   Array<int> domain_attributes);

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
                                     Array<int> boundary_attributes);

   /**
    * @brief Get the parent Mesh object
    *
    */
   const Mesh* GetParent() const
   {
      return &parent_;
   }

   /**
    * @brief Get the From indicator.
    *
    * Indicates whether the SubMesh has been created from a domain or
    * surface.
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
    * SubMesh element id (array index) to parent Mesh face id.
    */
   const Array<int>& GetParentFaceIDMap() const
   {
      return parent_face_ids_;
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
    * The @a src GridFunction can either be defined on a Mesh or a
    * SubMesh and is transferred appropriately.
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

   ~SubMesh();

private:
   /// Private constructor
   SubMesh(const Mesh &parent, From from, Array<int> attributes);

   /// The parent Mesh
   const Mesh &parent_;

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

   /// Mapping from SubMesh face ids (index of the array), to the parent Mesh
   /// face ids.
   Array<int> parent_face_ids_;

   Array<int> face_to_be;
};

} // namespace mfem

#endif
