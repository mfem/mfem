// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_SUBMESH_UTILS
#define MFEM_SUBMESH_UTILS

#include <type_traits>
#include <unordered_map>
#include "submesh.hpp"
#include "../../fem/fespace.hpp"

namespace mfem
{
namespace SubMeshUtils
{

/**
 * @brief Convenience object to create unique indices.
 */
struct UniqueIndexGenerator
{
   int counter = 0;
   std::unordered_map<int, int> idx;

   /**
    * @brief Returns the unique index from an index set.
    *
    * @param i index
    * @param new_index indicates if the index is new or was already present in
    * the set.
    */
   int Get(int i, bool &new_index);
};

/**
 * @brief Given an element @a el and a list of @a attributes, determine if that
 * element is in at least one attribute of @a attributes.
 *
 * @param el The element
 * @param attributes The attributes
 */
bool ElementHasAttribute(const Element &el, const Array<int> &attributes);

/**
 * @brief Given a Mesh @a parent and another Mesh @a mesh using the list of
 * attributes in @a attributes, this function adds matching elements with those
 * attributes from @a parent to @a mesh.
 *
 * It also returns a tuple containing first the parent vertex ids (mapping from
 * mesh vertex ids (index of the array), to the parent vertex ids) and second
 * the parent element ids (mapping from submesh element ids (index of the
 * array), to the parent element ids)
 *
 * @note Works with ParMesh.
 *
 * @param parent The Mesh where the elements are "extracted" from.
 * @param mesh The Mesh where the elements are extracted to.
 * @param attributes The attributes of the desired elements.
 * @param from_boundary Indication if the desired elements come from the
 * boundary of the parent.
 */
std::tuple< Array<int>, Array<int> >
AddElementsToMesh(const Mesh& parent,
                  Mesh& mesh, const Array<int> &attributes,
                  bool from_boundary = false);

/**
 * @brief Given two meshes that have a parent to SubMesh relationship create a
 * face map, using a SubMesh to parent Mesh element id map.
 *
 * @param parent The parent Mesh.
 * @param mesh The Mesh to match its parents faces.
 * @param parent_element_ids The Mesh element to parent element id map.
 */
Array<int> BuildFaceMap(const Mesh& parent, const Mesh& mesh,
                        const Array<int> &parent_element_ids);

/**
 * @brief Build the vdof to vdof mapping between two FiniteElementSpace objects.
 *
 * Given two FiniteElementSpace objects and the map parent_element_ids, which
 * maps the element ids of the subfes to elements on the parentfes (or boundary
 * elements depending on the type of transfer, volume or surface), create a vdof
 * to vdof map.
 *
 * This map is entirely serial and has no knowledge about parallel groups.
 *
 * @param[in] subfes
 * @param[in] parentfes
 * @param[in] from
 * @param[in] parent_element_ids
 * @param[out] vdof_to_vdof_map
 */
void BuildVdofToVdofMap(const FiniteElementSpace& subfes,
                        const FiniteElementSpace& parentfes,
                        const SubMesh::From& from,
                        const Array<int>& parent_element_ids,
                        Array<int>& vdof_to_vdof_map);

/**
 * @brief Identify the root parent of a given SubMesh.
 *
 * @tparam T The type of the input object which has to fulfill the
 * SubMesh::GetParent() interface.
 */
template <class T, class RT = decltype(std::declval<T>().GetParent())>
RT GetRootParent(const T &m)
{
   RT parent = m.GetParent();
   while (true)
   {
      const T* next = dynamic_cast<const T*>(parent);
      if (next == nullptr) { return parent; }
      else { parent = next->GetParent(); }
   }
}

} // namespace SubMeshUtils
} // namespace mfem

#endif
