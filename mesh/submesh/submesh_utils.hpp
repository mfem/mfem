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

#ifndef MFEM_SUBMESH_UTILS
#define MFEM_SUBMESH_UTILS

#include <type_traits>
#include <unordered_map>
#include "submesh.hpp"
#include "../../fem/fespace.hpp"

namespace mfem
{
class NCSubMesh;
class ParNCSubMesh;

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
template <class T>
auto GetRootParent(const T &m) -> decltype(std::declval<T>().GetParent())
{
   auto parent = m.GetParent();
   while (true)
   {
      const T* next = dynamic_cast<const T*>(parent);
      if (next == nullptr) { return parent; }
      else { parent = next->GetParent(); }
   }
}

/**
 * @brief Add boundary elements to the SubMesh.
 * @details An attempt to call this function for anything other than SubMesh or
 * ParSubMesh will result in a linker error as the template is only explicitly
 * instantiated for those types.
 * @param mesh The SubMesh to add boundary elements to.
 * @param lface_to_boundary_attribute Map from local faces in the submesh to
 * boundary attributes. Only necessary for interior boundary attributes of
 * volume submeshes, where the face owning the attribute might be on a
 * neighboring rank.
 * @tparam SubMeshT The SubMesh type, options SubMesh and ParSubMesh.
 */
template <typename SubMeshT>
void AddBoundaryElements(SubMeshT &mesh,
                         const std::unordered_map<int,int> &lface_to_boundary_attribute = {});

/**
 * @brief Construct a nonconformal mesh (serial or parallel) for a surface
 * submesh, from an existing nonconformal volume mesh (serial or parallel).
 * @details This function is only instantiated for NCSubMesh and ParNCSubMesh
 *    Attempting to use it with other classes will result in a linker error.
 * @tparam NCSubMeshT The NCSubMesh type
 * @param[out] submesh The surface submesh to be filled.
 * @param attributes The set of attributes defining the submesh.
 */
template<typename NCSubMeshT>
void ConstructFaceTree(NCSubMeshT &submesh, const Array<int> &attributes);

/**
 * @brief Construct a nonconformal mesh (serial or parallel) for a volume
 * submesh, from an existing nonconformal volume mesh (serial or parallel).
 * @details This function is only instantiated for NCSubMesh and ParNCSubMesh
 *    Attempting to use it with other classes will result in a linker error.
 * @tparam NCSubMeshT The NCSubMesh type
 * @param[out] submesh The volume submesh to be filled from parent.
 * @param attributes The set of attributes defining the submesh.
 */
template <typename NCSubMeshT>
void ConstructVolumeTree(NCSubMeshT &submesh, const Array<int> &attributes);

/**
 * @brief Helper for checking if an object's attributes match a list
 *
 * @tparam T Object Type
 * @param el Instance of T, requires method `GetAttribute()`
 * @param attributes Set of attributes to match against
 * @return true The attribute of el is contained within attributes
 * @return false
 */
template <typename T>
bool HasAttribute(const T &el, const Array<int> &attributes)
{
   for (int a = 0; a < attributes.Size(); a++)
   {
      if (el.GetAttribute() == attributes[a])
      {
         return true;
      }
   }
   return false;
}

/**
 * @brief Forwarding dispatch to HasAttribute for backwards compatability
 *
 * @param el Instance of T, requires method `GetAttribute()`
 * @param attributes Set of attributes to match against
 * @return true The attribute of el is contained within attributes
 * @return false
 */
MFEM_DEPRECATED inline bool ElementHasAttribute(const Element &el,
                                                const Array<int> &attributes)
{
   return HasAttribute(el,attributes);
}

/**
 * @brief Apply permutation to a container type
 *
 * @tparam T1 Container type 1
 * @tparam T2 Container type 2
 * @tparam T3 Container type 3
 * @param indices Set of indices that define the permutation
 * @param t1 First collection to be permuted
 * @param t2 Second collection to be permuted
 * @param t3 Third collection to be permuted
 */
template <typename T1, typename T2, typename T3>
void Permute(const Array<int>& indices, T1& t1, T2& t2, T3& t3)
{
   Permute(Array<int>(indices), t1, t2, t3);
}

/**
 * @brief Apply permutation to a container type
 * @details Sorts the indices variable in the process, thereby destroying the
 * permutation.
 *
 * @tparam T1 Container type 1
 * @tparam T2 Container type 2
 * @tparam T3 Container type 3
 * @param indices Set of indices that define the permutation
 * @param t1 First collection to be permuted
 * @param t2 Second collection to be permuted
 * @param t3 Third collection to be permuted
 */
template <typename T1, typename T2, typename T3>
void Permute(Array<int>&& indices, T1& t1, T2& t2, T3& t3)
{
   /*
   TODO: In c++17 can replace this with a parameter pack expansion technique to
   operate on arbitrary collections of reference accessible containers of
   arbitrary type.
   template <typename ...T> void Permute(Array<int>&&indices, T&... t)
   {
      for (int i = 0; i < indices.Size(); i++)
      {
         auto current = i;
         while (i != indices[current])
         {
               auto next = indices[current];
               // Lambda allows iteration over expansion in c++17
               // https://stackoverflow.com/a/60136761
               ([&]{std::swap(t[current], t[next]);} (), ...);
               current = next;
         }
         indices[current] = current;
      }
   }
   */

   for (int i = 0; i < indices.Size(); i++)
   {
      auto current = i;
      while (i != indices[current])
      {
         auto next = indices[current];
         std::swap(t1[current], t1[next]);
         std::swap(t2[current], t2[next]);
         std::swap(t3[current], t3[next]);
         indices[current] = current;
         current = next;
      }
      indices[current] = current;
   }
}


} // namespace SubMeshUtils
} // namespace mfem

#endif
