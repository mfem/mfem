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

// /**
//  * @brief Given an element @a el and a list of @a attributes, determine if that
//  * element is in at least one attribute of @a attributes.
//  *
//  * @param el The element
//  * @param attributes The attributes
//  */
// bool ElementHasAttribute(const Element &el, const Array<int> &attributes);

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
 * @brief Add boundary elements to the Mesh.
 * @details An attempt to call this function for anything other than SubMesh or ParSubMesh
 * will result in a linker error as the template is only explicitly instantiated for those
 * types.
 * @param mesh The Mesh to add boundary elements to.
 * @tparam MeshT The SubMesh type, options SubMesh and ParSubMesh.
 */
template <typename SubMeshT>
void AddBoundaryElements(SubMeshT &mesh, const std::unordered_map<int,int> &lface_to_boundary_attribute = {});


template<typename MeshT, typename SubMeshT>
void ConstructFaceTree(const MeshT &parent, SubMeshT &submesh);

/**
 * @brief Helper for checking if an object's attributes match a list
 *
 * @tparam T Object Type
 * @param el Instance of T
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
 * @brief Apply permutation to a container type
 *
 * @tparam T1 Container type
 * @param indices Set of indices that define the permutation
 * @param t1 Array to be permuted
 */
template <typename T1>
void Permute(const Array<int>& indices, T1& t1)
{
   Permute(Array<int>(indices), t1);
}

/**
 * @brief Apply permutation to a container type
 * @details Sorts the indices variable in the process, thereby destroying the permutation.
 *
 * @tparam T1 Container type
 * @param indices Set of indices that define the permutation
 * @param t1 Array to be permuted
 */
template <typename T1>
void Permute(Array<int>&& indices, T1& t1)
{
   for (int i = 0; i < indices.Size(); i++)
   {
      auto current = i;
      while (i != indices[current])
      {
         auto next = indices[current];
         std::swap(t1[current], t1[next]);
         indices[current] = current;
         current = next;
      }
      indices[current] = current;
   }
}

/**
 * @brief Apply permutation to a container type
 *
 * @tparam T1 Container type 1
 * @tparam T2 Container type 2
 * @tparam T3 Container type 3
 * @param indices Set of indices that define the permutation
 * @param t1 First array to be permuted
 * @param t2 Second array to be permuted
 * @param t3 Third array to be permuted
 */
template <typename T1, typename T2, typename T3>
void Permute(const Array<int>& indices, T1& t1, T2& t2, T3& t3)
{
   Permute(Array<int>(indices), t1, t2, t3);
}

/**
 * @brief Apply permutation to a container type
 * @details Sorts the indices variable in the process, thereby destroying the permutation.
 *
 * @tparam T1 Container type 1
 * @tparam T2 Container type 2
 * @tparam T3 Container type 3
 * @param indices Set of indices that define the permutation
 * @param t1 First array to be permuted
 * @param t2 Second array to be permuted
 * @param t3 Third array to be permuted
 */
template <typename T1, typename T2, typename T3>
void Permute(Array<int>&& indices, T1& t1, T2& t2, T3& t3)
{
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

/**
 * @brief Reorder a container of nodes based on orientation and geometry.
 *
 * @tparam FaceNodes Type of the container of nodes
 * @param nodes Instance to be reordered
 * @param geom Geometry defining the face
 * @param orientation Orientation of the face
 */
template <typename FaceNodes>
void ReorientFaceNodesByOrientation(FaceNodes &nodes, Geometry::Type geom, int orientation)
{
   auto permute = [&]() -> std::array<int, NCMesh::MaxFaceNodes>
   {
      if (geom == Geometry::Type::SEGMENT)
      {
         switch (orientation) // degenerate (0,0,1,1)
         {
            case 0: return {0,1,2,3};
            case 1: return {2,3,0,1};
            default: MFEM_ABORT("Unexpected orientation!");
         }
      }
      else if (geom == Geometry::Type::TRIANGLE)
      {
         switch (orientation)
         {
            case 0: return {0,1,2,3};
            case 5: return {0,2,1,3};
            case 2: return {1,2,0,3};
            case 1: return {1,0,2,3};
            case 4: return {2,0,1,3};
            case 3: return {2,1,0,3};
            default: MFEM_ABORT("Unexpected orientation!");
         }
      }
      else if (geom == Geometry::Type::SQUARE)
      {
         switch (orientation)
         {
            case 0: return {0,1,2,3};
            case 1: return {0,3,2,1};
            case 2: return {1,2,3,0};
            case 3: return {1,0,3,2};
            case 4: return {2,3,0,1};
            case 5: return {2,1,0,3};
            case 6: return {3,0,1,2};
            case 7: return {3,2,1,0};
            default: MFEM_ABORT("Unexpected orientation!");
         }
      }
      else { MFEM_ABORT("Unexpected face geometry!"); }
   }();
   Permute(Array<int>(permute.data(), NCMesh::MaxFaceNodes), nodes);
}

} // namespace SubMeshUtils
} // namespace mfem

#endif
