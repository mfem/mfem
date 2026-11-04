// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
#pragma once

#include "../../config/config.hpp" // IWYU pragma: keep

#ifdef MFEM_USE_MPI

#include "util.hpp"
#include "field_restriction.hpp"

namespace mfem::future
{

/// @brief Entity struct
///
/// This struct is used to store information about an entity type.
namespace Entity
{
struct Element;
struct BoundaryElement;
struct Face;
struct BoundaryFace;
}

/// @brief Properties of the entity type @a entity_t (see Entity).
///
/// This is the one place that defines what an entity type means: how many
/// there are, their reference dimension, where their attributes come from,
/// how fields are restricted to them and which finite element describes a
/// field on them.
template <typename entity_t>
struct EntityTraits
{
   static_assert(dfem::always_false<entity_t>, "entity type not supported");
};


/// Volume elements use the element restriction; face settings are unused.
template <>
struct EntityTraits<Entity::Element>
{
   static constexpr RestrictionDescriptor restriction
   {
      false, FaceType::Interior, L2FaceValues::SingleValued
   };

   static int Count(const Mesh &mesh) { return mesh.GetNE(); }

   static int Dimension(const Mesh &mesh) { return mesh.Dimension(); }

   static const Array<int> &Attributes(const Mesh &mesh)
   {
      return mesh.GetElementAttributes();
   }

   static const FiniteElement *TypicalFE(const FiniteElementSpace &fes)
   {
      return fes.GetTypicalFE();
   }
};

/// Boundary elements use a single-sided restriction to boundary faces and
/// the trace finite element on a reference entity of dimension mesh.Dimension() - 1.
template <>
struct EntityTraits<Entity::BoundaryElement>
{
   static constexpr RestrictionDescriptor restriction
   {
      true, FaceType::Boundary, L2FaceValues::SingleValued
   };

   static int Count(const Mesh &mesh)
   {
      return mesh.GetNFbyType(FaceType::Boundary);
   }

   static int Dimension(const Mesh &mesh) { return mesh.Dimension() - 1; }

   static const Array<int> &Attributes(const Mesh &mesh)
   {
      return mesh.GetBdrFaceAttributes();
   }

   static const FiniteElement *TypicalFE(const FiniteElementSpace &fes)
   {
      return fes.GetTypicalTraceElement();
   }
};

/// @brief Get the reference dimension of the entities of a given type.
///
/// @param mesh the mesh.
/// @tparam entity_t the entity type (see Entity).
/// @returns the reference dimension of the given entity type, i.e. the mesh
/// dimension for elements and one less for boundary elements.
template <typename entity_t>
int GetEntityDimension(const mfem::Mesh &mesh)
{
   return EntityTraits<entity_t>::Dimension(mesh);
}

/// @brief Get the number of entities of a given type.
///
/// @param mesh the mesh.
/// @tparam entity_t the entity type (see Entity).
/// @returns the number of entities of the given type.
template <typename entity_t>
int GetNumEntities(const mfem::Mesh &mesh)
{
   return EntityTraits<entity_t>::Count(mesh);
}

} // namespace mfem::future

#endif // MFEM_USE_MPI
