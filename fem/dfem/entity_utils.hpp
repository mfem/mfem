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

namespace detail
{

/// @brief Restrictions and E-vectors of the integrators of one entity type,
/// and the steps that move data between them and the operator's L-vectors.
///
/// The input E-vectors hold the restricted input fields, the output E-vectors
/// the residual (or derivative action) and the input result E-vectors the
/// result of a transpose action, which lives in the input fields. Operators
/// only use the steps below, so that an entity type which needs more than a
/// restriction to move its data can provide it here.
class EntityEVectors
{
public:
   EntityEVectors(const RestrictionDescriptor &rest_descriptor,
                  size_t num_inputs, size_t num_outputs) :
      in_e(num_inputs, nullptr),
      out_e(num_outputs, nullptr),
      in_result_e(num_inputs, nullptr),
      in_rcache(rest_descriptor),
      out_rcache(rest_descriptor) { }

   /// Restrict the L-vectors @a x_l of the fields @a infds to in_e.
   void RestrictInputs(const std::vector<FieldDescriptor> &infds,
                       const std::vector<Vector *> &x_l)
   {
      restriction(infds, in_rcache, x_l, in_e);
   }

   /// Size out_e for the fields @a outfds and zero it.
   void ZeroOutputs(const std::vector<FieldDescriptor> &outfds)
   {
      prepare_residual(outfds, out_rcache, out_e);
      for (auto *v : out_e) { *v = 0.0; }
   }

   /// Size in_result_e for the fields @a infds and zero it.
   void ZeroInputResults(const std::vector<FieldDescriptor> &infds)
   {
      prepare_residual(infds, in_rcache, in_result_e);
      for (auto *v : in_result_e) { *v = 0.0; }
   }

   /// Apply the transposed restrictions of the fields @a outfds to out_e and
   /// write the result to @a y, a std::vector<Vector *> or a MultiVector.
   /// Writing may alias the entries of a std::vector to out_e, see
   /// restriction_transpose().
   template <typename lvectors_t>
   void TransposeOutputs(const std::vector<FieldDescriptor> &outfds,
                         lvectors_t &y)
   {
      restriction_transpose(outfds, out_rcache, out_e, y);
   }

   /// Like TransposeOutputs(), for in_result_e and the fields @a infds.
   template <typename lvectors_t>
   void TransposeInputResults(const std::vector<FieldDescriptor> &infds,
                              lvectors_t &y)
   {
      restriction_transpose(infds, in_rcache, in_result_e, y);
   }

   std::vector<Vector *> in_e;
   std::vector<Vector *> out_e;
   std::vector<Vector *> in_result_e;

private:
   RestrictionCache in_rcache;
   RestrictionCache out_rcache;
};

} // namespace detail

} // namespace mfem::future

#endif // MFEM_USE_MPI
