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
///
/// The E-vector arrays have one slot per field of the operator, but only the
/// slots of the used fields are filled; the others stay nullptr.
class EntityEVectors
{
public:
   EntityEVectors(const RestrictionDescriptor &rest_descriptor,
                  size_t num_inputs, size_t num_outputs) :
      in_e(num_inputs, nullptr),
      out_e(num_outputs, nullptr),
      in_result_e(num_inputs, nullptr),
      used_inputs(num_inputs, false),
      used_outputs(num_outputs, false),
      in_rcache(rest_descriptor),
      out_rcache(rest_descriptor) { }

   /// @brief The E-vectors of a derivative of the operator whose E-vectors of
   /// the same entity type are @a parent. The derivative has the same inputs,
   /// so it takes the used inputs and the input restrictions of @a parent. If
   /// @a copy_outputs, i.e. if both have the same output fields, it takes the
   /// outputs the same way; otherwise it has its own @a num_outputs outputs
   /// (the derivatives of a functional), all marked used. The E-vectors are
   /// its own, allocated on first use.
   EntityEVectors(const EntityEVectors &parent, size_t num_outputs,
                  bool copy_outputs) :
      in_e(parent.in_e.size(), nullptr),
      out_e(num_outputs, nullptr),
      in_result_e(parent.in_result_e.size(), nullptr),
      used_inputs(parent.used_inputs),
      used_outputs(copy_outputs ? parent.used_outputs :
                   std::vector<bool>(num_outputs, true)),
      in_rcache(parent.in_rcache),
      out_rcache(copy_outputs ? parent.out_rcache :
                 RestrictionCache(parent.out_rcache.Descriptor()))
   {
      MFEM_ASSERT(!copy_outputs || parent.out_e.size() == num_outputs,
                  "internal error");
   }

   /// The E-vectors are owned raw pointers, so an EntityEVectors is never
   /// copied or moved. It is built in place (MakeEntityIntegrators() and
   /// MakeDerivativeIntegrators() return it as a prvalue); a derivative gets
   /// its own through the constructor above.
   /// @note When a deep copy of DifferentiableOperator is implemented, this
   /// needs a deep copy constructor as well (new E-vectors with copied values).
   EntityEVectors(const EntityEVectors &) = delete;
   EntityEVectors &operator=(const EntityEVectors &) = delete;
   EntityEVectors(EntityEVectors &&) = delete;
   EntityEVectors &operator=(EntityEVectors &&) = delete;

   /// Mark the fields used by an integrator, given by the maps from its
   /// inputs and outputs to the fields of the operator, as used. The used
   /// fields are the union over all integrators of the entity type.
   template <size_t ni, size_t no>
   void MarkUsed(const std::array<size_t, ni> &input_to_field,
                 const std::array<size_t, no> &output_to_field)
   {
      for (size_t idx : input_to_field)
      {
         if (idx != SIZE_MAX) { used_inputs[idx] = true; }
      }
      for (size_t idx : output_to_field)
      {
         if (idx != SIZE_MAX) { used_outputs[idx] = true; }
      }
   }

   /// Restrict the L-vectors @a x_l of the fields @a infds to in_e.
   void RestrictInputs(const std::vector<FieldDescriptor> &infds,
                       const std::vector<Vector *> &x_l)
   {
      restriction(infds, in_rcache, x_l, in_e, used_inputs);
   }

   /// Size out_e for the fields @a outfds and zero it.
   void ZeroOutputs(const std::vector<FieldDescriptor> &outfds)
   {
      prepare_residual(outfds, out_rcache, out_e, used_outputs);
      for (size_t i = 0; i < out_e.size(); i++)
      {
         if (used_outputs[i]) { *out_e[i] = 0.0; }
      }
   }

   /// Size in_result_e for the fields @a infds and zero it.
   void ZeroInputResults(const std::vector<FieldDescriptor> &infds)
   {
      prepare_residual(infds, in_rcache, in_result_e, used_inputs);
      for (size_t i = 0; i < in_result_e.size(); i++)
      {
         if (used_inputs[i]) { *in_result_e[i] = 0.0; }
      }
   }

   /// Apply the transposed restrictions of the used fields of @a outfds to
   /// out_e. A field with !written[i] is overwritten in @a y, otherwise the
   /// result is added to @a y; written[i] is then set. Fields that are not
   /// used are left alone, see ZeroUnwritten() for the ones nobody wrote.
   /// @a y is a std::vector<Vector *> or a MultiVector. Writing may alias the
   /// entries of a std::vector to out_e, see restriction_transpose().
   template <typename lvectors_t>
   void TransposeOutputs(const std::vector<FieldDescriptor> &outfds,
                         lvectors_t &y, std::vector<bool> &written)
   {
      Transpose(outfds, out_rcache, out_e, y, used_outputs, written);
   }

   /// Like TransposeOutputs(), for in_result_e and the fields @a infds.
   template <typename lvectors_t>
   void TransposeInputResults(const std::vector<FieldDescriptor> &infds,
                              lvectors_t &y, std::vector<bool> &written)
   {
      Transpose(infds, in_rcache, in_result_e, y, used_inputs, written);
   }

   /// Delete the owned E-vectors that are not in @a deleted yet, see
   /// DeleteOwnedVectorPointers().
   void DeleteOwned(std::vector<Vector *> &deleted)
   {
      DeleteOwnedVectorPointersImpl(deleted, in_e);
      DeleteOwnedVectorPointersImpl(deleted, out_e);
      DeleteOwnedVectorPointersImpl(deleted, in_result_e);
   }

   std::vector<Vector *> in_e;
   std::vector<Vector *> out_e;
   std::vector<Vector *> in_result_e;

   /// The input and output fields of the operator that the integrators of this
   /// entity type read or write, indexed like infds and outfds. Fields that
   /// are not used are never resolved, restricted, sized, zeroed or
   /// transposed. Unrelated to the derivative activity maps of the backends.
   std::vector<bool> used_inputs;
   std::vector<bool> used_outputs;

private:
   template <typename lvectors_t>
   void Transpose(const std::vector<FieldDescriptor> &fields,
                  RestrictionCache &cache,
                  const std::vector<Vector *> &x_e,
                  lvectors_t &y, const std::vector<bool> &used,
                  std::vector<bool> &written)
   {
      MFEM_ASSERT(written.size() == fields.size(),
                  "written mask should match the number of fields");

      // No-op after first call. It should be  already set up by 
      // ZeroOutputs() or ZeroInputResults().
      cache.EnsureSetup(fields, used);

      // Split the used fields: the ones nobody wrote yet are overwritten,
      // the others are added to. Mark them all as written.
      overwrite.assign(fields.size(), false);
      add.assign(fields.size(), false);
      for (size_t i = 0; i < fields.size(); i++)
      {
         overwrite[i] = used[i] && !written[i];
         add[i] = used[i] && written[i];
         if (used[i]) { written[i] = true; }
      }

      // Transpose each field with its mode.
      restriction_transpose(fields, cache, x_e, y, overwrite);
      restriction_transpose_add(fields, cache, x_e, y, add);
   }

   RestrictionCache in_rcache;
   RestrictionCache out_rcache;

   /// The fields Transpose() overwrites and adds to, kept as members so that
   /// they are not allocated on every call.
   std::vector<bool> overwrite;
   std::vector<bool> add;
};

/// @brief The callbacks and E-vectors of the integrators of one entity type.
template <typename callbacks_t>
struct EntityIntegrators
{
   EntityEVectors ev;
   callbacks_t callbacks;
};

/// @brief The EntityIntegrators of the entity type @a entity_t, with
/// E-vectors for @a num_inputs inputs and @a num_outputs outputs restricted
/// as EntityTraits<entity_t>::restriction describes.
template <typename entity_t, typename callbacks_t>
EntityIntegrators<callbacks_t> MakeEntityIntegrators(size_t num_inputs,
                                                     size_t num_outputs,
                                                     callbacks_t callbacks = {})
{
   return
   {
      EntityEVectors(EntityTraits<entity_t>::restriction, num_inputs,
                     num_outputs),
      std::move(callbacks)
   };
}

/// @brief Size the L-vectors @a y of the @a fields that no entity type wrote
/// (see EntityEVectors::TransposeOutputs()) and set them to zero.
///
/// @a y is a std::vector<Vector *> or a MultiVector.
template <typename lvectors_t>
void ZeroUnwritten(const std::vector<FieldDescriptor> &fields,
                   const std::vector<bool> &written, lvectors_t &y)
{
   for (size_t i = 0; i < fields.size(); i++)
   {
      if (written[i]) { continue; }
      const int s = GetVSize(fields[i]);
      if constexpr (std::is_same_v<lvectors_t, MultiVector>)
      {
         y[i].SetSize(s);
         y[i] = 0.0;
      }
      else
      {
         if (y[i] == nullptr)
         {
            y[i] = new Vector(s);
            y[i]->UseDevice(true);
         }
         else
         {
            y[i]->SetSize(s);
         }
         *y[i] = 0.0;
      }
   }
}

/// @brief Forget the entries of @a y that alias an E-vector of any of the
/// EntityIntegrators @a kinds.
///
/// An overwriting transpose (restriction_transpose()) aliases L-vectors of
/// pass-through fields to the E-vectors of the entity type that writes
/// first, and deletes an entry that does not alias its own E-vector. When
/// another entity type writes first later on, the entry must not be
/// deleted or written through. The E-vectors stay owned by their
/// EntityEVectors.
template <typename kinds_t>
void ForgetEVectorAliases(const kinds_t &kinds, std::vector<Vector *> &y)
{
   for (auto *&v : y)
   {
      for (const auto *k : kinds)
      {
         for (const auto *e_vectors : { &k->ev.out_e, &k->ev.in_result_e })
         {
            if (std::find(e_vectors->begin(), e_vectors->end(), v) !=
                e_vectors->end()) { v = nullptr; }
         }
      }
   }
}

/// @brief Write to @a y the sum of the transposed output E-vectors of the
/// EntityIntegrators @a kinds that have callbacks in @a member. The first
/// entity type to use a field overwrites it, the others add to it, and the
/// fields no entity type uses are set to zero. @a y is a
/// std::vector<Vector *> or a MultiVector.
template <typename kinds_t, typename member_t, typename lvectors_t>
void TransposeOutputs(const kinds_t &kinds, member_t member,
                      const std::vector<FieldDescriptor> &outfds,
                      lvectors_t &y)
{
   // Drop aliases of E-vectors left over from the previous call, so that
   // the overwriting transpose below never deletes or writes through them.
   if constexpr (std::is_same_v<lvectors_t, std::vector<Vector *>>)
   {
      ForgetEVectorAliases(kinds, y);
   }

   // Sum the entity types: the first one to use a field overwrites it,
   // the others add to it.
   std::vector<bool> written(outfds.size(), false);
   for (auto *k : kinds)
   {
      if ((k->callbacks.*member).empty()) { continue; }
      k->ev.TransposeOutputs(outfds, y, written);
   }

   // Zero the fields no entity type wrote.
   ZeroUnwritten(outfds, written, y);
}

/// @brief Like TransposeOutputs(), for the input result E-vectors and the
/// fields @a infds.
template <typename kinds_t, typename member_t, typename lvectors_t>
void TransposeInputResults(const kinds_t &kinds, member_t member,
                           const std::vector<FieldDescriptor> &infds,
                           lvectors_t &y)
{
   // Drop aliases of E-vectors left over from the previous call, so that
   // the overwriting transpose below never deletes or writes through them.
   if constexpr (std::is_same_v<lvectors_t, std::vector<Vector *>>)
   {
      ForgetEVectorAliases(kinds, y);
   }

   // Sum the entity types: the first one to use a field overwrites it,
   // the others add to it.
   std::vector<bool> written(infds.size(), false);
   for (auto *k : kinds)
   {
      if ((k->callbacks.*member).empty()) { continue; }
      k->ev.TransposeInputResults(infds, y, written);
   }

   // Zero the fields no entity type wrote.
   ZeroUnwritten(infds, written, y);
}

/// @brief Delete the owned E-vectors of the EntityIntegrators @a kinds
/// together with the @a other vector groups, which may alias them, see
/// DeleteOwnedVectorPointers().
template <typename kinds_t, typename... groups_t>
void DeleteOwnedEVectors(const kinds_t &kinds, groups_t &... other)
{
   std::vector<Vector *> deleted;
   (DeleteOwnedVectorPointersImpl(deleted, other), ...);
   for (auto *k : kinds) { k->ev.DeleteOwned(deleted); }
}

} // namespace detail

} // namespace mfem::future

#endif // MFEM_USE_MPI
