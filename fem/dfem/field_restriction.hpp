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

namespace mfem::future
{

/// @brief How the dofs of a field are restricted to an entity kind.
///
/// Elements use the element restriction. Face kinds use the face restriction
/// of @a face_type, with one (SingleValued) or both (DoubleValued) sides of
/// every face.
struct RestrictionDescriptor
{
   bool is_face_restriction;
   FaceType face_type;
   L2FaceValues face_values;
};

/// @brief Get the element restriction operator for a field descriptor.
///
/// @param f the field descriptor.
/// @param o the element dof ordering.
/// @returns the element restriction operator for the field descriptor in
/// specified ordering.
inline
std::shared_ptr<const Operator> get_element_restriction(
   const FieldDescriptor &f,
   ElementDofOrdering o)
{
   return std::visit([&o](auto&& arg) -> std::shared_ptr<const Operator>
   {
      using T = std::decay_t<decltype(arg)>;
      if constexpr (std::is_same_v<T, const FiniteElementSpace *>
                    || std::is_same_v<T, const ParFiniteElementSpace *>)
      {
         // Non-owning shared_ptr with no-op deleter
         return std::shared_ptr<const Operator>(arg->GetElementRestriction(o), [](
         const Operator*) {});
      }
      else if constexpr (std::is_same_v<T, const VectorQuadratureSpace *>)
      {
         // For VectorQuadratureSpace, create an identity operator
         // Data is already at quadrature points, so restriction is identity
         const int size = arg->GetVSize();
         return std::make_shared<IdentityOperator>(size);
      }
      else if constexpr (std::is_same_v<T, const ParameterSpace *>)
      {
         // Non-owning shared_ptr with no-op deleter
         return std::shared_ptr<const Operator>(arg->GetElementRestriction(o), [](
         const Operator*) {});
      }
      else
      {
         static_assert(dfem::always_false<T>,
                       "can't use get_element_restriction on type");
      }
      return nullptr; // Unreachable, but avoids compiler warning
   }, f.data);
}

/// @brief Get the face restriction operator for a field descriptor.
///
/// @param f the field descriptor.
/// @param o the face dof ordering.
/// @param ft the face type
/// @param m indicator if single or double valued
/// @returns the face restriction operator for the field descriptor in
/// specified ordering.
inline
std::shared_ptr<const Operator> get_face_restriction(const FieldDescriptor &f,
                                                     ElementDofOrdering o,
                                                     FaceType ft,
                                                     L2FaceValues m)
{
   return std::visit([&o, &ft, &m](auto&& arg) -> std::shared_ptr<const Operator>
   {
      using T = std::decay_t<decltype(arg)>;
      if constexpr (std::is_same_v<T, const FiniteElementSpace *> ||
                    std::is_same_v<T, const ParFiniteElementSpace *>)
      {
         // Non-owning shared_ptr with no-op deleter
         return std::shared_ptr<const Operator>(arg->GetFaceRestriction(o, ft,
         m), [](const Operator*) {});
      }
      else if constexpr (std::is_same_v<T, const VectorQuadratureSpace *>)
      {
         // VectorQuadratureSpace does not support face restrictions
         MFEM_ABORT("internal error");
      }
      else if constexpr (std::is_same_v<T, const ParameterSpace *>)
      {
         // ParameterSpace does not support face restrictions
         MFEM_ABORT("internal error");
      }
      else
      {
         static_assert(dfem::always_false<T>,
                       "can't use get_face_restriction on type");
      }
      return nullptr; // Unreachable, but avoids compiler warning
   }, f.data);
}

/// @brief Get the restriction operator for a field descriptor.
///
/// @param f the field descriptor.
/// @param o the element dof ordering.
/// @param r how the field is restricted, see RestrictionDescriptor.
/// @returns the restriction operator for the field descriptor in
/// specified ordering.
inline std::shared_ptr<const Operator> get_restriction(const FieldDescriptor &f,
                                                       const ElementDofOrdering &o,
                                                       const RestrictionDescriptor &r)
{
   // Both interior and boundary faces use the face restriction.
   if (r.is_face_restriction)
   {
      return get_face_restriction(f, o, r.face_type, r.face_values);
   }
   return get_element_restriction(f, o);
}

/// @brief Restriction operators for a fixed list of fields, resolved once.
///
/// get_restriction() runs a std::visit over the descriptor variant and wraps
/// its result in a std::shared_ptr with a no-op deleter, which allocates a
/// control block on every call. Callers additionally dynamic_cast the result to
/// detect identity restrictions. Both are properties that are fixed for the
/// lifetime of the owning operator, so resolve them once and index into the
/// result on the hot path instead.
///
/// A cache instance is bound to the field list it was first set up with; the
/// owning operator should keep one instance per (field list, entity type) pair.
class RestrictionCache
{
public:
   /// @brief A cache for restrictions described by @a rest_descriptor, see
   /// RestrictionDescriptor and EntityTraits.
   explicit RestrictionCache(const RestrictionDescriptor &rest_descriptor) :
      rest_descriptor(rest_descriptor) { }

   /// @brief Resolve the restrictions for @a fields. A no-op after the first
   /// call.
   void EnsureSetup(const std::vector<FieldDescriptor> &fields,
                    ElementDofOrdering o = ElementDofOrdering::LEXICOGRAPHIC)
   {
      if (ready)
      {
         MFEM_ASSERT(entries.size() == fields.size() && ordering == o,
                     "restriction cache reused with a different field list");
         return;
      }
      entries.reserve(fields.size());
      for (const auto &f : fields) { entries.push_back(Resolve(f, o)); }
      ordering = o;
      ready = true;
   }

   /// @brief Resolve the restriction for the single field @a f. A no-op after
   /// the first call.
   void EnsureSetup(const FieldDescriptor &f,
                    ElementDofOrdering o = ElementDofOrdering::LEXICOGRAPHIC)
   {
      if (ready)
      {
         MFEM_ASSERT(entries.size() == 1 && ordering == o,
                     "restriction cache reused with a different field");
         return;
      }
      entries.push_back(Resolve(f, o));
      ordering = o;
      ready = true;
   }

   /// @brief The restriction operator of field @a i, or nullptr if the field
   /// has none (in which case the identity is assumed).
   const Operator *Get(std::size_t i = 0) const { return entries[i].op; }

   /// @brief True if applying the restriction of field @a i is a no-op, i.e.
   /// the field has no restriction or an identity one. Such fields alias the
   /// L-vector memory instead of copying through the operator.
   bool IsPassthrough(std::size_t i = 0) const
   {
      return entries[i].op == nullptr || entries[i].is_identity;
   }

   /// @brief The height of the restriction of field @a i, or -1 if it has none.
   int Height(std::size_t i = 0) const { return entries[i].height; }

   /// @brief The width of the restriction of field @a i, or -1 if it has none.
   int Width(std::size_t i = 0) const { return entries[i].width; }

   /// @brief The restriction of field @a i as a FaceRestriction, or nullptr
   /// if it is not one. Face restrictions also provide normal derivatives.
   const FaceRestriction *GetFaceRestriction(std::size_t i = 0) const
   {
      return entries[i].face;
   }

private:
   struct Entry
   {
      const Operator *op = nullptr;
      const FaceRestriction *face = nullptr;
      bool is_identity = false;
      int height = -1;
      int width = -1;
   };

   Entry Resolve(const FieldDescriptor &f, ElementDofOrdering o)
   {
      Entry e;
      if (std::holds_alternative<const VectorQuadratureSpace *>(f.data))
      {
         // Data already lives at quadrature points, so the restriction is the
         // identity regardless of the entity type. Resolving this through
         // get_restriction() would abort for boundary entities, and would
         // allocate a fresh IdentityOperator per call for element entities.
         const int size = std::get<const VectorQuadratureSpace *>(f.data)->GetVSize();
         owned.push_back(std::make_shared<const IdentityOperator>(size));
         e.op = owned.back().get();
      }
      else
      {
         // The shared_ptr returned here is non-owning (no-op deleter): the
         // FiniteElementSpace owns the restriction and outlives this cache, so
         // the raw pointer stays valid past the temporary.
         const std::shared_ptr<const Operator> R =
            get_restriction(f, o, rest_descriptor);
         e.op = R.get();
         e.face = dynamic_cast<const FaceRestriction *>(e.op);
      }
      e.is_identity = dynamic_cast<const IdentityOperator *>(e.op) != nullptr;
      if (e.op != nullptr) { e.height = e.op->Height(); e.width = e.op->Width(); }
      return e;
   }

   RestrictionDescriptor rest_descriptor;
   std::vector<Entry> entries;
   /// Identity operators synthesized for quadrature space fields. Shared
   /// rather than unique so that the cache stays copyable: the backend structs
   /// holding one are stored in a std::function.
   std::vector<std::shared_ptr<const IdentityOperator>> owned;
   ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   bool ready = false;
};

namespace detail
{
template <std::size_t... Is>
std::array<RestrictionCache, sizeof...(Is)> MakeRestrictionCachesImpl(
   const RestrictionDescriptor &rest_descriptor, std::index_sequence<Is...>)
{
   return {{ (static_cast<void>(Is), RestrictionCache(rest_descriptor))... }};
}
}

/// @brief @a N restriction caches for restrictions described by
/// @a rest_descriptor.
template <std::size_t N>
std::array<RestrictionCache, N> MakeRestrictionCaches(
   const RestrictionDescriptor &rest_descriptor)
{
   return detail::MakeRestrictionCachesImpl(rest_descriptor,
                                            std::make_index_sequence<N> {});
}

/// @brief Apply the restrictions of @a fields, using @a cache to avoid
/// re-resolving them on every call.
///
/// @see RestrictionCache
inline
void restriction(
   const std::vector<FieldDescriptor> &fields,
   RestrictionCache &cache,
   const std::vector<Vector *> &x_l,
   std::vector<Vector *> &x_e)
{
   MFEM_ASSERT(x_l.size() == x_e.size(),
               "internal error " << x_l.size() << " vs " << x_e.size());
   cache.EnsureSetup(fields);
   for (size_t i = 0; i < fields.size(); i++)
   {
      // If there is no restriction, assume Identity.
      const int s = (cache.Get(i) == nullptr) ? x_l[i]->Size() : cache.Height(i);

      // TODO
      if (x_e[i] == nullptr)
      {
         x_e[i] = new Vector(s);
         x_e[i]->UseDevice(true);
      }
      x_e[i]->SetSize(s);

      if (cache.IsPassthrough(i))
      {
         x_e[i]->NewMemoryAndSize(x_l[i]->GetMemory(), x_l[i]->Size(), false);
         x_e[i]->UseDevice(x_l[i]->UseDevice());
         x_e[i]->SyncMemory(*x_l[i]);
      }
      else
      {
         MFEM_ASSERT(cache.Width(i) == x_l[i]->Size(),
                     "restriction not applicable to given input data size " <<
                     cache.Width(i) << " vs " << x_l[i]->Size());
         cache.Get(i)->Mult(*x_l[i], *x_e[i]);
      }
   }
}

/// @brief Size the residual E-vectors of @a fields, using @a cache to avoid
/// re-resolving the restrictions on every call.
///
/// @see RestrictionCache
inline
void prepare_residual(
   const std::vector<FieldDescriptor> &fields,
   RestrictionCache &cache,
   std::vector<Vector *> &r_e)
{
   cache.EnsureSetup(fields);
   for (size_t i = 0; i < fields.size(); i++)
   {
      const int s = cache.Height(i);
      MFEM_ASSERT(s >= 0, "output field has no restriction");

      // TODO
      if (r_e[i] == nullptr)
      {
         r_e[i] = new Vector(s);
         r_e[i]->UseDevice(true);
      }
      else
      {
         r_e[i]->SetSize(s);
      }
   }
}

/// @brief Apply the transposed restrictions of @a fields, using @a cache to
/// avoid re-resolving them on every call.
///
/// @see RestrictionCache
inline
void restriction_transpose(
   const std::vector<FieldDescriptor> &fields,
   RestrictionCache &cache,
   const std::vector<Vector *> &x_e,
   std::vector<Vector *> &x_l)
{
   cache.EnsureSetup(fields);
   for (size_t i = 0; i < fields.size(); i++)
   {
      const Operator *R = cache.Get(i);

      // If there is no restriction, assume Identity and alias the E-vector.
      // This is decided before allocating, so nothing is allocated only to be
      // freed again on the same call.
      if (R == nullptr || cache.IsPassthrough(i))
      {
         if (x_l[i] != nullptr && x_l[i] != x_e[i]) { delete x_l[i]; }
         x_l[i] = x_e[i];
         continue;
      }

      const int s = cache.Width(i);

      // TODO
      if (x_l[i] == x_e[i])
      {
         x_l[i] = nullptr;
      }
      if (x_l[i] == nullptr)
      {
         x_l[i] = new Vector(s);
         x_l[i]->UseDevice(true);
      }
      x_l[i]->SetSize(s);

      R->MultTranspose(*x_e[i], *x_l[i]);
   }
}

/// @brief Apply the transposed restrictions of @a fields straight into the
/// blocks of @a y, which keep their own memory.
///
/// @see RestrictionCache
inline
void restriction_transpose(
   const std::vector<FieldDescriptor> &fields,
   RestrictionCache &cache,
   const std::vector<Vector *> &x_e,
   MultiVector &y)
{
   cache.EnsureSetup(fields);
   for (size_t i = 0; i < fields.size(); i++)
   {
      const Operator *R = cache.Get(i);
      if (R == nullptr || cache.IsPassthrough(i))
      {
         y[i] = *x_e[i];
         continue;
      }
      y[i].SetSize(cache.Width(i));
      R->MultTranspose(*x_e[i], y[i]);
   }
}

/// @brief Add the transposed restrictions of @a fields applied to @a x_e to
/// the L-vectors @a x_l.
///
/// Unlike restriction_transpose(), this never aliases @a x_l to @a x_e, so the
/// E-vectors of several entity types can be summed into the same L-vectors.
/// @a x_l must already have the L-vector sizes of @a fields.
///
/// @see RestrictionCache
inline
void restriction_transpose_add(
   const std::vector<FieldDescriptor> &fields,
   RestrictionCache &cache,
   const std::vector<Vector *> &x_e,
   std::vector<Vector *> &x_l)
{
   cache.EnsureSetup(fields);
   for (size_t i = 0; i < fields.size(); i++)
   {
      const Operator *R = cache.Get(i);
      if (R == nullptr || cache.IsPassthrough(i))
      {
         *x_l[i] += *x_e[i];
         continue;
      }
      MFEM_ASSERT(x_l[i]->Size() == cache.Width(i),
                  "L-vector size " << x_l[i]->Size() << " does not match the "
                  "restriction width " << cache.Width(i));
      R->AddMultTranspose(*x_e[i], *x_l[i]);
   }
}

/// @brief Add the transposed restrictions of @a fields applied to @a x_e to
/// the blocks of @a y, see the std::vector overload.
inline
void restriction_transpose_add(
   const std::vector<FieldDescriptor> &fields,
   RestrictionCache &cache,
   const std::vector<Vector *> &x_e,
   MultiVector &y)
{
   cache.EnsureSetup(fields);
   for (size_t i = 0; i < fields.size(); i++)
   {
      const Operator *R = cache.Get(i);
      if (R == nullptr || cache.IsPassthrough(i))
      {
         y[i] += *x_e[i];
         continue;
      }
      MFEM_ASSERT(y[i].Size() == cache.Width(i),
                  "L-vector size " << y[i].Size() << " does not match the "
                  "restriction width " << cache.Width(i));
      R->AddMultTranspose(*x_e[i], y[i]);
   }
}

/// @brief Apply the restriction operator to a field, using @a cache to avoid
/// re-resolving it on every call.
///
/// @param u the field descriptor.
/// @param cache the restriction cache belonging to the caller.
/// @param u_l the input vector in vdofs.
/// @param field_e the output vector in edofs.
/// @param ordering the element dof ordering.
/// @tparam entity_t the entity type (see Entity).
/// @see RestrictionCache
inline
void restriction(const FieldDescriptor &u,
                 RestrictionCache &cache,
                 const Vector &u_l,
                 Vector &field_e,
                 ElementDofOrdering ordering)
{
   cache.EnsureSetup(u, ordering);
   MFEM_ASSERT(cache.Width() == u_l.Size(),
               "restriction not applicable to given data size");
   field_e.SetSize(cache.Height());
   cache.Get()->Mult(u_l, field_e);
}

} // namespace mfem::future

#endif // MFEM_USE_MPI
