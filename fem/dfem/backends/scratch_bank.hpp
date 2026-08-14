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
#pragma once
#include "../../../general/error.hpp"
#include "../../../linalg/vector.hpp"
#include "../tuple.hpp"
#include <initializer_list>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace mfem::future
{
// Scratch storage and q-function shadow helpers for dFEM backends. The bank
// supports two scratch kinds:
// - quadrature-point scratch: real_t buffers sized as NQ * components_per_qp,
// - global scratch: one tuple of qfunction-local temporaries, independent of
//   NQ, used for values such as flags, scalars, or small Vector workspaces.
//
// @a scalar_t is the scalar the owning q-function uses at a quadrature point.
// With Enzyme this is real_t and the tangent lives in a separate shadow bank.
// Without Enzyme the q-function is evaluated on native duals, which carry the
// tangent inside the value itself; the bank then widens its backing storage
// accordingly so that a scratch entry can round-trip a dual without dropping
// the gradient. Backing storage stays a real_t Vector in both cases, so the
// device and shadow plumbing is unchanged.
template <typename scalar_t, typename... GlobalScratchTypes>
struct ScratchBank
{
   static_assert(sizeof(scalar_t) % sizeof(real_t) == 0,
                 "scratch scalar must be a whole number of real_t");

   /// Number of real_t needed to back one scalar_t scratch entry.
   static constexpr int scalar_size = sizeof(scalar_t) / sizeof(real_t);


   //=================================
   ///<--- Global scratch utilities.
   //=================================

   using GlobalScratchTuple = tuple<GlobalScratchTypes...>;

   template <typename T>
   static T MakeGlobalScratchShadow(const T &)
   {
      return T {};
   }

   static Vector MakeGlobalScratchShadow(const Vector &primal)
   {
      Vector shadow(primal.Size());
      shadow.UseDevice(true);
      shadow = 0.0;
      return shadow;
   }

   template <typename Tuple, size_t... Is>
   static auto MakeGlobalScratchShadowTuple(const Tuple &primal,
                                            std::index_sequence<Is...>)
   {
      return make_tuple(MakeGlobalScratchShadow(get<Is>(primal))...);
   }

   template <typename Tuple>
   static auto MakeGlobalScratchShadowTuple(const Tuple &primal)
   {
      return MakeGlobalScratchShadowTuple(
                primal, std::make_index_sequence<tuple_size<Tuple>::value> {});
   }


   //===========================
   ///<--- Scratch objects
   //===========================

   mutable GlobalScratchTuple global;

   int nq = 0;
   std::vector<int> components;
   std::vector<int> sizes;
   std::vector<std::shared_ptr<Vector>> owned;
   std::vector<real_t *> ptrs;


   //===========================
   ///<--- Setter methods
   //===========================

   void SetScratch(const int nq_,
                   std::initializer_list<int> components_per_qp = {1})
   {
      SetScratch(nq_, std::vector<int>(components_per_qp));
   }

   void SetScratch(const int nq_, const std::vector<int> &components_per_qp)
   {
      nq = nq_;
      components.clear();
      sizes.clear();
      owned.clear();
      ptrs.clear();
      for (int component_count : components_per_qp)
      {
         AddScratch(component_count);
      }
   }

   void AddScratch(const int components_per_qp = 1)
   {
      MFEM_VERIFY(nq > 0, "SetScratch must be called before AddScratch");
      MFEM_VERIFY(components_per_qp > 0,
                  "scratch components per quadrature point must be positive");
      owned.push_back(std::make_shared<Vector>());
      Vector &scratch = *owned.back();
      const int size = components_per_qp * nq * scalar_size;
      scratch.SetSize(size);
      scratch.UseDevice(true);
      scratch = 0.0;
      components.push_back(components_per_qp);
      sizes.push_back(scratch.Size());
      ptrs.push_back(scratch.ReadWrite());
   }

   void SetGlobalScratch(const GlobalScratchTuple &global_)
   {
      global = global_;
   }


   //===========================
   ///<--- Getter methods
   //===========================

   /// Scratch buffer @a i viewed as the q-function's scalar type.
   scalar_t *GetScratchPointer(const int i) const
   {
      return reinterpret_cast<scalar_t *>(ptrs[i]);
   }

   scalar_t *operator[](const int i) const { return GetScratchPointer(i); }

   /// Raw real_t backing storage of scratch buffer @a i. Its size is
   /// scalar_size times the number of scalar_t entries.
   Vector &GetScratchVector(const int i) const { return *owned[i]; }

   template <int I>
   auto &GetGlobalScratch() const
   {
      return get<I>(global);
   }


   //===========================
   ///<--- Utils methods
   //===========================

   void CloneScratchLayoutTo(ScratchBank &shadow) const
   {
      shadow.SetScratch(nq, components);
      shadow.SetGlobalScratch(MakeGlobalScratchShadowTuple(global));
   }

   int Size() const { return static_cast<int>(ptrs.size()); }
};

// Shared base for Q-functions that use ScratchBank. Under Enzyme a matching
// scratch shadow is created for forward differentiation; with native duals the
// tangent rides along in the scratch entry and no shadow is created.
template <typename scalar_t, typename... GlobalScratchTypes>
struct QFWithScratch
{
   using GlobalScratchTuple = tuple<GlobalScratchTypes...>;
   using ScratchScalar = scalar_t;

   /// Number of real_t backing one scratch entry; see ScratchBank.
   static constexpr int scalar_size =
      ScratchBank<scalar_t, GlobalScratchTypes...>::scalar_size;

   int nq = 0;
   ScratchBank<scalar_t, GlobalScratchTypes...> scratch;

   void SetScratch(const int nq_,
                   std::initializer_list<int> components_per_qp = {1})
   {
      nq = nq_;
      scratch.SetScratch(nq, components_per_qp);
   }

   void SetScratch(const int nq_, const std::vector<int> &components_per_qp)
   {
      nq = nq_;
      scratch.SetScratch(nq, components_per_qp);
   }

   void SetScratch(const int nq_, const int num_scratch_elem,
                   const int components_per_qp = 1)
   {
      nq = nq_;
      scratch.SetScratch(nq,
                         std::vector<int>(num_scratch_elem, components_per_qp));
   }

   void SetGlobalScratch(const GlobalScratchTuple &global_scratch_)
   {
      scratch.SetGlobalScratch(global_scratch_);
   }

   Vector &GetScratchVector(const int i) const
   {
      return scratch.GetScratchVector(i);
   }

   scalar_t *GetScratchPointer(const int i) const
   {
      return scratch.GetScratchPointer(i);
   }

   template <int I>
   auto &GetGlobalScratch() const
   {
      return scratch.template GetGlobalScratch<I>();
   }

   void CloneScratchLayoutTo(QFWithScratch &shadow) const
   {
      shadow.nq = nq;
      scratch.CloneScratchLayoutTo(shadow.scratch);
   }

   QFWithScratch CreateShadow() const
   {
      QFWithScratch shadow;
      CloneScratchLayoutTo(shadow);
      return shadow;
   }
};

/// Q-function base with quadrature-point scratch only. @a scalar_t is the
/// scalar the q-function signature uses (real_t under Enzyme, dual otherwise).
template <typename scalar_t = real_t>
using QFWithScratchType = QFWithScratch<scalar_t>;

/// Q-function base with quadrature-point scratch and a global scratch tuple.
template <typename scalar_t = real_t>
using QFWithGlobalScratchType =
   QFWithScratch<scalar_t, bool, real_t, Vector>;

namespace detail
{

template <typename T>
struct qfunc_uses_scratch
{
private:
   template <typename scalar_t, typename... GlobalScratchTypes>
   static std::true_type Test(
      const QFWithScratch<scalar_t, GlobalScratchTypes...> *);

   static std::false_type Test(...);

public:
   static constexpr bool value = decltype(Test(
                                             static_cast<std::remove_cv_t<std::remove_reference_t<T>> *>(nullptr)))::value;
};

template <typename T>
inline constexpr bool qfunc_uses_scratch_v =
   qfunc_uses_scratch<T>::value;

struct unused_qfunc_shadow { };

// A separate shadow scratch bank only exists for Enzyme, which writes tangents
// into shadow memory. The native dual fallback carries the tangent inside the
// scratch entry itself (see ScratchBank::scalar_size), so a shadow bank would
// be allocated and never read; it is dropped entirely there.
template <typename T>
inline constexpr bool qfunc_needs_shadow_v =
#ifdef MFEM_USE_ENZYME
   qfunc_uses_scratch_v<T>;
#else
   false;
#endif

template <typename qfunc_t, bool needs_shadow>
struct qfunc_shadow_type
{
   using type = unused_qfunc_shadow;
};

template <typename qfunc_t>
struct qfunc_shadow_type<qfunc_t, true>
{
   using type = decltype(std::declval<const qfunc_t &>().CreateShadow());
};

template <typename qfunc_t>
using qfunc_shadow_t = typename qfunc_shadow_type<qfunc_t,
      qfunc_needs_shadow_v<qfunc_t>>::type;

// Create a persistent q-function shadow if one is needed, otherwise return an empty struct.
template <typename qfunc_t>
inline qfunc_shadow_t<qfunc_t> MakeQFunctionShadowStorage(
   const qfunc_t &qfunc)
{
   if constexpr (qfunc_needs_shadow_v<qfunc_t>)
   {
      return qfunc.CreateShadow();
   }
   else
   {
      MFEM_CONTRACT_VAR(qfunc);
      return {};
   }
}

} // namespace detail

}
