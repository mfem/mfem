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
#include "../../../mfem.hpp"
#include "../tuple.hpp"
#include <memory>
#include <type_traits>
#include <vector>

namespace mfem::future
{
// Scratch storage and q-function shadow helpers for dFEM backends. The bank
// supports two scratch kinds:
// - quadrature-point scratch: real_t buffers sized as NQ * components_per_qp,
// - global scratch: one tuple of qfunction-local temporaries, independent of
//   NQ, used for values such as flags, scalars, or small Vector workspaces.
template <typename... GlobalScratchTypes>
struct ScratchBank
{

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
      const int size = components_per_qp * nq;
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

   real_t *GetScratchPointer(const int i) const { return ptrs[i]; }
   real_t *operator[](const int i) const { return ptrs[i]; }

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

// Shared base for Q-functions that use ScratchBank and need a matching scratch
// shadow for forward differentiation.
template <typename... GlobalScratchTypes>
struct QFWithScratch
{
   using GlobalScratchTuple = tuple<GlobalScratchTypes...>;

   int nq = 0;
   ScratchBank<GlobalScratchTypes...> scratch;

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

   real_t *GetScratchPointer(const int i) const
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

using QFWithScratchType = QFWithScratch<>;
using QFWithGlobalScratchType = QFWithScratch<bool, real_t, Vector>;

namespace detail
{

template <typename T>
struct qfunc_uses_scratch
{
private:
   template <typename... GlobalScratchTypes>
   static std::true_type Test(const QFWithScratch<GlobalScratchTypes...> *);

   static std::false_type Test(...);

public:
   static constexpr bool value = decltype(Test(
                                             static_cast<std::remove_cv_t<std::remove_reference_t<T>> *>(nullptr)))::value;
};

template <typename T>
inline constexpr bool qfunc_uses_scratch_v =
   qfunc_uses_scratch<T>::value;

struct unused_qfunc_shadow { };

template <typename qfunc_t, bool uses_scratch>
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
      qfunc_uses_scratch_v<qfunc_t>>::type;

// Create a persistent q-function shadow if the q-function uses scratch, otherwise return an empty struct.
template <typename qfunc_t>
inline qfunc_shadow_t<qfunc_t> MakeQFunctionShadowStorage(
   const qfunc_t &qfunc)
{
   if constexpr (qfunc_uses_scratch_v<qfunc_t>)
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
