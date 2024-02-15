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

#ifndef MFEM_KERNELDISPATCH_HPP
#define MFEM_KERNELDISPATCH_HPP

#include "../general/error.hpp"

#include <functional>
#include <unordered_map>

namespace mfem
{
namespace internal {
template<typename... Types>
struct KernelTypeList {};
}

constexpr int ipow(int x, int p) { return p == 0 ? 1 : x*ipow(x, p-1); }

template<typename... T>
class ApplyPAKernelsClassTemplate {};

template<typename... T>
class DiagonalPAKernelsClassTemplate {};

template<typename Signature, typename... Params2D, typename... Params3D>
class ApplyPAKernelsClassTemplate<Signature, internal::KernelTypeList<Params2D...>, internal::KernelTypeList<Params3D...>>
{
private:
   constexpr static int D(int D1D) { return (11 - D1D) / 2; }
public:
   using KernelSignature = Signature;
   using KernelArgTypes2D = internal::KernelTypeList<Params2D...>;
   using KernelArgTypes3D = internal::KernelTypeList<Params3D...>;

   constexpr static int NBZ(int D1D, int Q1D)
   {
      return ipow(2, D(D1D) >= 0 ? D(D1D) : 0);
   }

   template<Params2D... params>
   static KernelSignature Kernel2D();

   template<Params3D... params>
   static KernelSignature Kernel3D();

   static KernelSignature Fallback2D();

   static KernelSignature Fallback3D();
};

template<typename Signature, typename... Params2D, typename... Params3D>
class DiagonalPAKernelsClassTemplate<Signature, internal::KernelTypeList<Params2D...>, internal::KernelTypeList<Params3D...>>
{
public:
   using KernelSignature = Signature;
   using KernelArgTypes2D = internal::KernelTypeList<Params2D...>;
   using KernelArgTypes3D = internal::KernelTypeList<Params3D...>;

   template<Params2D... params>
   static KernelSignature Kernel2D();

   template<Params3D... params>
   static KernelSignature Kernel3D();

   static KernelSignature Fallback2D();

   static KernelSignature Fallback3D();
};

template <typename KeyType, typename KernelType, typename HashFunction>
class DispatchTable
{
protected:
   std::unordered_map<KeyType, KernelType, HashFunction> table;
};


template<typename ...KernelParameters>
struct KernelDispatchKeyHash
{
   std::hash<size_t> h;
   std::size_t hasher_helper(const std::tuple<KernelParameters...> t, int hashing_index) {
      if (hashing_index >= sizeof...(KernelParameters)) {
         return 1;
      }
      size_t lhs_hash = h(static_cast<size_t>(std::get<hashing_index>(t)));
      if (hashing_index == sizeof...(KernelParameters) - 1) {
         return lhs_hash;
      }
      size_t rhs_hash = hasher_helper(t, hashing_index + 1);
      // This formula for combining hashes is from Boost.
      return lhs_hash ^(rhs_hash + 0x9e3779b9 + (lhs_hash << 6) + (lhs_hash >> 2));
   }

   std::size_t operator()(const std::tuple<KernelParameters...> &k) const
   {
      return hasher_helper(k, 0);
   }
};

template<typename... T>
class KernelDispatchTable {};

template <typename ApplyKernelsHelperClass, typename... Params2D, typename... Params3D>
class KernelDispatchTable<ApplyKernelsHelperClass, internal::KernelTypeList<Params2D...>, internal::KernelTypeList<Params3D...>>
{

   // These typedefs prefent AddSpecialization from compiling unless the provided
   // kernel parameters match the kernel parameters specified to ApplyKernelsHelperClass.
   using KernelArgTypes2D = typename ApplyKernelsHelperClass::KernelArgTypes2D;
   using KernelArgTypes3D = typename ApplyKernelsHelperClass::KernelArgTypes3D;
   using Signature = typename ApplyKernelsHelperClass::KernelSignature;

private:
   static auto lookup_table_2d = DispatchTable<std::tuple<Params2D...>, typename ApplyKernelsHelperClass::KernelSignature, KernelDispatchKeyHash<Params2D...>>{};
   static auto lookup_table_3d = DispatchTable<std::tuple<Params2D...>, typename ApplyKernelsHelperClass::KernelSignature, KernelDispatchKeyHash<Params2D...>>{};
   // If the type U has member U::NBZ, this overload will be selected, and will
   // return U::NBZ(d1d, q1d).
   template <typename U>
   static constexpr int GetNBZ_(int d1d, int q1d, decltype(U::NBZ(0,0),nullptr))
   {
      return U::NBZ(d1d, q1d);
   }

   // If the type U does not have member U::NBZ, this "fallback" overload will
   // be selected
   template <typename U> static constexpr int GetNBZ_(int d1d, int q1d, ...)
   {
      return 0;
   }

   // Return T::NBZ(d1d, q1d) if T::NBZ is defined, 0 otherwise.
   static constexpr int GetNBZ(int d1d, int q1d)
   { return GetNBZ_<>(d1d, q1d, nullptr); }

public:

   // TODO(bowen) Force this to use the same signature as the Signature typedef
   // above.
   template<Params2D... params, typename... KernelArgs>
   void Run2D(KernelArgs... args) {
      constexpr auto key = std::make_tuple(params...);
      const auto it = lookup_table_2d.table.find(key);
      if (it != this->table.end())
      {
         printf("Specialized.\n");
         it->second(args...);
      }
      else
      {
         ApplyKernelsHelperClass::Fallback2D()(args...);
      }
   }

   template<Params3D... params, typename... KernelArgs>
   void Run3D(KernelArgs... args) {
      constexpr auto key = std::make_tuple(params...);
      const auto it = lookup_table_3d.table.find(key);
      if (it != this->table.end())
      {
         printf("Specialized.\n");
         it->second(args...);
      }
      else
      {
         ApplyKernelsHelperClass::Fallback3D()(args...);
      }
   }

   template<Params2D... params>
   void AddSpecialization2D() {
      constexpr int DIM = 2;
      constexpr auto param_tuple = std::make_tuple<params...>;
      // All kernels require at least D1D and Q1D
      static_assert(sizeof...(params) >= 2 &&
            std::is_same<decltype(std::get<0>(param_tuple)), int>::value &&
            std::is_same<decltype(std::get<1>(param_tuple)), int>::value);

      constexpr int NBZ = GetNBZ(std::get<0>(param_tuple), std::get<1>(param_tuple));
      constexpr auto key_tuple = std::make_tuple<DIM, params...>;

      lookup_table_2d.table[param_tuple] = ApplyKernelsHelperClass::template Kernel2D<NBZ, params...>();
   }

   template<Params3D... params>
   void AddSpecialization3D() {
      constexpr int DIM = 3;
      constexpr auto param_tuple = std::make_tuple<DIM, params...>;
      static_assert(sizeof...(params) >= 2 &&
            std::is_same<decltype(std::get<0>(param_tuple)), int>::value &&
            std::is_same<decltype(std::get<1>(param_tuple)), int>::value);

      lookup_table_3d.table[param_tuple] = ApplyKernelsHelperClass::template Kernel3D<params...>();
   }
};

}

#endif
