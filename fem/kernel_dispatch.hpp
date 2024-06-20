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

#ifndef MFEM_KERNELDISPATCH_HPP
#define MFEM_KERNELDISPATCH_HPP

#include "../general/error.hpp"

#include <functional>
#include <unordered_map>
#include <tuple>
#include <cmath>

namespace mfem
{

namespace internal
{
template<typename... Types>
struct KernelTypeList {};
}

// Expands a variable length macro parameter so that multiple variable length parameters
// can be passed to the same macro.
#define MFEM_PARAM_LIST(...)                                                                 \
__VA_ARGS__                                                                                  \

// Declare the class used to dispatch shared memory kernels when the fallback methods require
// templateization.  Note that in this case the 1D kernel also uses identical template parameters
// as the fallback methods.
#define MFEM_DECLARE_KERNELS_WITH_FALLBACK_PARAMS(KernelClassName, KernelType, UserParams, FallbackParams)        \
class KernelClassName : public                                                               \
   KernelsClassTemplate<KernelType,                                                          \
   internal::KernelTypeList<UserParams>, internal::KernelTypeList<FallbackParams>>           \
   {                                                                                         \
   public:                                                                                   \
      template<FallbackParams>                                                                \
      static KernelSignature Kernel1D();                                                     \
      template<UserParams, int>                                                              \
      static KernelSignature Kernel2D();                                                     \
      template<UserParams>                                                                   \
      static KernelSignature Kernel3D();                                                     \
      template<FallbackParams>                                                               \
      static KernelSignature Fallback2D();                                                   \
      template<FallbackParams>                                                               \
      static KernelSignature Fallback3D();                                                   \
   };                                                                                        \

// Declare the class used to dispatch shared memory kernels when the fallback methods don't
// require template parameters.  Note that the 2D kernel always requires an extra integral
// parameter corresponding to `NBZ`.
#define MFEM_DECLARE_KERNELS(KernelClassName, KernelType, UserParams, FallbackParams)        \
class KernelClassName : public                                                               \
   KernelsClassTemplate<KernelType,                                                          \
   internal::KernelTypeList<UserParams>, internal::KernelTypeList<FallbackParams>>           \
   {                                                                                         \
   public:                                                                                   \
      static KernelSignature Kernel1D();                                                     \
      template<UserParams, int>                                                              \
      static KernelSignature Kernel2D();                                                     \
      template<UserParams>                                                                   \
      static KernelSignature Kernel3D();                                                     \
      static KernelSignature Fallback2D();                                                   \
      static KernelSignature Fallback3D();                                                   \
   };

constexpr int ipow(int x, int p) { return p == 0 ? 1 : x*ipow(x, p-1); }

// This forward declaration of KernelClassTemplate allows a specialization accepting multiple variadic parameters.
template<typename... T>
class KernelsClassTemplate {};

// KernelsClassTemplate holds the template methods for 2D and 3D kernels.  The class DispatchTable defined below
// accesses `KernelClassTemplate` when assigning key value pairs, assigning a tuple key of `DIM, UserParams`
// a specialized kernel.
template<typename Signature, typename... UserParams, typename... FallbackParams>
class KernelsClassTemplate<Signature, internal::KernelTypeList<UserParams...>, internal::KernelTypeList<FallbackParams...>>
{
private:
   constexpr static int D(int D1D) { return (11 - D1D) / 2; }
public:
   using KernelSignature = Signature;

   constexpr static int NBZ(int D1D, int Q1D)
   {
      return ipow(2, D(D1D) >= 0 ? D(D1D) : 0);
   }

   static KernelSignature Kernel1D();

   template<UserParams... params>
   static KernelSignature Kernel2D();

   template<UserParams... params>
   static KernelSignature Kernel3D();

   template<FallbackParams... params>
   static KernelSignature Fallback2D();

   template<FallbackParams... params>
   static KernelSignature Fallback3D();
};

// DispatchTable is the class that stores specialized kernels, assigning a key tuple of `KeyType`
// a value of type `KernelType`.
template <typename KeyType, typename KernelType, typename HashFunction>
class DispatchTable
{
protected:
   std::unordered_map<KeyType, KernelType, HashFunction> table;
};

// KernelDispatchKeyHash is a functor that hashes variadic packs for which each type contained in the
// variadic pack has a specialization of `std::hash` available.  For example, packs containing int,
// bool, enum values, etc.
template<typename ...KernelParameters>
struct KernelDispatchKeyHash
{
   using Tuple = std::tuple<KernelParameters...>;

private:
   template<int N>
   size_t operator()(Tuple value) const { return 0; }

   // The hashing formula here is taken directly from the Boost library, with the magic number 0x9e3779b9
   // chosen to minimize hashing collisions.
   template<std::size_t N, typename THead, typename... TTail>
   size_t operator()(Tuple value) const
   {
      constexpr int Index = N - sizeof...(TTail) - 1;
      auto lhs_hash = std::hash<THead>()(std::get<Index>(value));
      auto rhs_hash = operator()<N, TTail...>(value);
      return lhs_hash ^(rhs_hash + 0x9e3779b9 + (lhs_hash << 6) + (lhs_hash >> 2));
   }

public:
   size_t operator()(Tuple value) const
   {
      return operator()<sizeof...(KernelParameters), KernelParameters...>(value);
   }
};

template<typename... T>
class KernelDispatchTable {};

// KernelDispatchTable is derived from `DispatchTable` using the `KernelDispatchKeyHash` functor above
// to assign specialized kernels with individual keys.
template <typename ApplyKernelsHelperClass, typename... UserParams, typename... FallbackParams>
class KernelDispatchTable<ApplyKernelsHelperClass, internal::KernelTypeList<UserParams...>, internal::KernelTypeList<FallbackParams...>> :
         DispatchTable<std::tuple<int, UserParams...>, typename ApplyKernelsHelperClass::KernelSignature, KernelDispatchKeyHash<int, UserParams...>>
{
   // These typedefs prevent AddSpecialization from compiling unless the provided
   // kernel parameters match the kernel parameters specified to ApplyKernelsHelperClass.
   using Signature = typename ApplyKernelsHelperClass::KernelSignature;

private:
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
   { return GetNBZ_<ApplyKernelsHelperClass>(d1d, q1d, nullptr); }

public:
   // TODO(bowen) Force this to use the same signature as the Signature typedef
   // above.
   template<typename... KernelArgs>
   void Run(int dim, UserParams... params, KernelArgs&... args)
   {
      std::tuple<int, UserParams...>  key;
      if (dim != 1)
      {
         key = std::make_tuple(dim, params...);
      }
      else
      {
         // In one dimension, only one kernel exists with all user params set to zero.
         // In this case, ignore the params... variable pack.
         std::get<0>(key) = 1;
      }
      const auto it = this->table.find(key);
      if (it != this->table.end())
      {
         printf("Using specialized kernel\n");
         it->second(args...);
      }
      else if (dim == 1)
      {
         MFEM_ABORT("1 dimensional kernel not registered.  This is an internal MFEM error.")
      }
      else if (dim == 2)
      {
         printf("Using non-specialized kernel\n");
         ApplyKernelsHelperClass::Fallback2D()(args...);
      }
      else if (dim == 3)
      {
         printf("Using non-specialized kernel\n");
         ApplyKernelsHelperClass::Fallback3D()(args...);
      }
      else
      {
         MFEM_ABORT("Kernels only exist for spatial dimensions 3 and less.")
      }
   }

   using SpecializedTableType =
      KernelDispatchTable<ApplyKernelsHelperClass, internal::KernelTypeList<UserParams...>, internal::KernelTypeList<FallbackParams...>>;

   template<UserParams... params>
   struct AddSpecialization1D
   {
      void operator()(SpecializedTableType* table_ptr)
      {
         constexpr int DIM = 1;
         std::tuple<int, UserParams...> param_tuple  (DIM, params...);
         table_ptr->table[param_tuple] = ApplyKernelsHelperClass::Kernel1D();
      }
   };

   template<typename ... args>
   static constexpr int getD1D(int D1D, args...)
   {
      return D1D;
   }
   template<typename ... args>
   static constexpr int getQ1D(int /*D1D*/, int Q1D, args...)
   {
      return Q1D;
   }
   /// Functors are needed here instead of functions because of a bug in GCC where a variadic
   /// type template cannot be used to define a parameter pack.
   template<UserParams... params>
   struct AddSpecialization2D
   {
      void operator()(SpecializedTableType* table_ptr)
      {
         constexpr int DIM = 2;
         std::tuple<int, UserParams...> param_tuple (DIM, params...);

         // All kernels require at least D1D and Q1D, which are listed first in a
         // parameter pack.
         constexpr int D1D = getD1D(params...);
         constexpr int Q1D = getQ1D(params...);
         constexpr int NBZ = GetNBZ(D1D, Q1D);

         table_ptr->table[param_tuple] = ApplyKernelsHelperClass::template
                                         Kernel2D<params..., NBZ>();
      }
   };

   template<UserParams... params>
   struct AddSpecialization3D
   {
      void operator()(SpecializedTableType* table_ptr)
      {
         constexpr int DIM = 3;
         std::tuple<int, UserParams...> param_tuple (DIM, params...);
         table_ptr->table[param_tuple] = ApplyKernelsHelperClass::template
                                         Kernel3D<params...>();
      }
   };

};

}

#endif
