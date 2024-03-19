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

namespace mfem
{
namespace internal
{
template<typename... Types>
struct KernelTypeList {};
}

constexpr int ipow(int x, int p) { return p == 0 ? 1 : x*ipow(x, p-1); }

template<typename... T>
class ApplyPAKernelsClassTemplate {};

template<typename... T>
class DiagonalPAKernelsClassTemplate {};

template<typename Signature, typename... UserParams, typename... KernelParams>
class ApplyPAKernelsClassTemplate<Signature, internal::KernelTypeList<UserParams...>, internal::KernelTypeList<KernelParams...>>
{
private:
   constexpr static int D(int D1D) { return (11 - D1D) / 2; }
public:
   using KernelSignature = Signature;
   using KernelArgTypes2D = internal::KernelTypeList<UserParams...>;
   using KernelArgTypes3D = internal::KernelTypeList<KernelParams...>;

   constexpr static int NBZ(int D1D, int Q1D)
   {
      return ipow(2, D(D1D) >= 0 ? D(D1D) : 0);
   }

   template<KernelParams... params>
   static KernelSignature Kernel2D();

   template<KernelParams... params>
   static KernelSignature Kernel3D();

   static KernelSignature Fallback2D();

   static KernelSignature Fallback3D();
};

template<typename Signature, typename... UserParams, typename... KernelParams>
class DiagonalPAKernelsClassTemplate<Signature, internal::KernelTypeList<UserParams...>, internal::KernelTypeList<KernelParams...>>
{
public:
   using KernelSignature = Signature;
   using KernelArgTypes2D = internal::KernelTypeList<UserParams...>;
   using KernelArgTypes3D = internal::KernelTypeList<KernelParams...>;

   template<KernelParams... params>
   static KernelSignature Kernel2D();

   template<KernelParams... params>
   static KernelSignature Kernel3D();

   static KernelSignature Fallback2D();

   static KernelSignature Fallback3D();
};

template <typename KeyType, typename KernelType, typename HashFunction>
class DispatchTable
{
protected:
   static std::unordered_map<KeyType, KernelType, HashFunction> table;
};


template<typename ...KernelParameters>
struct KernelDispatchKeyHash
{
   using Tuple = std::tuple<KernelParameters...>;

private:
   template<int N>
   size_t operator()(Tuple value) const { return 0; }

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

template <typename ApplyKernelsHelperClass, typename... UserParams, typename... KernelParams>
class KernelDispatchTable<ApplyKernelsHelperClass, internal::KernelTypeList<UserParams...>, internal::KernelTypeList<KernelParams...>> :
         DispatchTable<std::tuple<int, UserParams...>, typename ApplyKernelsHelperClass::KernelSignature, KernelDispatchKeyHash<int, UserParams...>>
{

   // These typedefs prevent AddSpecialization from compiling unless the provided
   // kernel parameters match the kernel parameters specified to ApplyKernelsHelperClass.
   using KernelArgTypes2D = typename ApplyKernelsHelperClass::KernelArgTypes2D;
   using KernelArgTypes3D = typename ApplyKernelsHelperClass::KernelArgTypes3D;
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
   void Run2D(UserParams... params, KernelArgs... args)
   {
      auto key = std::make_tuple(2, params...);
      const auto it = this->table.find(key);
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

   template<typename... KernelArgs>
   void Run3D(UserParams... params, KernelArgs... args)
   {
      auto key = std::make_tuple(3, params...);
      const auto it = this->table.find(key);
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

   template<typename... KernelArgs>
   void Run(int dim, UserParams... params, KernelArgs... args)
   {
      if (dim == 2)
      {
         Run2D(params..., args...);
      }
      else if (dim == 3)
      {
         Run3D(params..., args...);
      }
      else
      {
         MFEM_ABORT("Only 2 and 3 dimensional kernels exist");
      }

   }

   template<UserParams... params>
   struct AddSpecialization2D
   {
      void operator()(KernelDispatchTable* table_ptr)
      {
         constexpr int DIM = 2;
         constexpr std::tuple<int, UserParams...> param_tuple = std::make_tuple(DIM,
                                                                                params...);
         // All kernels require at least D1D and Q1D
         static_assert(sizeof...(params) >= 2,
                       "All specializations require at least two template parameters");

         constexpr int NBZ = GetNBZ(std::get<0>(param_tuple), std::get<1>(param_tuple));

         table_ptr->table[param_tuple] = ApplyKernelsHelperClass::template
                                         Kernel2D<params..., NBZ>();
      }
   };

   template<UserParams... params>
   struct AddSpecialization3D
   {
      void operator()(KernelDispatchTable* table_ptr)
      {
         constexpr int DIM = 3;
         constexpr std::tuple<int, UserParams...> param_tuple = std::make_tuple(DIM,
                                                                                params...);
         static_assert(sizeof...(UserParams) >= 2,
                       "All specializations require at least two template parameters");

         table_ptr->table[param_tuple] = ApplyKernelsHelperClass::template
                                         Kernel3D<params...>();
      }
   };

};

}

#endif
