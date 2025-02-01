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

#ifndef MFEM_KERNEL_DISPATCH_HPP
#define MFEM_KERNEL_DISPATCH_HPP

#include "../config/config.hpp"
#include "kernel_reporter.hpp"
#include <unordered_map>
#include <tuple>
#include <cstddef>

namespace mfem
{

// The MFEM_REGISTER_KERNELS macro registers kernels for runtime dispatch using
// a dispatch map.
//
// This creates a dispatch table (a static member variable) named @a KernelName
// containing function points of type @a KernelType. These are followed by one
// or two sets of parenthesized argument types.
//
// The first set of argument types contains the types that are used to dispatch
// to either specialized or fallback kernels. The second set of argument types
// can be used to further specialize the kernel without participating in
// dispatch (a canonical example is NBZ, determining the size of the thread
// blocks; this is required to specialize kernels for optimal performance, but
// is not relevant for dispatch).
//
// After calling this macro, the user must implement the Kernel and Fallback
// static member functions, which return pointers to the appropriate kernel
// functions depending on the parameters.
//
// Specialized functions can be registered using the static AddSpecialization
// member function.

#define MFEM_EXPAND(X) X // Workaround needed for MSVC compiler

#define MFEM_REGISTER_KERNELS(KernelName, KernelType, ...)                     \
   MFEM_EXPAND(MFEM_EXPAND(MFEM_REGISTER_KERNELS_N(__VA_ARGS__,2,1,))          \
      (KernelName,KernelType,__VA_ARGS__))

#define MFEM_REGISTER_KERNELS_N(_1, _2, N, ...) MFEM_REGISTER_KERNELS_##N

// Expands a variable length macro parameter so that multiple variable length
// parameters can be passed to the same macro.
#define MFEM_PARAM_LIST(...) __VA_ARGS__

// Version of MFEM_REGISTER_KERNELS without any "optional" (non-dispatch)
// parameters.
#define MFEM_REGISTER_KERNELS_1(KernelName, KernelType, Params)                \
   MFEM_REGISTER_KERNELS_(KernelName, KernelType, Params, (), Params)

// Version of MFEM_REGISTER_KERNELS without any optional (non-dispatch)
// parameters (e.g. NBZ).
#define MFEM_REGISTER_KERNELS_2(KernelName, KernelType, Params, OptParams)     \
   MFEM_REGISTER_KERNELS_(KernelName, KernelType, Params, OptParams,           \
                          (MFEM_PARAM_LIST Params, MFEM_PARAM_LIST OptParams))

// P1 are the parameters, P2 are the optional (non-dispatch parameters), and P3
// is the concatenation of P1 and P2. We need to pass it as a separate argument
// to avoid a trailing comma in the case that P2 is empty.
#define MFEM_REGISTER_KERNELS_(KernelName, KernelType, P1, P2, P3)             \
   class KernelName : public                                                   \
   KernelDispatchTable<KernelName, KernelType,                                 \
      internal::KernelTypeList<MFEM_PARAM_LIST P1>,                            \
      internal::KernelTypeList<MFEM_PARAM_LIST P2>>                            \
   {                                                                           \
   public:                                                                     \
      const char *kernel_name = MFEM_KERNEL_NAME(KernelName);                  \
      using KernelSignature = KernelType;                                      \
      template <MFEM_PARAM_LIST P3>                                            \
      static MFEM_EXPORT KernelSignature Kernel();                             \
      static MFEM_EXPORT KernelSignature Fallback(MFEM_PARAM_LIST P1);         \
      static MFEM_EXPORT KernelName &Get()                                     \
      { static KernelName table; return table;}                                \
   }

/// @brief Hashes variadic packs for which each type contained in the variadic
/// pack has a specialization of `std::hash` available.
///
/// For example, packs containing int, bool, enum values, etc.
template<typename ...KernelParameters>
struct KernelDispatchKeyHash
{
private:
   template<int N>
   size_t operator()(std::tuple<KernelParameters...> value) const { return 0; }

   // The hashing formula here is taken directly from the Boost library, with
   // the magic number 0x9e3779b9 chosen to minimize hashing collisions.
   template<std::size_t N, typename THead, typename... TTail>
   size_t operator()(std::tuple<KernelParameters...> value) const
   {
      constexpr int Index = N - sizeof...(TTail) - 1;
      auto lhs_hash = std::hash<THead>()(std::get<Index>(value));
      auto rhs_hash = operator()<N, TTail...>(value);
      return lhs_hash^(rhs_hash + 0x9e3779b9 + (lhs_hash<<6) + (lhs_hash>>2));
   }
public:
   /// Returns the hash of the given @a value.
   size_t operator()(std::tuple<KernelParameters...> value) const
   {
      return operator()<sizeof...(KernelParameters),KernelParameters...>(value);
   }
};

namespace internal { template<typename... Types> struct KernelTypeList { }; }

template<typename... T> class KernelDispatchTable { };

template <typename Kernels,
          typename Signature,
          typename... Params,
          typename... OptParams>
class KernelDispatchTable<Kernels,
         Signature,
         internal::KernelTypeList<Params...>,
         internal::KernelTypeList<OptParams...>>
{
   using TableType = std::unordered_map<std::tuple<Params...>,
         Signature, KernelDispatchKeyHash<Params...>>;
   TableType table;

public:
   /// @brief Run the kernel with the given dispatch parameters and arguments.
   ///
   /// If a compile-time specialized version of the kernel with the given
   /// parameters has been registered, it will be called. Otherwise, the
   /// fallback kernel will be called.
   template<typename... Args>
   static void Run(Params... params, Args&&... args)
   {
      const auto &table = Kernels::Get().table;
      const std::tuple<Params...> key = std::make_tuple(params...);
      const auto it = table.find(key);
      if (it != table.end())
      {
         it->second(std::forward<Args>(args)...);
      }
      else
      {
         KernelReporter::ReportFallback(Kernels::Get().kernel_name, params...);
         Kernels::Fallback(params...)(std::forward<Args>(args)...);
      }
   }

   /// Register a specialized kernel for dispatch.
   template <Params... PARAMS>
   struct Specialization
   {
      // Version without optional parameters
      static void Add()
      {
         std::tuple<Params...> param_tuple(PARAMS...);
         Kernels::Get().table[param_tuple] =
            Kernels:: template Kernel<PARAMS..., OptParams{}...>();
      };
      // Version with optional parameters
      template <OptParams... OPT_PARAMS>
      struct Opt
      {
         static void Add()
         {
            std::tuple<Params...> param_tuple(PARAMS...);
            Kernels::Get().table[param_tuple] =
               Kernels:: template Kernel<PARAMS..., OPT_PARAMS...>();
         }
      };
   };

   /// Return the dispatch map table
   static const TableType &GetDispatchTable()
   {
      return Kernels::Get().table;
   }
};

}

#endif
