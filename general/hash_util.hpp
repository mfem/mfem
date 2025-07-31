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

#ifndef MFEM_HASH_UTIL_HPP
#define MFEM_HASH_UTIL_HPP

#include <cstddef>
#include <tuple>
#include <functional>
#include <utility>

namespace
{

inline size_t VariadicHash() { return 0; }

template<typename TFirst, typename... TRest>
size_t VariadicHash(const TFirst &first, const TRest&... rest)
{
   // The hashing formula here is taken directly from the Boost library, with
   // the magic number 0x9e3779b9 chosen to minimize hashing collisions.
   const size_t lhs_hash = std::hash<TFirst>()(first);
   const size_t rhs_hash = VariadicHash(rest...);
   return lhs_hash^(rhs_hash + 0x9e3779b9 + (lhs_hash<<6) + (lhs_hash>>2));
}

} // anonymous namespace

/// Specialization of std::hash for std::tuple of hashable types.
template<typename ...Args>
struct std::hash<std::tuple<Args...>>
{
public:
   /// Returns the hash of the given @a value.
   size_t operator()(const std::tuple<Args...> &value) const
   {
      return std::apply(VariadicHash<Args...>, value);
   }
};

/// Specialization of std::hash for std::pair of hashable types.
template<typename T1, typename T2>
struct std::hash<std::pair<T1, T2>>
{
   std::size_t operator()(const std::pair<T1, T2> &value) const
   {
      return VariadicHash(value.first, value.second);
   }
};


#endif
