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

#include <array>
#include <cstddef>
#include <tuple>
#include <functional>
#include <utility>
#include <cstdint>

namespace mfem
{

/// @brief streaming implementation for murmurhash3 128 (x64).
///
/// Constructs the hash in 3 stages: init, append, finalize.
struct Hasher
{
   /// @brief Storage for the final hash result after finalize() is called.
   ///
   /// Use data[1] when only 64 bits are required.
   uint64_t data[2] = {0, 0};

private:
   uint64_t nbytes = 0;
   uint64_t buf_[2] = {0, 0};

public:

   /// Resets the Hasher back to an initial seed
   void init(uint64_t seed = 0);

   /// Append data @a vs of size @a bytes.
   void append(const std::byte *vs, uint64_t bytes);

   void finalize();

private:
   /// Add a block of 16 bytes.
   void add_block(uint64_t k1, uint64_t k2);

   /// @brief Add [1-8] more bytes, then finalize.
   ///
   /// @a num must satisfy 0 < num < 9.
   void finalize(uint64_t k1, int num);

   /// @brief Add [1-15] more bytes, then finalize.
   ///
   /// @a num must satisfy 0 < num < 16.
   void finalize(uint64_t k1, uint64_t k2, int num);
};

template <class T> struct ChainedHasher
{
   static void Append(Hasher &hasher, const T &value)
   {
      if constexpr (std::is_fundamental_v<T> || std::is_pointer_v<T>)
      {
         hasher.append(reinterpret_cast<const std::byte *>(&value), sizeof(T));
      }
      else
      {
         std::hash<T> h;
         auto v = h(value);
         hasher.append(reinterpret_cast<std::byte *>(&v), sizeof(v));
      }
   }
};

template <class T, class V> struct ChainedHasher<std::pair<T, V>>
{
   static void Append(Hasher &hasher, const std::pair<T, V> &value)
   {
      ChainedHasher<T>::Append(hasher, value.first);
      ChainedHasher<V>::Append(hasher, value.second);
   }
};

template <class T, size_t N> struct ChainedHasher<std::array<T, N>>
{
   static void Append(Hasher &hasher, const std::array<T, N> &value)
   {
      for (size_t i = 0; i < N; ++i)
      {
         ChainedHasher<T>::Append(hasher, value[i]);
      }
   }
};

template<class... Ts> struct ChainedHasher<std::tuple<Ts...>>
{
private:
   template <size_t N>
   static void AppendImpl(Hasher &hasher, const std::tuple<Ts...> &value)
   {
      ChainedHasher<std::decay_t<decltype(std::get<N>(value))>>::Append(
                                                                hasher, std::get<N>(value));
      if constexpr (N + 1 < sizeof...(Ts))
      {
         AppendImpl<N + 1>(hasher, value);
      }
   }

public:
   static void Append(Hasher &hasher, const std::tuple<Ts...> &value)
   {
      if constexpr (sizeof...(Ts))
      {
         AppendImpl<0>(hasher, value);
      }
   }
};

/// Helper class for hashing std::pair of hashable types.
struct PairHasher
{
   template <class T, class V>
   size_t operator()(const std::pair<T, V> &v) const noexcept
   {
      Hasher hash;
      // chosen randomly with a 2^64-sided dice
      hash.init(0xfebd1fe69813c14full);
      ChainedHasher<std::pair<T, V>>::Append(hash, v);
      hash.finalize();
      return hash.data[1];
   }
};

/// Helper class for hashing std::array of a hashable type.
struct ArrayHasher
{
   template <class T, size_t N>
   size_t operator()(const std::array<T, N> &v) const noexcept
   {
      Hasher hash;
      // chosen randomly with a 2^64-sided dice
      hash.init(0xfebd1fe69813c14full);
      ChainedHasher<std::array<T, N>>::Append(hash, v);
      hash.finalize();
      return hash.data[1];
   }
};

/// Helper class for hashing std::tuple of hashable types.
struct TupleHasher
{
   template <class T>
   size_t operator()(const T &v) const noexcept
   {
      Hasher hash;
      // chosen randomly with a 2^64-sided dice
      hash.init(0xfebd1fe69813c14full);
      ChainedHasher<T>::Append(hash, v);
      hash.finalize();
      return hash.data[1];
   }
};

} // namespace mfem

#endif
