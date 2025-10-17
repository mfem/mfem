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

#include "hash.hpp"

#ifdef MFEM_USE_GNUTLS
#include <gnutls/gnutls.h>
#include <gnutls/crypto.h>
#if GNUTLS_VERSION_NUMBER >= 0x020a00
#define HAVE_GNUTLS_HASH_FUNCTIONS
#endif
#endif

namespace mfem
{

#ifdef HAVE_GNUTLS_HASH_FUNCTIONS
constexpr gnutls_digest_algorithm_t HASH_ALGORITHM = GNUTLS_DIG_SHA256;
#endif

HashFunction::HashFunction()
{
#ifndef HAVE_GNUTLS_HASH_FUNCTIONS
   hash_data = nullptr;
#else
   gnutls_hash_init((gnutls_hash_hd_t*)(&hash_data), HASH_ALGORITHM);
#endif
}

HashFunction::~HashFunction()
{
#ifdef HAVE_GNUTLS_HASH_FUNCTIONS
   gnutls_hash_deinit((gnutls_hash_hd_t)hash_data, nullptr);
#endif
}

void HashFunction::HashBuffer(const void *buffer, size_t num_bytes)
{
#ifndef HAVE_GNUTLS_HASH_FUNCTIONS
   MFEM_CONTRACT_VAR(buffer);
   MFEM_CONTRACT_VAR(num_bytes);
#else
   gnutls_hash((gnutls_hash_hd_t)hash_data, buffer, num_bytes);
#endif
}

inline constexpr char to_hex(unsigned char u)
{
   return (u < 10) ? '0' + u : (u < 16) ? 'a' + (u - 10) : '?';
}

std::string HashFunction::GetHash() const
{
   std::string hash;
#ifndef MFEM_USE_GNUTLS
   hash = "(GnuTLS is required for hashing)";
#elif !defined (HAVE_GNUTLS_HASH_FUNCTIONS)
   hash = "(Old GnuTLS version: does not support hashing)";
#else
   constexpr unsigned max_hash_len = 64;
   unsigned char hash_bytes[max_hash_len];
   unsigned hash_len = gnutls_hash_get_len(HASH_ALGORITHM);
   MFEM_VERIFY(hash_len <= max_hash_len, "internal error");
   hash.reserve(2*hash_len);
   gnutls_hash_output((gnutls_hash_hd_t)hash_data, hash_bytes);
   for (unsigned i = 0; i < hash_len; i++)
   {
      hash += to_hex(hash_bytes[i]/16);
      hash += to_hex(hash_bytes[i]%16);
   }
#endif
   return hash;
}

constexpr static uint64_t rotl64(uint64_t x, int r)
{
   return (x << r) | (x >> (64 - r));
}

void Hasher::init(uint64_t seed)
{
   data[0] = seed;
   data[1] = seed;
   nbytes = 0;
}

void Hasher::add_block(uint64_t k1, uint64_t k2)
{
   constexpr uint64_t c1 = 0x87c37b91114253d5ull;
   constexpr uint64_t c2 = 0x4cf5ad432745937full;

   k1 *= c1;
   k1 = rotl64(k1, 31);
   k1 *= c2;
   data[0] ^= k1;

   data[0] = rotl64(data[0], 27);
   data[0] += data[1];
   data[0] = data[0] * 5 + 0x52dce729ull;

   k2 *= c2;
   k2 = rotl64(k2, 33);
   k2 *= c1;
   data[1] ^= k2;

   data[1] = rotl64(data[1], 31);
   data[1] += data[0];
   data[1] = data[1] * 5 + 0x38495ab5ull;
}

static uint64_t fmix64(uint64_t k)
{
   // http://zimbry.blogspot.com/2011/09/better-bit-mixing-improving-on.html
   // mix13
   k ^= k >> 30;
   k *= 0xbf58476d1ce4e5b9ull;
   k ^= k >> 27;
   k *= 0x94d049bb133111ebull;
   k ^= k >> 31;
   return k;
}

void Hasher::append(const uint8_t *vs, uint64_t bytes)
{
   if (bytes == 0)
   {
      return;
   }
   auto rem = nbytes % 16;
   nbytes += bytes;
   uint8_t *tmp = reinterpret_cast<uint8_t *>(buf_);
   while (true)
   {
      if (bytes + rem >= 16)
      {
         std::copy(vs, vs + 16 - rem, tmp + rem);
         add_block(buf_[0], buf_[1]);
         vs += (16 - rem);
         bytes -= (16 - rem);
         rem = 0;
      }
      else
      {
         std::copy(vs, vs + bytes, tmp + rem);
         return;
      }
   }
}

void Hasher::finalize()
{
   auto rem = nbytes % 16;
   if (rem > 0)
   {
      nbytes -= rem;
      if (rem <= 8)
      {
         finalize(buf_[0], rem);
      }
      else
      {
         finalize(buf_[0], buf_[1], rem);
      }
      return;
   }
   data[0] ^= nbytes;
   data[1] ^= nbytes;

   data[0] += data[1];
   data[1] += data[0];

   data[0] = fmix64(data[0]);
   data[1] = fmix64(data[1]);

   data[0] += data[1];
   data[1] += data[0];
}

void Hasher::finalize(uint64_t k1, int num)
{
   constexpr uint64_t c1 = 0x87c37b91114253d5ull;
   constexpr uint64_t c2 = 0x4cf5ad432745937full;
   nbytes += num;
   k1 *= c1;
   k1 = rotl64(k1, 31);
   k1 *= c2;
   data[0] ^= k1;

   data[0] ^= nbytes;
   data[1] ^= nbytes;

   data[0] += data[1];
   data[1] += data[0];

   data[0] = fmix64(data[0]);
   data[1] = fmix64(data[1]);

   data[0] += data[1];
   data[1] += data[0];
}

void Hasher::finalize(uint64_t k1, uint64_t k2, int num)
{
   constexpr uint64_t c1 = 0x87c37b91114253d5ull;
   constexpr uint64_t c2 = 0x4cf5ad432745937full;
   nbytes += num;
   k2 *= c2;
   k2 = rotl64(k2, 33);
   k2 *= c1;
   data[1] ^= k2;

   k1 *= c1;
   k1 = rotl64(k1, 31);
   k1 *= c2;
   data[0] ^= k1;

   data[0] ^= nbytes;
   data[1] ^= nbytes;

   data[0] += data[1];
   data[1] += data[0];

   data[0] = fmix64(data[0]);
   data[1] = fmix64(data[1]);

   data[0] += data[1];
   data[1] += data[0];
}

} // namespace mfem
