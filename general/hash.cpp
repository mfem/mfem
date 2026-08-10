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

} // namespace mfem
