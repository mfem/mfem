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

#ifndef MFEM_LIBCEED_UTIL
#define MFEM_LIBCEED_UTIL

#include "../../../general/error.hpp"
#include "ceed.hpp"
#ifdef MFEM_USE_CEED
#include <ceed/backend.h>
#endif
#include <array>
#include <functional>
#include <tuple>
#include <unordered_map>
#include <string>

namespace mfem
{

class FiniteElement;
class FiniteElementSpace;
class ElementTransformation;
class IntegrationRule;
class Vector;

/** @brief Function that determines if a CEED kernel should be used, based on
    the current mfem::Device configuration. */
bool DeviceCanUseCeed();

namespace ceed
{

/** @brief Remove from ceed_basis_map and ceed_restr_map the entries associated
    with the given @a fes when @a fes gets destroyed. */
void RemoveBasisAndRestriction(const mfem::FiniteElementSpace *fes);

#ifdef MFEM_USE_CEED

#define PCeedChk(err) do {                                                     \
     if ((err))                                                                \
     {                                                                         \
        const char *errmsg;                                                    \
        CeedGetErrorMessage(internal::ceed, &errmsg);                          \
        MFEM_ABORT(errmsg);                                                    \
     }                                                                         \
  } while(0);

/// Initialize a CeedVector from an mfem::Vector
void InitVector(const mfem::Vector &v, Ceed ceed, CeedVector &cv);

int CeedOperatorGetActiveField(CeedOperator oper, CeedOperatorField *field);

/// Return the path to the libCEED QFunction headers.
const std::string &GetCeedPath();

/// Wrapper for std::hash.
template <typename T>
inline std::size_t CeedHash(const T key)
{
   return std::hash<T> {}(key);
}

/// Effective way to combine hashes (from libCEED).
inline std::size_t CeedHashCombine(std::size_t seed, std::size_t hash)
{
   // See https://doi.org/10.1002/asi.10170, or
   //     https://dl.acm.org/citation.cfm?id=759509.
   return seed ^ (hash + (seed << 6) + (seed >> 2));
}

// Hash table for CeedBasis
using BasisKey =
   std::tuple<const mfem::FiniteElementSpace *, const mfem::FiniteElementSpace *,
   const mfem::IntegrationRule *, std::array<int, 3>>;
struct BasisHash
{
   std::size_t operator()(const BasisKey &k) const
   {
      return CeedHashCombine(
                CeedHashCombine(
                   CeedHashCombine(
                      CeedHash(std::get<0>(k)),
                      CeedHash(std::get<1>(k))),
                   CeedHash(std::get<2>(k))),
                CeedHashCombine(
                   CeedHashCombine(CeedHash(std::get<3>(k)[0]),
                                   CeedHash(std::get<3>(k)[1])),
                   CeedHash(std::get<3>(k)[2])));
   }
};
using BasisMap = std::unordered_map<const BasisKey, CeedBasis, BasisHash>;

// Hash table for CeedElemRestriction
using RestrKey =
   std::tuple<const mfem::FiniteElementSpace *, std::array<int, 4>,
   std::array<int, 3>>;
struct RestrHash
{
   std::size_t operator()(const RestrKey &k) const
   {
      return CeedHashCombine(
                CeedHash(std::get<0>(k)),
                CeedHashCombine(
                   CeedHashCombine(
                      CeedHashCombine(CeedHash(std::get<1>(k)[0]),
                                      CeedHash(std::get<1>(k)[1])),
                      CeedHashCombine(CeedHash(std::get<1>(k)[2]),
                                      CeedHash(std::get<1>(k)[3]))),
                   CeedHashCombine(
                      CeedHashCombine(CeedHash(std::get<2>(k)[0]),
                                      CeedHash(std::get<2>(k)[1])),
                      CeedHash(std::get<2>(k)[2]))));
   }
};
using RestrMap =
   std::unordered_map<const RestrKey, CeedElemRestriction, RestrHash>;

#endif

} // namespace ceed

namespace internal
{

#ifdef MFEM_USE_CEED
/** @warning These maps have a tendency to create bugs when adding new "types"
    of CeedBasis and CeedElemRestriction. Definitions in general/device.cpp. */
extern ceed::BasisMap ceed_basis_map;
#ifdef MFEM_USE_OPENMP
#pragma omp threadprivate(ceed_basis_map)
#endif
extern ceed::RestrMap ceed_restr_map;
#ifdef MFEM_USE_OPENMP
#pragma omp threadprivate(ceed_restr_map)
#endif
#endif

} // namespace internal

} // namespace mfem

#endif // MFEM_LIBCEED_UTIL
