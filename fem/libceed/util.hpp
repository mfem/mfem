// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
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

#include <tuple>
#include <unordered_map>
#include <string>
// #ifdef MFEM_USE_CEED
#include <ceed.h>
#include <ceed-hash.h>
// #endif

namespace mfem
{

class FiniteElementSpace;
class IntegrationRule;
class Vector;

/// Initialize a CeedVector from a Vector
void InitCeedVector(const Vector &v, CeedVector &cv);

/// Initialize a strided CeedElemRestriction
void InitCeedStridedRestriction(const FiniteElementSpace &fes,
                                CeedInt nelem, CeedInt nqpts, CeedInt qdatasize,
                                const CeedInt *strides,
                                CeedElemRestriction *restr);

/// Initialize a CeedBasis and a CeedElemRestriction
void InitCeedBasisAndRestriction(const FiniteElementSpace &fes,
                                 const IntegrationRule &ir,
                                 Ceed ceed, CeedBasis *basis,
                                 CeedElemRestriction *restr);

/** @brief Remove from ceed_basis_map and ceed_restr_map the entries associated
    with the given @a fes. */
void RemoveCeedBasisAndRestriction(const FiniteElementSpace *fes);

/** @brief Function that determines if a CEED kernel should be used, based on
    the current mfem::Device configuration. */
bool DeviceCanUseCeed();

/// Return the path to the libCEED q-function headers.
const std::string &GetCeedPath();

// Hash table for CeedBasis
using CeedBasisKey =
   std::tuple<const FiniteElementSpace*, const IntegrationRule*, int, int, int>;
struct CeedBasisHash
{
   std::size_t operator()(const CeedBasisKey& k) const
   {
      return CeedHashCombine(
                CeedHashCombine(
                   CeedHashInt(reinterpret_cast<CeedHash64_t>(std::get<0>(k))),
                   CeedHashInt(reinterpret_cast<CeedHash64_t>(std::get<1>(k)))),
                CeedHashCombine(
                   CeedHashCombine(CeedHashInt(std::get<2>(k)),
                                   CeedHashInt(std::get<3>(k))),
                   CeedHashInt(std::get<4>(k))));
   }
};
using CeedBasisMap =
   std::unordered_map<const CeedBasisKey, CeedBasis, CeedBasisHash>;

enum restr_type {Standard, Strided};

// Hash table for CeedElemRestriction
using CeedRestrKey = std::tuple<const FiniteElementSpace*, int, int, int, int>;
struct CeedRestrHash
{
   std::size_t operator()(const CeedRestrKey& k) const
   {
      return CeedHashCombine(
                CeedHashCombine(
                   CeedHashCombine(
                      CeedHashInt(reinterpret_cast<CeedHash64_t>(std::get<0>(k))),
                      CeedHashInt(std::get<1>(k))),
                   CeedHashCombine(CeedHashInt(std::get<2>(k)),
                                   CeedHashInt(std::get<3>(k)))),
                CeedHashInt(std::get<4>(k)));
   }
};
using CeedRestrMap =
   std::unordered_map<const CeedRestrKey, CeedElemRestriction, CeedRestrHash>;

} // namespace mfem

#endif // MFEM_LIBCEED_UTIL
