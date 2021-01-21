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

#include "../../config/config.hpp"
#ifdef MFEM_USE_CEED
#include <ceed.h>
#include <ceed-hash.h>
#endif
#include <tuple>
#include <unordered_map>
#include <string>

namespace mfem
{

class FiniteElementSpace;
class IntegrationRule;
class Vector;

/** @brief Function that determines if a CEED kernel should be used, based on
    the current mfem::Device configuration. */
bool DeviceCanUseCeed();

namespace ceed
{

/** @brief Remove from ceed_basis_map and ceed_restr_map the entries associated
    with the given @a fes. */
void RemoveBasisAndRestriction(const mfem::FiniteElementSpace *fes);

#ifdef MFEM_USE_CEED

/// Initialize a CeedVector from a Vector
void InitVector(const mfem::Vector &v, CeedVector &cv);

/// Initialize a strided CeedElemRestriction
void InitStridedRestriction(const mfem::FiniteElementSpace &fes,
                            CeedInt nelem, CeedInt nqpts, CeedInt qdatasize,
                            const CeedInt *strides,
                            CeedElemRestriction *restr);

/// Initialize a CeedBasis and a CeedElemRestriction
void InitBasisAndRestriction(const mfem::FiniteElementSpace &fes,
                             const mfem::IntegrationRule &ir,
                             Ceed ceed, CeedBasis *basis,
                             CeedElemRestriction *restr);

/// Return the path to the libCEED q-function headers.
const std::string &GetCeedPath();

// Hash table for CeedBasis
using BasisKey = std::tuple<const mfem::FiniteElementSpace*,
                            const mfem::IntegrationRule*,
                            int, int, int>;
struct BasisHash
{
   std::size_t operator()(const BasisKey& k) const
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
using BasisMap = std::unordered_map<const BasisKey, CeedBasis, BasisHash>;

enum restr_type {Standard, Strided};

// Hash table for CeedElemRestriction
using RestrKey = std::tuple<const mfem::FiniteElementSpace*, int, int, int, int>;
struct RestrHash
{
   std::size_t operator()(const RestrKey& k) const
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
using RestrMap =
   std::unordered_map<const RestrKey, CeedElemRestriction, RestrHash>;


#endif

} // namespace ceed

namespace internal
{
extern Ceed ceed;
extern ceed::BasisMap ceed_basis_map;
extern ceed::RestrMap ceed_restr_map;
}

} // namespace mfem

#endif // MFEM_LIBCEED_UTIL
