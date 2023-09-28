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

#include "../../../config/config.hpp"
#include <functional>
#include <string>
#include <tuple>
#include <unordered_map>

#include "ceed.hpp"
#ifdef MFEM_USE_CEED
#include <ceed/backend.h>  // for CeedOperatorField
#endif

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
    with the given @a fes. */
void RemoveBasisAndRestriction(const mfem::FiniteElementSpace *fes);

#ifdef MFEM_USE_CEED

#define PCeedChk(err) do {                                                     \
     if ((err))                                                                \
     {                                                                         \
        const char * errmsg;                                                   \
        CeedGetErrorMessage(internal::ceed, &errmsg);                          \
        MFEM_ABORT(errmsg);                                                    \
     }                                                                         \
  } while(0);

/// Initialize a CeedVector from an mfem::Vector
void InitVector(const mfem::Vector &v, CeedVector &cv);

/** @brief Initialize a CeedBasis and a CeedElemRestriction based on an
    mfem::FiniteElementSpace @a fes, and an mfem::IntegrationRule @a ir.

    @param[in] fes The finite element space.
    @param[in] ir The integration rule.
    @param[in] ceed The Ceed object.
    @param[out] basis The `CeedBasis` to initialize.
    @param[out] restr The `CeedElemRestriction` to initialize.

    @warning Only for non-mixed finite element spaces. */
void InitBasisAndRestriction(const mfem::FiniteElementSpace &fes,
                             const mfem::IntegrationRule &ir,
                             Ceed ceed, CeedBasis *basis,
                             CeedElemRestriction *restr);

/** @brief Initialize a CeedBasis and a CeedElemRestriction based on an
    mfem::FiniteElementSpace @a fes, and an mfem::IntegrationRule @a ir,
    and a list of @a nelem elements of indices @a indices.

    @param[in] fes The finite element space.
    @param[in] ir The integration rule.
    @param[in] nelem The number of elements.
    @param[in] indices The indices of the elements of same type in the
                       `FiniteElementSpace`. If `indices == nullptr`, assumes
                       that the `FiniteElementSpace` is not mixed.
    @param[in] ceed The Ceed object.
    @param[out] basis The `CeedBasis` to initialize.
    @param[out] restr The `CeedElemRestriction` to initialize. */
void InitBasisAndRestriction(const FiniteElementSpace &fes,
                             const IntegrationRule &ir,
                             int nelem,
                             const int* indices,
                             Ceed ceed, CeedBasis *basis,
                             CeedElemRestriction *restr);

int CeedOperatorGetActiveField(CeedOperator oper, CeedOperatorField *field);


template <typename Integrator>
const IntegrationRule & GetRule(
   const Integrator &integ,
   const FiniteElement &trial_fe,
   const FiniteElement &test_fe,
   ElementTransformation &Trans);

/// Return the path to the libCEED q-function headers.
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
using BasisKey = std::tuple<const mfem::FiniteElementSpace*,
      const mfem::IntegrationRule*,
      int, int, int>;
struct BasisHash
{
   std::size_t operator()(const BasisKey& k) const
   {
      return CeedHashCombine(
                CeedHashCombine(
                   CeedHash(std::get<0>(k)),
                   CeedHash(std::get<1>(k))),
                CeedHashCombine(
                   CeedHashCombine(CeedHash(std::get<2>(k)),
                                   CeedHash(std::get<3>(k))),
                   CeedHash(std::get<4>(k))));
   }
};
using BasisMap = std::unordered_map<const BasisKey, CeedBasis, BasisHash>;

enum restr_type {Standard, Strided, Coeff};

// Hash table for CeedElemRestriction
using RestrKey =
   std::tuple<const mfem::FiniteElementSpace*, int, int, int, int>;
struct RestrHash
{
   std::size_t operator()(const RestrKey& k) const
   {
      return CeedHashCombine(
                CeedHashCombine(
                   CeedHashCombine(
                      CeedHash(std::get<0>(k)),
                      CeedHash(std::get<1>(k))),
                   CeedHashCombine(CeedHash(std::get<2>(k)),
                                   CeedHash(std::get<3>(k)))),
                CeedHash(std::get<4>(k)));
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
    of CeedBasis and CeedElemRestriction. */
extern ceed::BasisMap ceed_basis_map;
extern ceed::RestrMap ceed_restr_map;
#endif

} // namespace internal

} // namespace mfem

#endif // MFEM_LIBCEED_UTIL
