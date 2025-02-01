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

#ifndef MFEM_ELTRANS_BASIS
#define MFEM_ELTRANS_BASIS

#include "general/forall.hpp"

// this file contains utilities for computing nodal basis functions and their
// derivatives in device kernels

namespace mfem {

namespace eltrans {

/// 1D Lagrange basis from [0, 1]
class Lagrange {
public:
  /// interpolant node locations, in reference space
  real_t *z;

  /// number of points
  int pN;

  /// @b Evaluates the @a i'th Lagrange polynomial at @a x
  real_t MFEM_HOST_DEVICE eval(real_t x, int i) const {
    real_t u0 = 1;
    real_t den = 1;
    for (int j = 0; j < pN; ++j) {
      if (i != j) {
        real_t d_j = (x - z[j]);
        u0 = d_j * u0;
        den *= (z[i] - z[j]);
      }
    }
    den = 1 / den;
    return u0 * den;
  }

  /// @b Evaluates the @a i'th Lagrange polynomial and its first derivative at
  /// @a x
  void MFEM_HOST_DEVICE eval_d1(real_t &p, real_t &d1, real_t x, int i) const {
    real_t u0 = 1;
    real_t u1 = 0;
    real_t den = 1;
    for (int j = 0; j < pN; ++j) {
      if (i != j) {
        real_t d_j = (x - z[j]);
        u1 = d_j * u1 + u0;
        u0 = d_j * u0;
        den *= (z[i] - z[j]);
      }
    }
    den = 1 / den;
    p = u0 * den;
    d1 = u1 * den;
  }

  /// @b Evaluates the @a i'th Lagrange polynomial and its first and second
  /// derivatives at @a x
  void MFEM_HOST_DEVICE eval_d2(real_t &p, real_t &d1, real_t &d2, real_t x,
                                int i) const {
    real_t u0 = 1;
    real_t u1 = 0;
    real_t u2 = 0;
    real_t den = 1;
    for (int j = 0; j < pN; ++j) {
      if (i != j) {
        real_t d_j = (x - z[j]);
        u2 = d_j * u2 + u1;
        u1 = d_j * u1 + u0;
        u0 = d_j * u0;
        den *= (z[i] - z[j]);
      }
    }
    den = 1 / den;
    p = den * u0;
    d1 = den * u1;
    d2 = 2 * den * u2;
  }
};

} // namespace eltrans
} // namespace mfem

#endif
