// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// Internal header, included only by .cpp files

#include "quadinterpolator.hpp"

namespace mfem
{

namespace internal
{

namespace quadrature_interpolator
{

// Tensor-product evaluation of quadrature point values: dispatch function.
template<QVectorLayout VL>
void TensorValues(const int NE,
                  const int vdim,
                  const DofToQuad &maps,
                  const Vector &e_vec,
                  Vector &q_val);

// Tensor-product evaluation of quadrature point derivatives: dispatch function.
template<QVectorLayout VL>
void TensorDerivatives(const int NE,
                       const int vdim,
                       const DofToQuad &maps,
                       const Vector &e_vec,
                       Vector &q_der);

// Tensor-product evaluation of quadrature point physical derivatives: dispatch
// function.
template<QVectorLayout VL>
void TensorPhysDerivatives(const int NE,
                           const int vdim,
                           const DofToQuad &maps,
                           const GeometricFactors &geom,
                           const Vector &e_vec,
                           Vector &q_der);

// Tensor-product evaluation of quadrature point determinants: dispatch
// function.
void TensorDeterminants(const int NE,
                        const int vdim,
                        const DofToQuad &maps,
                        const Vector &e_vec,
                        Vector &q_det,
                        Vector &d_buff);

} // namespace quadrature_interpolator

} // namespace internal

} // namespace mfem
