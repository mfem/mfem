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

#ifndef MFEM_TMOP_PA_HPP
#define MFEM_TMOP_PA_HPP

#include "../tmop.hpp"
#include "../tmop_tools.hpp"

#include "../../config/config.hpp"
#include "../../general/forall.hpp"
#include "../../fem/kernels.hpp"
#include "../../linalg/kernels.hpp"
#include "../../linalg/dinvariants.hpp"

namespace mfem
{

/// Abstract base class for the 2D metric TMOP kernels.
struct MetricTMOPKer2D
{
   static constexpr int DIM = 2;
   using Args = kernels::InvariantsEvaluator2D::Buffers;
   virtual MFEM_HOST_DEVICE
   void EvalP(const double (&Jpt)[4], const double *w, double (&P)[4]) {}
   virtual MFEM_HOST_DEVICE
   void AssembleH(const int qx, const int qy, const int e,
                  const double weight, const double (&Jpt)[4],
                  const double *w, const DeviceTensor<7> &H) {}
};

/// Abstract base class for the 3D metric TMOP kernels.
struct MetricTMOPKer3D
{
   static constexpr int DIM = 3;
   using Args = kernels::InvariantsEvaluator3D::Buffers;
   virtual MFEM_HOST_DEVICE
   void EvalP(const double (&Jpt)[9], const double *w, double (&P)[9]) {}
   virtual MFEM_HOST_DEVICE
   void AssembleH(const int qx, const int qy, const int qz, const int e,
                  const double weight, double *Jrt, double *Jpr,
                  const double (&Jpt)[9], const double *w,
                  const DeviceTensor<8> &H) {}
};

} // namespace mfem

#endif // MFEM_TMOP_PA_HPP
