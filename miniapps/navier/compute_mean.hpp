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

#ifndef MFEM_COMPUTE_MEAN_HPP
#define MFEM_COMPUTE_MEAN_HPP

#include "mfem.hpp"

namespace mfem
{
namespace navier
{

class MeanEvaluator
{
protected:
   ParFiniteElementSpace &fes;
   const IntegrationRule &ir;
   QuadratureInterpolator quad_interp;

   const GeometricFactors *geom;

   double volume;

   mutable ParGridFunction x_evec, x_qvec;
public:
   MeanEvaluator(ParFiniteElementSpace &fes_, const IntegrationRule &ir_);

   /// Compute the integral of a grid function defined by the T-DOF vector @a x.
   double ComputeIntegral(const Vector &x) const;

   /// Subtract the mean of the grid function defined by the T-DOF vector @a x
   /// so that it has mean zero.
   void MakeMeanZero(Vector &x) const;
};

} // namespace navier
} // namespace mfem

#endif
