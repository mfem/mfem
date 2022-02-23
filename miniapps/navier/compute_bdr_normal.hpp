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

#ifndef MFEM_COMPUTE_BDR_NORMAL_HPP
#define MFEM_COMPUTE_BDR_NORMAL_HPP

#include "mfem.hpp"

namespace mfem
{
namespace navier
{

class BoundaryNormalEvaluator
{
protected:
   ParFiniteElementSpace &vfes, &pfes;

   const IntegrationRule &ir;

   const DofToQuad *maps;
   const FaceGeometricFactors *geom;

   mutable ParGridFunction g_gf, y_gf;
   mutable Vector g_face, y_face;
public:
   BoundaryNormalEvaluator(
      ParFiniteElementSpace &vfes_,
      ParFiniteElementSpace &pfes_,
      const IntegrationRule &ir_);

   /// @brief Evaluate y = (g.n, v), where g is a grid function defined by the
   /// vector-valued T-DOF vector @a g, and @a y is a scalar-valued T-DOF
   /// vector.
   void Mult(const Vector &g, Vector &y) const;
};

} // namespace navier
} // namespace mfem

#endif
