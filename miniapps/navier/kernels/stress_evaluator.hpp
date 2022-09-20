// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_STRESS_EVALUATOR_HPP
#define MFEM_STRESS_EVALUATOR_HPP

#include "mfem.hpp"

namespace mfem
{
namespace navier
{

class StressEvaluator
{
public:
   StressEvaluator(const ParFiniteElementSpace &kvfes,
                   const ParFiniteElementSpace &ufes, const IntegrationRule &i);

   void Apply(const Vector &kv, const Vector &u, Vector &y);

   ~StressEvaluator() {};

private:
   const ParFiniteElementSpace &kvfes;
   const ParFiniteElementSpace &ufes;
   const IntegrationRule &ir;
   const Operator *Pkv = nullptr;
   const Operator *Pu = nullptr;
   const Operator *Rkv = nullptr;
   const Operator *Ru = nullptr;
   const QuadratureInterpolator* qi = nullptr;
   const DofToQuad *maps = nullptr;
   const GeometricFactors *geom = nullptr;
   const int dim, ne = 0;
   Vector kv_l, kv_e, dkv_qp, u_l, u_e, y_l, y_e;
};

} // namespace navier
} // namespace mfem

#endif