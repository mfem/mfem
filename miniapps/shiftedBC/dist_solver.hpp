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

#ifndef MFEM_DIST_FUNCTION_HPP
#define MFEM_DIST_FUNCTION_HPP

#include "mfem.hpp"

namespace mfem
{

class DistanceSolver
{
public:
   DistanceSolver() { }

   virtual ~DistanceSolver() { }

   virtual void ComputeDistance(Coefficient &zero_level_set,
                                ParGridFunction &distance) = 0;
};

// K. Crane et al:
// Geodesics in Heat: A New Approach to Computing Distance Based on Heat Flow
class HeatDistanceSolver : public DistanceSolver
{
private:
   ParGridFunction source, diffused_source;

public:
   HeatDistanceSolver(double diff_coeff)
      : DistanceSolver(),
        parameter_t(diff_coeff), smooth_steps(0), transform(true) { }

   void ComputeDistance(Coefficient &zero_level_set,
                        ParGridFunction &distance);

   const ParGridFunction &GetLastSourceGF() const
   { return source; }
   const ParGridFunction &GetLastDiffusedSourceGF() const
   { return diffused_source; }

   int parameter_t, smooth_steps;
   bool transform;
};

class GradientCoefficient : public VectorCoefficient
{
private:
   const GridFunction &u;

public:
   GradientCoefficient(const GridFunction &u_gf, int dim)
      : VectorCoefficient(dim), u(u_gf) { }

   void Eval(Vector &V, ElementTransformation &T, const IntegrationPoint &ip)
   {
      T.SetIntPoint(&ip);

      u.GetGradient(T, V);
      const double norm = V.Norml2() + 1e-12;
      V /= -norm;
   }
};

} // namespace mfem

#endif
