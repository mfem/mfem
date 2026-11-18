// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef LID_DRIVEN_CAVITY_FACTORY_HPP
#define LID_DRIVEN_CAVITY_FACTORY_HPP

#include "mfem.hpp"
#include "problem.hpp"
#include <cmath>

class LidDrivenCavity :  public CoefficientFactory
{
private:
   real_t mu = 0.01;
   real_t rho = 1.0;

public:

   LidDrivenCavity(int vdim) : CoefficientFactory(vdim), mu(0.01), rho(1.0) {}
   LidDrivenCavity(int vdim, real_t mu_) : CoefficientFactory(vdim), mu(mu_),
      rho(1.0) {}
   LidDrivenCavity(int vdim, real_t mu_, real_t rho_) : CoefficientFactory(vdim),
      mu(mu_), rho(rho_)  {}

   // Physical parameters
   mfem::real_t Density(const mfem::Vector &x) const override
   {
      return rho;
   }

   mfem::real_t Viscosity(const mfem::Vector &x) const override
   {
      return mu;
   }

   // Velocity
   void Velocity(const mfem::Vector &coord, mfem::Vector &u) const override
   {
      const mfem::real_t x = coord[0];
      const mfem::real_t y = coord[1];

      const real_t eps = 1e-6;

      u[0] = std::pow(4.0*x*(1.0-x),0.01)*y*y*y;
      u[1] = 0.0;

      if (y     < eps) { u[0] = 0.0; }
      if (x     < eps) { u[0] = 0.0; }
      if (1.0-x < eps) { u[0] = 0.0; }
      if (1.0-y < eps)
      {
         u[0] = 1.0;
         u[1] = 0.0;
         if (x     < eps) { u[0] = 0.5; }
         if (1.0-x < eps) { u[0] = 0.5; }
      }
   }
};

#endif

