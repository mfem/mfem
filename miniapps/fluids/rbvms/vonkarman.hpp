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

#ifndef VON_KARMAN_FLOW_FACTORY_HPP
#define VON_KARMAN_FLOW_FACTORY_HPP

#include "mfem.hpp"
#include "problem.hpp"
#include <cmath>

class VonKarmanFlow :  public CoefficientFactory
{
private:
   real_t mu = 0.01;
   real_t rho = 1.0;

public:
   VonKarmanFlow(int vdim) : CoefficientFactory(vdim), mu(0.01), rho(1.0) {}
   VonKarmanFlow(int vdim, real_t mu_) : CoefficientFactory(vdim), mu(mu_),
      rho(1.0) {}
   VonKarmanFlow(int vdim, real_t mu_, real_t rho_) : CoefficientFactory(vdim),
      mu(mu_), rho(rho_)  {}
   //  ~VonKarmanFlow() = default;

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
      mfem::real_t x = coord[0];
      mfem::real_t y = coord[1];

      real_t r = sqrt(x*x + y*y);
      real_t r0 = 0.5;

      // Base flow + BC
      u[0] = fmin(1.0, r - r0);
      u[1] = 0.0;

      // Disturbance 1
      x = coord[0];
      y = coord[1] - 3.0;
      r = sqrt(x*x + y*y);
      r0 = 1.0;
      u[0] += 2.0*fmax(0.0, r0 - r);
      u[1] -= 0.5*fmax(0.0, r0 - r);

      // Disturbance 2
      x = coord[0] + 3.0;
      y = coord[1];
      r = sqrt(x*x + y*y);
      r0 = 1.0;
      u[0] -= 0.5*fmax(0.0, r0 - r);
      u[1] += 2.0*fmax(0.0, r0 - r);
   }
};

#endif

