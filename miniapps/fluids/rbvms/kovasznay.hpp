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

#ifndef KOVASZNAY_FLOW_FACTORY_HPP
#define KOVASZNAY_FLOW_FACTORY_HPP

#include "mfem.hpp"
#include "problem.hpp"
#include <cmath>

class KovasznayFlow :  public CoefficientFactory
{
private:
   static constexpr mfem::real_t pi = 3.14159265358979323846;

   real_t mu = 0.01;
   real_t rho = 1.0;

   real_t reynolds;
   real_t lam;
   void Init()
   {
      reynolds = rho / mu;
      lam = 0.5 * reynolds - std::sqrt(0.25 * reynolds * reynolds + 4.0 * pi * pi);
   }
public:
   KovasznayFlow(int vdim) : CoefficientFactory(vdim), mu(1.0 / 40.0), rho(1.0)
   {
      Init();
   }
   KovasznayFlow(int vdim, real_t mu_) : CoefficientFactory(vdim), mu(mu_),
      rho(1.0)
   {
      Init();
   }
   // ~KovasznayFlow() =default;

   // Physical parameters
   mfem::real_t Density(const mfem::Vector &x) const override
   {
      return rho;
   }

   mfem::real_t Viscosity(const mfem::Vector &x) const override
   {
      return mu;
   }

   // Analytical solution: velocity
   void Velocity(const mfem::Vector &x, mfem::Vector &u) const override
   {
      const mfem::real_t xi = x[0];
      const mfem::real_t yi = x[1];

      u[0] = 1.0 - std::exp(lam * xi) * std::cos(2.0 * pi * yi);

      u[1] = lam / (2.0 * pi) *
             std::exp(lam * xi) * std::sin(2.0 * pi * yi);
   }

   // Analytical solution: pressure
   mfem::real_t Pressure(const mfem::Vector &x) const override
   {
      const mfem::real_t xi = x[0];
      return 0.5 * (1.0 - std::exp(2.0 * lam * xi))
             - 0.5 * (1.0 - std::exp(2.0 * lam));
   }

   // Traction
   void Traction(const mfem::Vector &x, mfem::Vector &traction) const override
   {
      const mfem::real_t xi = x[0];
      const mfem::real_t yi = x[1];

      const mfem::real_t exp_lam_x = std::exp(lam * xi);

      const mfem::real_t dudx =
         -lam * exp_lam_x * std::cos(2.0 * pi * yi);

      const mfem::real_t dudy =
         2.0 * pi * exp_lam_x * std::sin(2.0 * pi * yi);

      const mfem::real_t dvdx =
         lam * lam / (2.0 * pi) *
         exp_lam_x * std::sin(2.0 * pi * yi);

      traction[0] = 2.0 * mu * dudx;
      traction[1] = mu * (dudy + dvdx);
   }

};

#endif

