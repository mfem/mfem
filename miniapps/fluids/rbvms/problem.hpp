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

#ifndef COEFFICIENT_FACTORY_HPP
#define COEFFICIENT_FACTORY_HPP

#include "mfem.hpp"

class CoefficientFactory
{
public:
   FunctionCoefficient DensityCoefficient;
   FunctionCoefficient ViscosityCoefficient;
   FunctionCoefficient PressureCoefficient;
   VectorFunctionCoefficient VelocityCoefficient;
   VectorFunctionCoefficient TractionCoefficient;
   VectorFunctionCoefficient ForceCoefficient;

   CoefficientFactory(int vdim) :
      DensityCoefficient( [this](const mfem::Vector &x)
   {
      return Density(x);
   }),
   ViscosityCoefficient( [this](const mfem::Vector &x)
   {
      return Viscosity(x);
   }),
   PressureCoefficient([this](const mfem::Vector &x)
   {
      return Pressure(x);
   }),
   VelocityCoefficient(vdim, [this](const mfem::Vector &x, mfem::Vector &u)
   {
      Velocity(x, u);
   }),
   TractionCoefficient(vdim, [this](const mfem::Vector &x, mfem::Vector &traction)
   {
      Traction(x, traction);
   }),
   ForceCoefficient(vdim, [this](const mfem::Vector &x, mfem::Vector &force)
   {
      Force(x, force);
   })
   {}

   virtual ~CoefficientFactory() = default;

   // Analytical solution interface
   virtual mfem::real_t Density(const mfem::Vector &x) const
   {
      return 1.0;
   }

   virtual mfem::real_t Viscosity(const mfem::Vector &x) const
   {
      return 0.0;
   }

   virtual mfem::real_t Pressure(const mfem::Vector &x) const
   {
      return 0.0;
   }

   virtual void Velocity(const mfem::Vector &x,
                         mfem::Vector &u) const
   {
      u = 0.0;
   }

   virtual void Traction(const mfem::Vector &x,
                         mfem::Vector &traction) const
   {
      traction = 0.0;
   }

   virtual void Force(const mfem::Vector &x,
                      mfem::Vector &force) const
   {
      force = 0.0;
   }


};

#endif

