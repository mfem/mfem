// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_PLASMA_HPP
#define MFEM_PLASMA_HPP

#include <cmath>

namespace mfem
{

namespace plasma
{

// Physical Constants

// Permittivity of Free Space (units F/m)
static const double epsilon0_ = 8.8541878176e-12;

// Permeability of Free Space (units H/m)
static const double mu0_ = 4.0e-7 * M_PI;

// Speed of light in Free Space (units m/s)
static const double c0_ = 1.0 / sqrt(epsilon0_ * mu0_);

static const double q_     = 1.602176634e-19; // Elementary charge in coulombs
static const double amu_   = 1.660539040e-27; // Atomic mass unit in kilograms
static const double me_kg_ = 9.10938356e-31;  // Mass of electron in kilograms
static const double me_u_  = 5.4857990907e-4; // Mass of electron in a.m.u

/**
   Returns the mean Electron-Ion collision time in seconds
   Te is the electron temperature in eV
   ni is the density of ions (assuming ni=ne) in particles per meter^3
   zi is the charge number of the ion species
   lnLambda is the Coulomb Logarithm
*/
inline double meanElectronIonCollisionTime(double Te, double ni, int zi,
                                           double lnLambda)
{
   // The factor of q_^{3/2} is included to convert Te from eV to Joules
   return 0.75 * pow(4.0 * M_PI * epsilon0_, 2) *
          sqrt(0.5 * me_kg_ * pow(q_ * Te, 3) / M_PI) /
          (lnLambda * pow(q_, 4) * zi * zi * ni);
}

/**
   Returns the mean Ion-Ion collision time in seconds
   In the following 'a' refers to the species whose mean collision time is
   being computed and 'b' refers to the species being collided with.
   ma is the ion mass in a.m.u.
   Ta is the ion temperature in eV
   nb is the density of ions in particles per meter^3
   za is the charge number of the ion species
   zb is the charge number of the ion species
   lnLambda is the Coulomb Logarithm
*/
inline double meanIonIonCollisionTime(double ma, double Ta, double nb,
                                      int za, int zb, double lnLambda)
{
   // The factor of q_^{3/2} is included to convert Ti from eV to Joules
   return 0.75 * pow(4.0 * M_PI * epsilon0_, 2) *
          sqrt(ma * amu_ * pow(q_ * Ta, 3) / M_PI) /
          (lnLambda * pow(q_, 4) * za * za * zb * zb * nb);
}

} // namespace plasma

} // namespace mfem

#endif // MFEM_PLASMA_HPP

