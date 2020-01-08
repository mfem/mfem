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
static const double eV_    = 1.602176634e-19; // 1 eV in Joules
static const double amu_   = 1.660539040e-27; // Atomic mass unit in kilograms
static const double me_kg_ = 9.10938356e-31;  // Mass of electron in kilograms
static const double me_u_  = 5.4857990907e-4; // Mass of electron in a.m.u

/**
   Returns the cyclotron frequency in radians/second
   m is the mass in a.m.u
   q is the charge in units of elementary electric charge
   B is the magnetic field magnitude in tesla
 */
inline double cyclotronFrequency(double B, double m, double q)
{
   return fabs(q * q_ * B / (m * amu_));
}

} // namespace plasma

} // namespace mfem

#endif // MFEM_PLASMA_HPP

