// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_PLASMA_HPP
#define MFEM_PLASMA_HPP

#include <cmath>
#include <complex>

namespace mfem
{

namespace plasma
{

// Physical Constants

// Permittivity of Free Space (units F/m)
static const real_t epsilon0_ = 8.8541878176e-12;

// Permeability of Free Space (units H/m)
static const real_t mu0_ = 4.0e-7 * M_PI;

// Speed of light in Free Space (units m/s)
static const real_t c0_ = 1.0 / sqrt(epsilon0_ * mu0_);

// Impedance of Free Space (units Ohm)
static const real_t Z0_ = sqrt(mu0_ / epsilon0_);

static const real_t q_     = 1.602176634e-19; // Elementary charge in coulombs
static const real_t eV_    = 1.602176634e-19; // 1 eV in Joules
static const real_t amu_   = 1.660539040e-27; // Atomic mass unit in kilograms
static const real_t me_kg_ = 9.10938356e-31;  // Mass of electron in kilograms
static const real_t me_u_  = 5.4857990907e-4; // Mass of electron in a.m.u

/**
   Returns the cyclotron frequency in radians/second
   m is the mass in a.m.u
   q is the charge in units of elementary electric charge
   B is the magnetic field magnitude in tesla
 */
inline real_t cyclotronFrequency(real_t B, real_t m, real_t q)
{
   return fabs(q * q_ * B / (m * amu_));
}

typedef std::complex<real_t> complex_t;

} // namespace plasma

} // namespace mfem

#endif // MFEM_PLASMA_HPP

