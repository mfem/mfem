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

static const double q_  = 1.60217662e-19;  // Elementary charge in coulombs
static const double u_  = 1.660539040e-27; // Atomic mass unit in kilograms
static const double me_ = 9.10938356e-31;  // Mass of electron in kilograms

} // namespace plasma

} // namespace mfem

#endif // MFEM_PLASMA_HPP
  
