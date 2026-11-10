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

#include <cmath>
#include "config/config.hpp"

#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport) __cdecl
#else
#define EXPORT extern "C"
#endif

namespace mfem
{
constexpr real_t pi = 3.14159265358979323846;
real_t kinvis = 1.0 / 40.0;
real_t reynolds = 1.0 / kinvis;
real_t lam = 0.5 * reynolds - sqrt(0.25 * reynolds * reynolds + 4.0 * pi * pi);

EXPORT real_t rho(real_t *coord, int dim, real_t t)
{
   return 1.0;
}

EXPORT real_t mu(real_t *coord, int dim, real_t t)
{
   return kinvis;
}

EXPORT void sol_u(real_t *coord, int dim, real_t t, real_t *u, int vdim)
{
   real_t xi = coord[0];
   real_t yi = coord[1];

   u[0] = 1.0 - exp(lam * xi) * cos(2.0 * pi * yi);
   u[1] = lam / (2.0 * pi) * exp(lam * xi) * sin(2.0 * pi * yi);
}

EXPORT void traction(real_t *coord, int dim, real_t t,
                     real_t *traction, int vdim)
{
   real_t xi = coord[0];
   real_t yi = coord[1];

   real_t dudx = - lam* exp(lam * xi) * cos(2.0 * pi * yi);
   real_t dudy =  2.0 * pi*exp(lam * xi) * sin(2.0 * pi * yi);

   real_t dvdx = lam / (2.0 * pi) * lam * exp(lam * xi) * sin(2.0 * pi * yi);

   traction[0] = 2* kinvis* dudx;
   traction[1] = kinvis*(dudy  + dvdx );
}

EXPORT real_t sol_p(real_t *coord, int dim, real_t t)
{
   real_t xi = coord[0];

   return 0.5 * (1.0 - exp(2.0 * lam * xi))  - 0.5 * (1.0 - exp(2.0 * lam * 1.0));
}

} // end namespace

