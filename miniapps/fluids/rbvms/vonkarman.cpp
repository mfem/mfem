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

EXPORT void sol_u(real_t *coord, int dim, real_t t, real_t *u, int vdim)
{
   real_t x = coord[0];
   real_t y = coord[1];

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

EXPORT void traction(real_t *coord, int dim, real_t t,
                     real_t *traction, int vdim)
{
   traction[0] = 0.0;
   traction[1] = 0.0;
}

EXPORT real_t sol_p(real_t *coord, int dim, real_t t)
{
   return 0;
}

} // end namespace

