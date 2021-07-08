// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_LOR_MMS_HPP
#define MFEM_LOR_MMS_HPP

extern bool grad_div_problem;

namespace mfem
{

static constexpr double pi = M_PI, pi2 = M_PI*M_PI;

// Exact solution for definite Helmholtz problem with RHS corresponding to f
// defined below.
double u(const Vector &xvec)
{
   int dim = xvec.Size();
   double x = pi*xvec[0], y = pi*xvec[1];
   if (dim == 2) { return sin(x)*sin(y); }
   else { double z = pi*xvec[2]; return sin(x)*sin(y)*sin(z); }
}

double f(const Vector &xvec)
{
   int dim = xvec.Size();
   double x = pi*xvec[0], y = pi*xvec[1];

   if (dim == 2)
   {
      return sin(x)*sin(y) + 2*pi2*sin(x)*sin(y);
   }
   else // dim == 3
   {
      double z = pi*xvec[2];
      return sin(x)*sin(y)*sin(z) + 3*pi2*sin(x)*sin(y)*sin(z);
   }
}

// Exact solution for definite Maxwell and grad-div problems with RHS
// corresponding to f_vec below.
void u_vec(const Vector &xvec, Vector &u)
{
   int dim = xvec.Size();
   double x = pi*xvec[0], y = pi*xvec[1];
   if (dim == 2)
   {
      u[0] = cos(x)*sin(y);
      u[1] = sin(x)*cos(y);
   }
   else // dim == 3
   {
      double z = pi*xvec[2];
      u[0] = cos(x)*sin(y)*sin(z);
      u[1] = sin(x)*cos(y)*sin(z);
      u[2] = sin(x)*sin(y)*cos(z);
   }
}

void f_vec(const Vector &xvec, Vector &f)
{
   int dim = xvec.Size();
   double x = pi*xvec[0], y = pi*xvec[1];
   if (grad_div_problem)
   {
      if (dim == 2)
      {
         f[0] = (1 + 2*pi2)*cos(x)*sin(y);
         f[1] = (1 + 2*pi2)*cos(y)*sin(x);
      }
      else // dim == 3
      {
         double z = pi*xvec[2];
         f[0] = (1 + 3*pi2)*cos(x)*sin(y)*sin(z);
         f[1] = (1 + 3*pi2)*cos(y)*sin(x)*sin(z);
         f[2] = (1 + 3*pi2)*cos(z)*sin(x)*sin(y);
      }
   }
   else
   {
      if (dim == 2)
      {
         f[0] = cos(x)*sin(y);
         f[1] = sin(x)*cos(y);
      }
      else // dim == 3
      {
         double z = pi*xvec[2];
         f[0] = cos(x)*sin(y)*sin(z);
         f[1] = sin(x)*cos(y)*sin(z);
         f[2] = sin(x)*sin(y)*cos(z);
      }
   }
}

} // namespace mfem

#endif
