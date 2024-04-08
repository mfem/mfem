// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// Finite Element classes on Pyramid shaped elements

#include "fe_pyramid.hpp"

namespace mfem
{

using namespace std;

void FuentesPyramid::grad_lam1(const double x, const double y,
                               const double z, double du[])
{
   du[0] = (z < 1.0) ? - (1.0 - y - z) / (1.0 - z) : 0.0;
   du[1] = (z < 1.0) ? - (1.0 - x - z) / (1.0 - z) : 0.0;
   du[2] = (z < 1.0) ? x * y / ((1.0 - z) * (1.0 - z)) - 1.0 : 0.0;
}

void FuentesPyramid::grad_lam2(const double x, const double y,
                               const double z, double du[])
{
   du[0] = (z < 1.0) ? (1.0 - y - z) / (1.0 - z) : 0.0;
   du[1] = (z < 1.0) ? - x / (1.0 - z) : 0.0;
   du[2] = (z < 1.0) ? - x * y / ((1.0 - z) * (1.0 - z)) : 0.0;
}

void FuentesPyramid::grad_lam3(const double x, const double y,
                               const double z, double du[])
{
   du[0] = (z < 1.0) ? y / (1.0 - z) : 0.0;
   du[1] = (z < 1.0) ? x / (1.0 - z) : 0.0;
   du[2] = (z < 1.0) ? x * y / ((1.0 - z) * (1.0 - z)) : 0.0;
}

void FuentesPyramid::grad_lam4(const double x, const double y,
                               const double z, double du[])
{
   du[0] = (z < 1.0) ? - y / (1.0 - z) : 0.0;
   du[1] = (z < 1.0) ? (1.0 - x - z) / (1.0 - z) : 0.0;
   du[2] = (z < 1.0) ? - x * y / ((1.0 - z) * (1.0 - z)) : 0.0;
}

void FuentesPyramid::grad_lam5(const double x, const double y,
                               const double z, double du[])
{
   du[0] = 0.0;
   du[1] = 0.0;
   du[2] = 1.0;
}

void FuentesPyramid::phi_E(const int p, const double s0, double s1,
                           double *u)
{
   CalcIntegratedLegendre(p, s1, s0 + s1, u);
}

void FuentesPyramid::phi_E(const int p, const double s0, double s1,
                           double *u, double *duds0, double *duds1)
{
   CalcIntegratedLegendre(p, s1, s0 + s1, u, duds1, duds0);
   for (int i = 0; i <= p; i++) { duds1[i] += duds0[i]; }
}

void FuentesPyramid::CalcIntegratedLegendre(const int p, const double x,
                                            const double t,
                                            double *u)
{
   if (t > 0.0)
   {
      CalcScaledLegendre(p, x, t, u);
      for (int i = p; i >= 2; i--)
      {
         u[i] = (u[i] - t * t * u[i-2]) / (4.0 * i - 2.0);
      }
      if (p >= 1)
      {
         u[1] = x;
      }
      u[0] = 0.0;
   }
   else
   {
      for (int i = 0; i <= p; i++)
      {
         u[i] = 0.0;
      }
   }
}

void FuentesPyramid::CalcIntegratedLegendre(const int p, const double x,
                                            const double t,
                                            double *u,
                                            double *dudx, double *dudt)
{
   if (t > 0.0)
   {
      CalcScaledLegendre(p, x, t, u, dudx, dudt);
      for (int i = p; i >= 2; i--)
      {
         u[i] = (u[i] - t * t * u[i-2]) / (4.0 * i - 2.0);
         dudx[i] = (dudx[i] - t * t * dudx[i-2]) / (4.0 * i - 2.0);
         dudt[i] = (dudt[i] - t * t * dudt[i-2] - 2.0 * t * u[i-2]) /
                   (4.0 * i - 2.0);
      }
      if (p >= 1)
      {
         u[1] = x; dudx[1] = 1.0; dudt[1] = 0.0;
      }
      u[0] = 0.0; dudx[0] = 0.0; dudt[0] = 0.0;
   }
   else
   {
      for (int i = 0; i <= p; i++)
      {
         u[i] = 0.0;
         dudx[i] = 0.0;
         dudt[i] = 0.0;
      }
   }
}

/** Implements a scaled and shifted set of Legendre polynomials

      P_i(x / t) * t^i

   where t >= 0.0, x \in [0,t], and P_i is the shifted Legendre
   polynomial defined on [0,1] rather than the usual [-1,1].
*/
void FuentesPyramid::CalcScaledLegendre(const int p, const double x,
                                        const double t,
                                        double *u)
{
   if (t > 0.0)
   {
      Poly_1D::CalcLegendre(p, x / t, u);
      for (int i = 1; i <= p; i++)
      {
         u[i] *= pow(t, i);
      }
   }
   else
   {
      // This assumes x = 0 as well as t = 0 since x \in [0,t]
      u[0] = 1.0;
      for (int i = 1; i <= p; i++) { u[i] = 0.0; }
   }
}

void FuentesPyramid::CalcScaledLegendre(const int p, const double x,
                                        const double t,
                                        double *u,
                                        double *dudx, double *dudt)
{
   if (t > 0.0)
   {
      Poly_1D::CalcLegendre(p, x / t, u, dudx);
      dudx[0] = 0.0;
      dudt[0] = - dudx[0] * x / t;
      for (int i = 1; i <= p; i++)
      {
         u[i]    *= pow(t, i);
         dudx[i] *= pow(t, i - 1);
         dudt[i]  = (u[i] * i - dudx[i] * x) / t;
      }
   }
   else
   {
      // This assumes x = 0 as well as t = 0 since x \in [0,t]
      u[0]    = 1.0;
      dudx[0] = 0.0;
      dudt[0] = 0.0;
      if (p >=1)
      {
         u[1]    =  0.0;
         dudx[1] =  2.0;
         dudt[1] = -1.0;
      }
      for (int i = 2; i <= p; i++)
      {
         u[i] = 0.0;
         dudx[i] = 0.0;
         dudt[i] = 0.0;
      }
   }
}

void FuentesPyramid::CalcIntegratedJacobi(const int p,
                                          const double alpha,
                                          const double x,
                                          const double t,
                                          double *u)
{
   if (t > 0.0)
   {
      CalcScaledJacobi(p, alpha, x, t, u);
      for (int i = p; i >= 2; i--)
      {
         double d0 = 2.0 * i + alpha;
         double d1 = d0 - 1.0;
         double d2 = d0 - 2.0;
         double a = (alpha + i) / (d0 * d1);
         double b = alpha / (d0 * d2);
         double c = (double)(i - 1) / (d1 * d2);
         u[i] = a * u[i] + b * t * u[i - 1] - c * t * t * u[i - 2];
      }
      if (p >= 1)
      {
         u[1] = x;
      }
      u[0] = 0.0;
   }
   else
   {
      u[0] = 1.0;
      for (int i = 1; i <= p; i++)
      {
         u[i] = 0.0;
      }
   }
}

void FuentesPyramid::CalcIntegratedJacobi(const int p,
                                          const double alpha,
                                          const double x,
                                          const double t,
                                          double *u,
                                          double *dudx,
                                          double *dudt)
{
   CalcScaledJacobi(p, alpha, x, t, u, dudx, dudt);
   for (int i = p; i >= 2; i--)
   {
      double d0 = 2.0 * i + alpha;
      double d1 = d0 - 1.0;
      double d2 = d0 - 2.0;
      double a = (alpha + i) / (d0 * d1);
      double b = alpha / (d0 * d2);
      double c = (double)(i - 1) / (d1 * d2);
      u[i]    = a * u[i] + b * t * u[i - 1] - c * t * t * u[i - 2];
      dudx[i] = a * dudx[i] + b * t * dudx[i - 1] - c * t * t * dudx[i - 2];
      dudt[i] = a * dudt[i] + b * t * dudt[i - 1] + b * u[i - 1]
                - c * t * t * dudt[i - 2] - 2.0 * c * t * u[i - 2];
   }
   if (p >= 1)
   {
      u[1]    = x;
      dudx[1] = 1.0;
      dudt[1] = 0.0;
   }
   u[0]    = 0.0;
   dudx[0] = 0.0;
   dudt[0] = 0.0;
}

/** Implements a set of scaled and shifted subset of Jacobi polynomials

      P_i^{\alpha, 0}(x / t) * t^i

   where t >= 0.0, x \in [0,t], and P_i^{\alpha, \beta} is the shifted Jacobi
   polynomial defined on [0,1] rather than the usual [-1,1]. Note that we only
   consider the special case when \beta = 0.
*/
void FuentesPyramid::CalcScaledJacobi(const int p, const double alpha,
                                      const double x,
                                      const double t,
                                      double *u)
{
   u[0] = 1.0;
   if (p >= 1)
   {
      u[1] = (2.0 + alpha) * x - t;
   }
   for (int i = 2; i <= p; i++)
   {
      double a = 2.0 * i * (alpha + i) * (2.0 * i + alpha - 2.0);
      double b = 2.0 * i + alpha - 1.0;
      double c = (2.0 * i + alpha) * (2.0 * i + alpha - 2.0);
      double d = 2.0 * (alpha + i - 1.0) * (i - 1) * (2.0 * i - alpha);
      u[i] = (b * (c * (2.0 * x - t) + alpha * alpha * t) * u[i - 1]
              - d * t * t * u[i - 2]) / a;
   }
}

void FuentesPyramid::CalcScaledJacobi(const int p, const double alpha,
                                      const double x,
                                      const double t,
                                      double *u, double *dudx, double *dudt)
{
   u[0]    = 1.0;
   dudx[0] = 0.0;
   dudt[0] = 0.0;
   if (p >= 1)
   {
      u[1]    = (2.0 + alpha) * x - t;
      dudx[1] =  2.0 + alpha;
      dudt[1] = -1.0;
   }
   for (int i = 2; i <= p; i++)
   {
      double a = 2.0 * i * (alpha + i) * (2.0 * i + alpha - 2.0);
      double b = 2.0 * i + alpha - 1.0;
      double c = (2.0 * i + alpha) * (2.0 * i + alpha - 2.0);
      double d = 2.0 * (alpha + i - 1.0) * (i - 1) * (2.0 * i - alpha);
      u[i] = (b * (c * (2.0 * x - t) + alpha * alpha * t) * u[i - 1]
              - d * t * t * u[i - 2]) / a;
      dudx[i] = (b * ((c * (2.0 * x - t) + alpha * alpha * t) * dudx[i - 1] +
                      2.0 * c * u[i - 1])
                 - d * t * t * dudx[i - 2]) / a;
      dudt[i] = (b * ((c * (2.0 * x - t) + alpha * alpha * t) * dudt[i - 1] +
                      (alpha * alpha - c) * u[i - 1])
                 - d * t * t * dudt[i - 2] - 2.0 * d * t * u[i - 2]) / a;
   }
}

} // namespace mfem
