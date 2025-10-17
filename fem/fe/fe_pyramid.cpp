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

// Finite Element classes on Pyramid shaped elements

#include "fe_pyramid.hpp"

namespace mfem
{

using namespace std;

Vector FuentesPyramid::grad_lam1(real_t x, real_t y, real_t z)
{
   return Vector({CheckZ(z) ? - (1.0 - y - z) / (1.0 - z) : -0.5,
                  CheckZ(z) ? - (1.0 - x - z) / (1.0 - z) : -0.5,
                  CheckZ(z) ? x * y / ((1.0 - z) * (1.0 - z)) - 1.0 : -0.75});
}

Vector FuentesPyramid::grad_lam2(real_t x, real_t y, real_t z)
{
   return Vector({CheckZ(z) ? (1.0 - y - z) / (1.0 - z) : 0.5,
                  CheckZ(z) ? - x / (1.0 - z) : -0.5,
                  CheckZ(z) ? - x * y / ((1.0 - z) * (1.0 - z)) : -0.25});
}

Vector FuentesPyramid::grad_lam3(real_t x, real_t y, real_t z)
{
   return Vector({CheckZ(z) ? y / (1.0 - z) : 0.5,
                  CheckZ(z) ? x / (1.0 - z) : 0.5,
                  CheckZ(z) ? x * y / ((1.0 - z) * (1.0 - z)) : 0.25});
}

Vector FuentesPyramid::grad_lam4(real_t x, real_t y, real_t z)
{
   return Vector({CheckZ(z) ? - y / (1.0 - z) : -0.5,
                  CheckZ(z) ? (1.0 - x - z) / (1.0 - z) : 0.5,
                  CheckZ(z) ? - x * y / ((1.0 - z) * (1.0 - z)) : -0.25});
}

Vector FuentesPyramid::grad_lam5(real_t x, real_t y, real_t z)
{
   return Vector({0.0, 0.0, 1.0});
}

DenseMatrix FuentesPyramid::grad_lam15(real_t x, real_t y, real_t z)
{
   DenseMatrix dlam(2, 3);
   dlam.SetRow(0, grad_lam1(x, y, z));
   dlam.SetRow(1, grad_lam5(x, y, z));
   return dlam;
}

DenseMatrix FuentesPyramid::grad_lam25(real_t x, real_t y, real_t z)
{
   DenseMatrix dlam(2, 3);
   dlam.SetRow(0, grad_lam2(x, y, z));
   dlam.SetRow(1, grad_lam5(x, y, z));
   return dlam;
}

DenseMatrix FuentesPyramid::grad_lam35(real_t x, real_t y, real_t z)
{
   DenseMatrix dlam(2, 3);
   dlam.SetRow(0, grad_lam3(x, y, z));
   dlam.SetRow(1, grad_lam5(x, y, z));
   return dlam;
}

DenseMatrix FuentesPyramid::grad_lam45(real_t x, real_t y, real_t z)
{
   DenseMatrix dlam(2, 3);
   dlam.SetRow(0, grad_lam4(x, y, z));
   dlam.SetRow(1, grad_lam5(x, y, z));
   return dlam;
}

Vector FuentesPyramid::lam15_grad_lam15(real_t x, real_t y, real_t z)
{
   Vector lam = lam15(x, y, z);
   Vector lamdlam(3);
   add(lam(0), grad_lam5(x, y, z), -lam(1), grad_lam1(x, y, z), lamdlam);
   return lamdlam;
}

Vector FuentesPyramid::lam25_grad_lam25(real_t x, real_t y, real_t z)
{
   Vector lam = lam25(x, y, z);
   Vector lamdlam(3);
   add(lam(0), grad_lam5(x, y, z), -lam(1), grad_lam2(x, y, z), lamdlam);
   return lamdlam;
}

Vector FuentesPyramid::lam35_grad_lam35(real_t x, real_t y, real_t z)
{
   Vector lam = lam35(x, y, z);
   Vector lamdlam(3);
   add(lam(0), grad_lam5(x, y, z), -lam(1), grad_lam3(x, y, z), lamdlam);
   return lamdlam;
}

Vector FuentesPyramid::lam45_grad_lam45(real_t x, real_t y, real_t z)
{
   Vector lam = lam45(x, y, z);
   Vector lamdlam(3);
   add(lam(0), grad_lam5(x, y, z), -lam(1), grad_lam4(x, y, z), lamdlam);
   return lamdlam;
}

Vector FuentesPyramid::lam125_grad_lam125(real_t x, real_t y, real_t z)
{
   Vector lgl({-x * z / (one - z), y - one, z});
   lgl *= (one - y - z) / (one - z);
   return lgl;
}

Vector FuentesPyramid::lam235_grad_lam235(real_t x, real_t y, real_t z)
{
   Vector lgl({x, -y * z / (one - z), z});
   lgl *= x / (one - z);
   return lgl;
}

Vector FuentesPyramid::lam345_grad_lam345(real_t x, real_t y, real_t z)
{
   Vector lgl({-x * z / (one - z), y, z});
   lgl *= y / (one - z);
   return lgl;
}

Vector FuentesPyramid::lam435_grad_lam435(real_t x, real_t y, real_t z)
{
   Vector lgl({x * z / (one - z), -y, -z});
   lgl *= y / (one - z);
   return lgl;
}

Vector FuentesPyramid::lam415_grad_lam415(real_t x, real_t y, real_t z)
{
   Vector lgl({x - one, -y * z / (one - z), z});
   lgl *= (one - x - z) / (one - z);
   return lgl;
}

Vector FuentesPyramid::lam145_grad_lam145(real_t x, real_t y, real_t z)
{
   Vector lgl({one - x, y * z / (one - z), -z});
   lgl *= (one - x - z) / (one - z);
   return lgl;
}

real_t FuentesPyramid::div_lam125_grad_lam125(real_t x, real_t y, real_t z)
{ return (1.0 - z - y) / (1.0 - z); }

real_t FuentesPyramid::div_lam235_grad_lam235(real_t x, real_t y, real_t z)
{ return x / (1.0 - z); }

real_t FuentesPyramid::div_lam345_grad_lam345(real_t x, real_t y, real_t z)
{ return y / (1.0 - z); }

real_t FuentesPyramid::div_lam435_grad_lam435(real_t x, real_t y, real_t z)
{ return -y / (1.0 - z); }

real_t FuentesPyramid::div_lam415_grad_lam415(real_t x, real_t y, real_t z)
{ return (1.0 - z - x) / (1.0 - z); }

real_t FuentesPyramid::div_lam145_grad_lam145(real_t x, real_t y, real_t z)
{ return -(1.0 - z - x) / (1.0 - z); }

DenseMatrix FuentesPyramid::grad_mu01(real_t z)
{
   DenseMatrix dmu(2, 3);
   dmu.SetRow(0, grad_mu0(z));
   dmu.SetRow(1, grad_mu1(z));
   return dmu;
}

Vector FuentesPyramid::grad_mu0(real_t z, const Vector xy, unsigned int ab)
{
   Vector dmu({0.0, 0.0, - xy[ab-1] / pow(1.0 - z, 2)});
   dmu[ab-1] = -1.0 / (1.0 - z);
   return dmu;
}

Vector FuentesPyramid::grad_mu1(real_t z, const Vector xy, unsigned int ab)
{
   Vector dmu({0.0, 0.0, xy[ab-1] / pow(1.0 - z, 2)});
   dmu[ab-1] = 1.0 / (1.0 - z);
   return dmu;
}

DenseMatrix FuentesPyramid::grad_mu01(real_t z, Vector xy, unsigned int ab)
{
   DenseMatrix dmu(2, 3);
   dmu.SetRow(0, grad_mu0(z, xy, ab));
   dmu.SetRow(1, grad_mu1(z, xy, ab));
   return dmu;
}

Vector FuentesPyramid::mu01_grad_mu01(real_t z, Vector xy, unsigned int ab)
{
   Vector mu = mu01(z, xy, ab);
   Vector mudmu(3);
   add(mu(0), grad_mu1(z, xy, ab), -mu(1), grad_mu0(z, xy, ab), mudmu);
   return mudmu;
}

Vector FuentesPyramid::grad_nu0(real_t z, const Vector xy, unsigned int ab)
{
   Vector dnu({0.0, 0.0, -1.0}); dnu[ab-1] = -1.0;
   return dnu;
}

Vector FuentesPyramid::grad_nu1(real_t z, const Vector xy, unsigned int ab)
{
   Vector dnu({0.0, 0.0, 0.0}); dnu[ab-1] = 1.0;
   return dnu;
}

Vector FuentesPyramid::grad_nu2(real_t z, const Vector xy, unsigned int ab)
{
   return Vector({0.0, 0.0, 1.0});
}

DenseMatrix FuentesPyramid::grad_nu01(real_t z, Vector xy, unsigned int ab)
{
   DenseMatrix dnu(2, 3);
   dnu.SetRow(0, grad_nu0(z, xy, ab));
   dnu.SetRow(1, grad_nu1(z, xy, ab));
   return dnu;
}

DenseMatrix FuentesPyramid::grad_nu012(real_t z, Vector xy, unsigned int ab)
{
   DenseMatrix dnu(3, 3);
   dnu.SetRow(0, grad_nu0(z, xy, ab));
   dnu.SetRow(1, grad_nu1(z, xy, ab));
   dnu.SetRow(2, grad_nu2(z, xy, ab));
   return dnu;
}

DenseMatrix FuentesPyramid::grad_nu120(real_t z, Vector xy, unsigned int ab)
{
   DenseMatrix dnu(3, 3);
   dnu.SetRow(0, grad_nu1(z, xy, ab));
   dnu.SetRow(1, grad_nu2(z, xy, ab));
   dnu.SetRow(2, grad_nu0(z, xy, ab));
   return dnu;
}

Vector FuentesPyramid::nu01_grad_nu01(real_t z, Vector xy, unsigned int ab)
{
   Vector nu = nu01(z, xy, ab);
   Vector nudnu(3);
   add(nu(0), grad_nu1(z, xy, ab), -nu(1), grad_nu0(z, xy, ab), nudnu);
   return nudnu;
}

Vector FuentesPyramid::nu12_grad_nu12(real_t z, Vector xy, unsigned int ab)
{
   Vector nu = nu12(z, xy, ab);
   Vector nudnu(3);
   add(nu(0), grad_nu2(z, xy, ab), -nu(1), grad_nu1(z, xy, ab), nudnu);
   return nudnu;
}

Vector FuentesPyramid::nu012_grad_nu012(real_t z, Vector xy, unsigned int ab)
{
   Vector nu(nu012(z, xy, ab));
   Vector dnu0(grad_nu0(z, xy, ab));
   Vector dnu1(grad_nu1(z, xy, ab));
   Vector dnu2(grad_nu2(z, xy, ab));

   Vector v01(3), v12(3), v20(3);
   dnu0.cross3D(dnu1, v01);
   dnu1.cross3D(dnu2, v12);
   dnu2.cross3D(dnu0, v20);

   Vector nudnu(3);
   add(nu(0), v12, nu(1), v20, nudnu);
   nudnu.Add(nu(2), v01);
   return nudnu;
}

void FuentesPyramid::CalcScaledLegendre(int p, real_t x, real_t t,
                                        real_t *u)
{
   MFEM_ASSERT(p >= 0, "Polynomial order must be zero or larger");
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

void FuentesPyramid::CalcScaledLegendre(int p, real_t x, real_t t,
                                        real_t *u,
                                        real_t *dudx, real_t *dudt)
{
   MFEM_ASSERT(p >= 0, "Polynomial order must be zero or larger");
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

void FuentesPyramid::CalcScaledLegendre(int p, real_t x, real_t t,
                                        Vector &u)
{
   MFEM_ASSERT(p >= 0, "Polynomial order must be zero or larger");
   MFEM_ASSERT(u.Size() >= p+1, "Size of u is too small");
   CalcScaledLegendre(p, x, t, u.GetData());
}

void FuentesPyramid::CalcScaledLegendre(int p, real_t x, real_t t,
                                        Vector &u, Vector &dudx, Vector &dudt)
{
   MFEM_ASSERT(p >= 0, "Polynomial order must be zero or larger");
   MFEM_ASSERT(u.Size() >= p+1, "Size of u is too small");
   MFEM_ASSERT(dudx.Size() >= p+1, "Size of dudx is too small");
   MFEM_ASSERT(dudt.Size() >= p+1, "Size of dudt is too small");
   CalcScaledLegendre(p, x, t, u.GetData(), dudx.GetData(), dudt.GetData());
}

void FuentesPyramid::CalcIntegratedLegendre(int p, real_t x, real_t t,
                                            real_t *u)
{
   MFEM_ASSERT(p >= 0, "Polynomial order must be zero or larger");
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

void FuentesPyramid::CalcIntegratedLegendre(int p, real_t x, real_t t,
                                            real_t *u,
                                            real_t *dudx, real_t *dudt)
{
   MFEM_ASSERT(p >= 0, "Polynomial order must be zero or larger");
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

void FuentesPyramid::CalcIntegratedLegendre(int p, real_t x,
                                            real_t t, Vector &u)
{
   MFEM_ASSERT(p >= 0, "Polynomial order must be zero or larger");
   MFEM_ASSERT(u.Size() >= p+1, "Size of u is too small");
   CalcIntegratedLegendre(p, x, t, u.GetData());
}

void FuentesPyramid::CalcIntegratedLegendre(int p, real_t x,
                                            real_t t, Vector &u,
                                            Vector &dudx, Vector &dudt)
{
   MFEM_ASSERT(p >= 0, "Polynomial order must be zero or larger");
   MFEM_ASSERT(u.Size() >= p+1, "Size of u is too small");
   MFEM_ASSERT(dudx.Size() >= p+1, "Size of dudx is too small");
   MFEM_ASSERT(dudt.Size() >= p+1, "Size of dudt is too small");
   CalcIntegratedLegendre(p, x, t, u.GetData(),
                          dudx.GetData(), dudt.GetData());
}

void FuentesPyramid::CalcHomogenizedScaLegendre(int p, real_t s0, real_t s1,
                                                real_t *u)
{
   MFEM_ASSERT(p >= 0, "Polynomial order must be zero or larger");
   CalcScaledLegendre(p, s1, s0 + s1, u);
}

void FuentesPyramid::CalcHomogenizedScaLegendre(int p,
                                                real_t s0, real_t s1,
                                                real_t *u,
                                                real_t *duds0, real_t *duds1)
{
   MFEM_ASSERT(p >= 0, "Polynomial order must be zero or larger");
   CalcScaledLegendre(p, s1, s0+s1, u, duds1, duds0);
   for (int i = 0; i <= p; i++) { duds1[i] += duds0[i]; }
}

void FuentesPyramid::CalcHomogenizedScaLegendre(int p, real_t s0, real_t s1,
                                                Vector &u)
{
   MFEM_ASSERT(p >= 0, "Polynomial order must be zero or larger");
   MFEM_ASSERT(u.Size() >= p+1, "Size of u is too small");
   CalcHomogenizedScaLegendre(p, s0, s1, u.GetData());
}

void FuentesPyramid::CalcHomogenizedScaLegendre(int p,
                                                real_t s0, real_t s1,
                                                Vector &u,
                                                Vector &duds0, Vector &duds1)
{
   MFEM_ASSERT(p >= 0, "Polynomial order must be zero or larger");
   MFEM_ASSERT(u.Size() >= p+1, "Size of u is too small");
   MFEM_ASSERT(duds0.Size() >= p+1, "Size of duds0 is too small");
   MFEM_ASSERT(duds1.Size() >= p+1, "Size of duds1 is too small");
   CalcHomogenizedScaLegendre(p, s0, s1, u.GetData(),
                              duds0.GetData(), duds1.GetData());
}

void FuentesPyramid::CalcHomogenizedIntLegendre(int p,
                                                real_t t0, real_t t1,
                                                real_t *u)
{
   MFEM_ASSERT(p >= 0, "Polynomial order must be zero or larger");
   CalcIntegratedLegendre(p, t1, t0 + t1, u);
}

void FuentesPyramid::CalcHomogenizedIntLegendre(int p,
                                                real_t t0, real_t t1,
                                                real_t *u,
                                                real_t *dudt0, real_t *dudt1)
{
   MFEM_ASSERT(p >= 0, "Polynomial order must be zero or larger");
   CalcIntegratedLegendre(p, t1, t0+t1, u, dudt1, dudt0);
   for (int i = 0; i <= p; i++) { dudt1[i] += dudt0[i]; }
}

void FuentesPyramid::CalcHomogenizedIntLegendre(int p,
                                                real_t t0, real_t t1,
                                                Vector &u)
{
   MFEM_ASSERT(p >= 0, "Polynomial order must be zero or larger");
   MFEM_ASSERT(u.Size() >= p+1, "Size of u is too small");
   CalcHomogenizedIntLegendre(p, t0, t1, u.GetData());
}

void FuentesPyramid::CalcHomogenizedIntLegendre(int p,
                                                real_t t0, real_t t1,
                                                Vector &u,
                                                Vector &dudt0, Vector &dudt1)
{
   MFEM_ASSERT(p >= 0, "Polynomial order must be zero or larger");
   MFEM_ASSERT(u.Size() >= p+1, "Size of u is too small");
   MFEM_ASSERT(dudt0.Size() >= p+1, "Size of dudt0 is too small");
   MFEM_ASSERT(dudt1.Size() >= p+1, "Size of dudt1 is too small");
   CalcHomogenizedIntLegendre(p, t0, t1, u.GetData(),
                              dudt0.GetData(), dudt1.GetData());
}

void FuentesPyramid::CalcIntegratedJacobi(int p, real_t alpha,
                                          real_t x, real_t t,
                                          real_t *u)
{
   MFEM_ASSERT(p >= 0, "Polynomial order must be zero or larger");
   if (t > 0.0)
   {
      CalcScaledJacobi(p, alpha, x, t, u);
      for (int i = p; i >= 2; i--)
      {
         real_t d0 = 2.0 * i + alpha;
         real_t d1 = d0 - 1.0;
         real_t d2 = d0 - 2.0;
         real_t a = (alpha + i) / (d0 * d1);
         real_t b = alpha / (d0 * d2);
         real_t c = (real_t)(i - 1) / (d1 * d2);
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

void FuentesPyramid::CalcIntegratedJacobi(int p, real_t alpha,
                                          real_t x, real_t t,
                                          real_t *u,
                                          real_t *dudx,
                                          real_t *dudt)
{
   MFEM_ASSERT(p >= 0, "Polynomial order must be zero or larger");
   CalcScaledJacobi(p, alpha, x, t, u, dudx, dudt);
   for (int i = p; i >= 2; i--)
   {
      real_t d0 = 2.0 * i + alpha;
      real_t d1 = d0 - 1.0;
      real_t d2 = d0 - 2.0;
      real_t a = (alpha + i) / (d0 * d1);
      real_t b = alpha / (d0 * d2);
      real_t c = (real_t)(i - 1) / (d1 * d2);
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

void FuentesPyramid::CalcScaledJacobi(int p, real_t alpha,
                                      real_t x, real_t t,
                                      real_t *u)
{
   MFEM_ASSERT(p >= 0, "Polynomial order must be zero or larger");

   u[0] = 1.0;
   if (p >= 1)
   {
      u[1] = (2.0 + alpha) * x - t;
   }
   for (int i = 2; i <= p; i++)
   {
      real_t a = 2.0 * i * (alpha + i) * (2.0 * i + alpha - 2.0);
      real_t b = 2.0 * i + alpha - 1.0;
      real_t c = (2.0 * i + alpha) * (2.0 * i + alpha - 2.0);
      real_t d = 2.0 * (alpha + i - 1.0) * (i - 1) * (2.0 * i + alpha);
      u[i] = (b * (c * (2.0 * x - t) + alpha * alpha * t) * u[i - 1]
              - d * t * t * u[i - 2]) / a;
   }
}

void FuentesPyramid::CalcScaledJacobi(int p, real_t alpha,
                                      real_t x, real_t t,
                                      real_t *u, real_t *dudx, real_t *dudt)
{
   MFEM_ASSERT(p >= 0, "Polynomial order must be zero or larger");

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
      real_t a = 2.0 * i * (alpha + i) * (2.0 * i + alpha - 2.0);
      real_t b = 2.0 * i + alpha - 1.0;
      real_t c = (2.0 * i + alpha) * (2.0 * i + alpha - 2.0);
      real_t d = 2.0 * (alpha + i - 1.0) * (i - 1) * (2.0 * i + alpha);
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

void FuentesPyramid::CalcScaledJacobi(int p, real_t alpha,
                                      real_t x, real_t t,
                                      Vector &u)
{
   MFEM_ASSERT(p >= 0, "Polynomial order must be zero or larger");
   MFEM_ASSERT(u.Size() >= p+1, "Size of u is too small");
   CalcScaledJacobi(p, alpha, x, t, u.GetData());
}

void FuentesPyramid::CalcScaledJacobi(int p, real_t alpha,
                                      real_t x, real_t t,
                                      Vector &u, Vector &dudx, Vector &dudt)
{
   MFEM_ASSERT(p >= 0, "Polynomial order must be zero or larger");
   MFEM_ASSERT(u.Size() >= p+1, "Size of u is too small");
   MFEM_ASSERT(dudx.Size() >= p+1, "Size of dudx is too small");
   MFEM_ASSERT(dudt.Size() >= p+1, "Size of dudt is too small");
   CalcScaledJacobi(p, alpha, x, t, u.GetData(),
                    dudx.GetData(), dudt.GetData());
}

void FuentesPyramid::CalcHomogenizedScaJacobi(int p, real_t alpha,
                                              real_t t0, real_t t1,
                                              real_t *u,
                                              real_t *dudt0, real_t *dudt1)
{
   MFEM_ASSERT(p >= 0, "Polynomial order must be zero or larger");
   CalcScaledJacobi(p, alpha, t1, t0+t1, u, dudt1, dudt0);
   for (int i = 0; i <= p; i++) { dudt1[i] += dudt0[i]; }
}

void FuentesPyramid::CalcHomogenizedScaJacobi(int p, real_t alpha,
                                              real_t t0, real_t t1,
                                              Vector &u)
{
   MFEM_ASSERT(p >= 0, "Polynomial order must be zero or larger");
   MFEM_ASSERT(u.Size() >= p+1, "Size of u is too small");
   CalcHomogenizedScaJacobi(p, alpha, t0, t1, u.GetData());
}

void FuentesPyramid::CalcHomogenizedScaJacobi(int p, real_t alpha,
                                              real_t t0, real_t t1,
                                              Vector &u,
                                              Vector &dudt0, Vector &dudt1)
{
   MFEM_ASSERT(p >= 0, "Polynomial order must be zero or larger");
   MFEM_ASSERT(u.Size() >= p+1, "Size of u is too small");
   MFEM_ASSERT(dudt0.Size() >= p+1, "Size of dudt0 is too small");
   MFEM_ASSERT(dudt1.Size() >= p+1, "Size of dudt1 is too small");
   CalcHomogenizedScaJacobi(p, alpha, t0, t1, u.GetData(),
                            dudt0.GetData(), dudt1.GetData());
}

void FuentesPyramid::CalcHomogenizedIntJacobi(int p, real_t alpha,
                                              real_t t0, real_t t1,
                                              real_t *u,
                                              real_t *dudt0, real_t *dudt1)
{
   MFEM_ASSERT(p >= 0, "Polynomial order must be zero or larger");
   CalcIntegratedJacobi(p, alpha, t1, t0+t1, u, dudt1, dudt0);
   for (int i = 0; i <= p; i++) { dudt1[i] += dudt0[i]; }
}

void FuentesPyramid::CalcHomogenizedIntJacobi(int p, real_t alpha,
                                              real_t t0, real_t t1,
                                              Vector &u)
{
   MFEM_ASSERT(p >= 0, "Polynomial order must be zero or larger");
   MFEM_ASSERT(u.Size() >= p+1, "Size of u is too small");
   CalcHomogenizedIntJacobi(p, alpha, t0, t1, u.GetData());
}

void FuentesPyramid::CalcHomogenizedIntJacobi(int p, real_t alpha,
                                              real_t t0, real_t t1,
                                              Vector &u,
                                              Vector &dudt0, Vector &dudt1)
{
   MFEM_ASSERT(p >= 0, "Polynomial order must be zero or larger");
   MFEM_ASSERT(u.Size() >= p+1, "Size of u is too small");
   MFEM_ASSERT(dudt0.Size() >= p+1, "Size of dudt0 is too small");
   MFEM_ASSERT(dudt1.Size() >= p+1, "Size of dudt1 is too small");
   CalcHomogenizedIntJacobi(p, alpha, t0, t1, u.GetData(),
                            dudt0.GetData(), dudt1.GetData());
}

void FuentesPyramid::phi_E(int p, real_t s0, real_t s1,
                           real_t *u)
{
   MFEM_ASSERT(p >= 2, "Polynomial order must be two or larger");
   CalcHomogenizedIntLegendre(p, s0, s1, u);
}

void FuentesPyramid::phi_E(int p, real_t s0, real_t s1,
                           real_t *u, real_t *duds0, real_t *duds1)
{
   MFEM_ASSERT(p >= 2, "Polynomial order must be two or larger");
   CalcHomogenizedIntLegendre(p, s0, s1, u, duds0, duds1);
}

void FuentesPyramid::phi_E(int p, Vector s, Vector &u)
{
   MFEM_ASSERT(p >= 2, "Polynomial order must be two or larger");
   MFEM_ASSERT(u.Size() >= p+1, "Size of u is too small");
   phi_E(p, s[0], s[1], u.GetData());
}

void FuentesPyramid::phi_E(int p, Vector s, Vector &u, DenseMatrix &duds)
{
   MFEM_ASSERT(p >= 2, "Polynomial order must be two or larger");
   MFEM_ASSERT(s.Size() >= 2, "Size of s must be 2 or larger");
   MFEM_ASSERT(u.Size() >= p+1, "Size of u is too small");
   MFEM_ASSERT(duds.Height() >= p+1, "First dimension of duds is too small");
   MFEM_ASSERT(duds.Width() >= 2,
               "Second dimension of duds must be 2 or larger");
   phi_E(p, s[0], s[1], u.GetData(), duds.GetColumn(0), duds.GetColumn(1));
}

void FuentesPyramid::phi_E(int p, Vector s, const DenseMatrix &grad_s,
                           Vector &u, DenseMatrix &grad_u) const
{
   MFEM_ASSERT(p >= 2, "Polynomial order must be two or larger");
   MFEM_ASSERT(s.Size() >= 2, "Size of s must be 2 or larger");
   MFEM_ASSERT(grad_s.Height() >= 2,
               "First dimension of grad_s must be 2");
   MFEM_ASSERT(grad_s.Width() >= 3,
               "Second dimension of grad_s must be 3");
   MFEM_ASSERT(u.Size() >= p+1, "Size of u is too small");
   MFEM_ASSERT(grad_u.Height() >= p+1,
               "First dimension of grad_u is too small");
   MFEM_ASSERT(grad_u.Width() == grad_s.Width(),
               "Second dimension of grad_u must match that of grad_s");
#ifdef MFEM_THREAD_SAFE
   DenseMatrix phi_E_mtmp;
#endif
   DenseMatrix &duds = phi_E_mtmp;
   duds.SetSize(p + 1, grad_s.Height());

   phi_E(p, s[0], s[1], u.GetData(), duds.GetColumn(0), duds.GetColumn(1));
   Mult(duds, grad_s, grad_u);
}

void FuentesPyramid::phi_Q(int p, Vector s, Vector t, DenseMatrix &u) const
{
   MFEM_ASSERT(p >= 2, "Polynomial order must be two or larger");
   MFEM_ASSERT(s.Size() >= 2, "Size of s must be 2 or larger");
   MFEM_ASSERT(t.Size() >= 2, "Size of t must be 2 or larger");
   MFEM_ASSERT(u.Height() >= p+1, "First dimension of u is too small");
   MFEM_ASSERT(u.Width() >= p+1, "Second dimension of u is too small");

#ifdef MFEM_THREAD_SAFE
   Vector phi_Q_vtmp1;
   Vector phi_Q_vtmp2;
#endif
   Vector &phi_E_i = phi_Q_vtmp1;
   Vector &phi_E_j = phi_Q_vtmp2;

   phi_E_i.SetSize(p+1);
   phi_E(p, s, phi_E_i);

   phi_E_j.SetSize(p+1);
   phi_E(p, t, phi_E_j);

   for (int j=0; j<=p; j++)
      for (int i=0; i<=p; i++)
      {
         u(i,j) = phi_E_i[i] * phi_E_j[j];
      }
}

void FuentesPyramid::phi_Q(int p, Vector s, const DenseMatrix &grad_s,
                           Vector t, const DenseMatrix &grad_t,
                           DenseMatrix &u, DenseTensor &grad_u) const
{
   MFEM_ASSERT(p >= 2, "Polynomial order must be two or larger");
   MFEM_ASSERT(s.Size() >= 2, "Size of s must be 2 or larger");
   MFEM_ASSERT(grad_s.Height() >= 2,
               "First dimension of grad_s must be 2 or larger");
   MFEM_ASSERT(grad_s.Width() >= 3,
               "Second dimension of grad_s must be 3 or larger");
   MFEM_ASSERT(t.Size() >= 2, "Size of t must be 2 or larger");
   MFEM_ASSERT(grad_t.Height() >= 2,
               "First dimension of grad_t must be 2 or larger");
   MFEM_ASSERT(grad_t.Width() >= 3,
               "Second dimension of grad_t must be 3 or larger");
   MFEM_ASSERT(u.Height() >= p+1, "First dimension of u is too small");
   MFEM_ASSERT(u.Width() >= p+1, "First dimension of u is too small");
   MFEM_ASSERT(grad_u.SizeI() >= p+1,
               "First dimension of grad_u is too small");
   MFEM_ASSERT(grad_u.SizeJ() >= p+1,
               "Second dimension of grad_u is too small");
   MFEM_ASSERT(grad_u.SizeK() >= 3,
               "Third dimension of grad_u must be 3 or larger");

#ifdef MFEM_THREAD_SAFE
   Vector phi_Q_vtmp1;
   Vector phi_Q_vtmp2;
   DenseMatrix phi_Q_mtmp1;
   DenseMatrix phi_Q_mtmp2;
#endif
   Vector       &phi_E_i = phi_Q_vtmp1;
   Vector       &phi_E_j = phi_Q_vtmp2;
   DenseMatrix &dphi_E_i = phi_Q_mtmp1;
   DenseMatrix &dphi_E_j = phi_Q_mtmp2;

   phi_E_i.SetSize(p+1);
   dphi_E_i.SetSize(p+1, grad_s.Width());
   phi_E(p, s, grad_s, phi_E_i, dphi_E_i);

   phi_E_j.SetSize(p+1);
   dphi_E_j.SetSize(p+1, grad_t.Width());
   phi_E(p, t, grad_t, phi_E_j, dphi_E_j);

   for (int j=0; j<=p; j++)
      for (int i=0; i<=p; i++)
      {
         u(i,j) = phi_E_i[i] * phi_E_j[j];

         for (int k=0; k<3; k++)
            grad_u(i,j,k) =
               phi_E_i(i) * dphi_E_j(j,k) + dphi_E_i(i,k) * phi_E_j(j);
      }
}

void FuentesPyramid::phi_T(int p, Vector s, DenseMatrix &u) const
{
   MFEM_ASSERT(p >= 3, "Polynomial order must be three or larger");
   MFEM_ASSERT(s.Size() >= 3, "Size of s must be 3 or larger");
   MFEM_ASSERT(u.Height() >= p, "First dimension of u is too small");
   MFEM_ASSERT(u.Width() >= p-1, "Second dimension of u is too small");

#ifdef MFEM_THREAD_SAFE
   Vector phi_T_vtmp1;
   Vector phi_T_vtmp2;
#endif
   Vector &phi_E_i = phi_T_vtmp1;
   Vector &L_j     = phi_T_vtmp2;

   phi_E_i.SetSize(p);
   phi_E(p-1, s, phi_E_i);

   L_j.SetSize(p-1);

   u = 0.0;
   for (int i = 2; i < p; i++)
   {
      const real_t alpha = 2.0 * i;
      CalcHomogenizedIntJacobi(p-2, alpha, s[0] + s[1], s[2], L_j);

      for (int j = 1; i + j <= p; j++)
      {
         u(i,j) = phi_E_i[i] * L_j[j];
      }
   }
}

void FuentesPyramid::phi_T(int p, Vector s, const DenseMatrix &grad_s,
                           DenseMatrix &u, DenseTensor &grad_u) const
{
   MFEM_ASSERT(p >= 3, "Polynomial order must be three or larger");
   MFEM_ASSERT(s.Size() >= 3, "Size of s must be 3 or larger");
   MFEM_ASSERT(grad_s.Height() >= 3,
               "First dimension of grad_s must be 2 or larger");
   MFEM_ASSERT(grad_s.Width() >= 3,
               "Second dimension of grad_s must be 3 or larger");
   MFEM_ASSERT(u.Height() >= p, "First dimension of u is too small");
   MFEM_ASSERT(u.Width() >= p-1, "Second dimension of u is too small");
   MFEM_ASSERT(grad_u.SizeI() >= p,
               "First dimension of grad_u is too small");
   MFEM_ASSERT(grad_u.SizeJ() >= p-1,
               "Second dimension of grad_u is too small");
   MFEM_ASSERT(grad_u.SizeK() >= 3,
               "Third dimension of grad_u must be 3 or larger");

#ifdef MFEM_THREAD_SAFE
   Vector phi_T_vtmp1;
   Vector phi_T_vtmp2;
   Vector phi_T_vtmp3;
   Vector phi_T_vtmp4;
   DenseMatrix phi_T_mtmp1;
#endif
   Vector      &phi_E_i  = phi_T_vtmp1;
   DenseMatrix &dphi_E_i = phi_T_mtmp1;
   Vector      &L_j      = phi_T_vtmp2;
   Vector      &dL_j_dx  = phi_T_vtmp3;
   Vector      &dL_j_dt  = phi_T_vtmp4;

   phi_E_i.SetSize(p);
   dphi_E_i.SetSize(p, 3);
   phi_E(p-1, s, grad_s, phi_E_i, dphi_E_i);

   L_j.SetSize(p-1);
   dL_j_dx.SetSize(p-1);
   dL_j_dt.SetSize(p-1);

   u = 0.0;
   grad_u = 0.0;
   for (int i = 2; i < p; i++)
   {
      const real_t alpha = 2.0 * i;
      CalcHomogenizedIntJacobi(p-2, alpha, s[0] + s[1], s[2], L_j,
                               dL_j_dx, dL_j_dt);

      for (int j = 1; i + j <= p; j++)
      {
         u(i,j) = phi_E_i[i] * L_j[j];

         for (int d=0; d<3; d++)
            grad_u(i, j, d) = dphi_E_i(i, d) * L_j[j] +
                              phi_E_i[i] * (dL_j_dx[j] * (grad_s(0, d) +
                                                          grad_s(1, d)) +
                                            dL_j_dt[j] * grad_s(2, d));
      }
   }
}

void FuentesPyramid::E_E(int p, Vector s, Vector sds, DenseMatrix &u) const
{
   MFEM_ASSERT(p >= 1, "Polynomial order must be one or larger");
   MFEM_ASSERT(s.Size() >= 2, "Size of s must be 2 or larger");
   MFEM_ASSERT(sds.Size() >= 3, "Size of sds must be 3 or larger");
   MFEM_ASSERT(u.Height() >= p, "First dimension of u is too small");
   MFEM_ASSERT(u.Width() >= 3, "Second dimension of u must be 3 or larger");

#ifdef MFEM_THREAD_SAFE
   Vector E_E_vtmp;
#endif
   Vector &P_i = E_E_vtmp;

   P_i.SetSize(p);
   CalcHomogenizedScaLegendre(p - 1, s[0], s[1], P_i);
   for (int i=0; i<p; i++)
   {
      u(i,0) = P_i(i) * sds(0);
      u(i,1) = P_i(i) * sds(1);
      u(i,2) = P_i(i) * sds(2);
   }
}

void FuentesPyramid::E_E(int p, Vector s, const DenseMatrix &grad_s,
                         DenseMatrix &u, DenseMatrix &curl_u) const
{
   MFEM_ASSERT(p >= 1, "Polynomial order must be one or larger");
   MFEM_ASSERT(s.Size() >= 2, "Size of s must be 2 or larger");
   MFEM_ASSERT(grad_s.Height() >= 2,
               "First dimension of grad_s must be 2 or larger");
   MFEM_ASSERT(grad_s.Width() >= 3,
               "Second dimension of grad_s must be 3 or larger");
   MFEM_ASSERT(u.Height() >= p, "First dimension of u is too small");
   MFEM_ASSERT(u.Width() >= 3, "Second dimension of u must be 3 or larger");
   MFEM_ASSERT(curl_u.Height() >= p,
               "First dimension of curl_u is too small");
   MFEM_ASSERT(curl_u.Width() >= 3,
               "Second dimension of curl_u must be 3 or larger");

#ifdef MFEM_THREAD_SAFE
   Vector E_E_vtmp;
#endif
   Vector &P_i = E_E_vtmp;

   P_i.SetSize(p);
   CalcHomogenizedScaLegendre(p - 1, s[0], s[1], P_i);

   Vector grad_s0({grad_s(0,0), grad_s(0,1), grad_s(0,2)});
   Vector grad_s1({grad_s(1,0), grad_s(1,1), grad_s(1,2)});
   Vector sds(3);
   add(s(0), grad_s1, -s(1), grad_s0, sds);

   Vector dsxds(3);
   grad_s0.cross3D(grad_s1, dsxds);

   for (int i=0; i<p; i++)
   {
      u(i,0) = P_i(i) * sds(0);
      u(i,1) = P_i(i) * sds(1);
      u(i,2) = P_i(i) * sds(2);

      curl_u(i, 0) = (i + 2) * P_i(i) * dsxds(0);
      curl_u(i, 1) = (i + 2) * P_i(i) * dsxds(1);
      curl_u(i, 2) = (i + 2) * P_i(i) * dsxds(2);
   }
}

void FuentesPyramid::E_Q(int p, Vector s, Vector sds, Vector t,
                         DenseTensor &u) const
{
   MFEM_ASSERT(p >= 2, "Polynomial order must be two or larger");
   MFEM_ASSERT(s.Size() >= 2, "Size of s must be 2 or larger");
   MFEM_ASSERT(sds.Size() >= 3, "Size of sds must be 3 or larger");
   MFEM_ASSERT(t.Size() >= 2, "Size of t must be 2 or larger");
   MFEM_ASSERT(u.SizeI() >= p, "First dimension of u is too small");
   MFEM_ASSERT(u.SizeJ() >= p+1, "Second dimension of u is too small");
   MFEM_ASSERT(u.SizeK() >= 3, "Third dimension of u must be 3 or larger");

#ifdef MFEM_THREAD_SAFE
   Vector      E_Q_vtmp;
   DenseMatrix E_Q_mtmp1;
#endif

   DenseMatrix &E_E_i   = E_Q_mtmp1;
   Vector      &phi_E_j = E_Q_vtmp;

   E_E_i.SetSize(p, 3);
   E_E(p, s, sds, E_E_i);

   phi_E_j.SetSize(p + 1);
   phi_E(p, t, phi_E_j);

   for (int k=0; k<3; k++)
   {
      u(k).SetCol(0, 0.0);
      u(k).SetCol(1, 0.0);
   }

   for (int j=2; j<=p; j++)
      for (int i=0; i<p; i++)
         for (int k=0; k<3; k++)
         {
            u(i, j, k) = phi_E_j(j) * E_E_i(i, k);
         }
}

void FuentesPyramid::E_Q(int p, Vector s, const DenseMatrix &grad_s,
                         Vector t, const DenseMatrix &grad_t,
                         DenseTensor &u, DenseTensor &curl_u) const
{
   MFEM_ASSERT(p >= 2, "Polynomial order must be two or larger");
   MFEM_ASSERT(s.Size() >= 2, "Size of s must be 2 or larger");
   MFEM_ASSERT(grad_s.Height() >= 2,
               "First dimension of grad_s must be 2");
   MFEM_ASSERT(grad_s.Width() >= 3,
               "Second dimension of grad_s must be 3");
   MFEM_ASSERT(t.Size() >= 2, "Size of t must be 2 or larger");
   MFEM_ASSERT(grad_t.Height() >= 2,
               "First dimension of grad_t must be 2");
   MFEM_ASSERT(grad_t.Width() >= 3,
               "Second dimension of grad_t must be 3");
   MFEM_ASSERT(u.SizeI() >= p, "First dimension of u is too small");
   MFEM_ASSERT(u.SizeJ() >= p+1, "Second dimension of u is too small");
   MFEM_ASSERT(u.SizeK() >= 3, "Third dimension of u must be 3 or larger");
   MFEM_ASSERT(curl_u.SizeI() >= p, "First dimension of curl_u is too small");
   MFEM_ASSERT(curl_u.SizeJ() >= p+1,
               "Second dimension of curl_u is too small");
   MFEM_ASSERT(curl_u.SizeK() >= 3,
               "Third dimension of curl_u must be 3 or larger");

#ifdef MFEM_THREAD_SAFE
   Vector      E_Q_vtmp;
   DenseMatrix E_Q_mtmp1;
   DenseMatrix E_Q_mtmp2;
   DenseMatrix E_Q_mtmp3;
#endif
   Vector      &phi_E_j  = E_Q_vtmp;
   DenseMatrix &dphi_E_j = E_Q_mtmp1;
   DenseMatrix &E_E_i    = E_Q_mtmp2;
   DenseMatrix &dE_E_i   = E_Q_mtmp3;

   phi_E_j.SetSize(p + 1);
   dphi_E_j.SetSize(p + 1, grad_t.Width());
   phi_E(p, t, grad_t, phi_E_j, dphi_E_j);

   E_E_i.SetSize(p, 3);
   dE_E_i.SetSize(p, 3);
   E_E(p, s, grad_s, E_E_i, dE_E_i);

   for (int k=0; k<3; k++)
   {
      u(k).SetCol(0, 0.0);
      u(k).SetCol(1, 0.0);
      curl_u(k).SetCol(0, 0.0);
      curl_u(k).SetCol(1, 0.0);
   }

   for (int j=2; j<=p; j++)
      for (int i=0; i<p; i++)
      {
         for (int k=0; k<3; k++)
         {
            u(i, j, k) = phi_E_j(j) * E_E_i(i, k);
         }

         curl_u(i, j, 0) = phi_E_j(j) * dE_E_i(i, 0)
                           + dphi_E_j(j, 1) * E_E_i(i, 2)
                           - dphi_E_j(j, 2) * E_E_i(i, 1);
         curl_u(i, j, 1) = phi_E_j(j) * dE_E_i(i, 1)
                           + dphi_E_j(j, 2) * E_E_i(i, 0)
                           - dphi_E_j(j, 0) * E_E_i(i, 2);
         curl_u(i, j, 2) = phi_E_j(j) * dE_E_i(i, 2)
                           + dphi_E_j(j, 0) * E_E_i(i, 1)
                           - dphi_E_j(j, 1) * E_E_i(i, 0);
      }
}

void FuentesPyramid::E_T(int p, Vector s, Vector sds, DenseTensor &u) const
{
   MFEM_ASSERT(p >= 2, "Polynomial order must be two or larger");
   MFEM_ASSERT(s.Size() >= 3, "Size of s must be 3 or larger");
   MFEM_ASSERT(sds.Size() >= 3, "Size of sds must be 3 or larger");
   MFEM_ASSERT(u.SizeI() >= p - 1, "First dimension of u is too small");
   MFEM_ASSERT(u.SizeJ() >= p, "Second dimension of u is too small");
   MFEM_ASSERT(u.SizeK() >= 3, "Third dimension of u must be 3 or larger");

#ifdef MFEM_THREAD_SAFE
   Vector      E_T_vtmp1;
   DenseMatrix E_T_mtmp1;
#endif
   Vector &L_j = E_T_vtmp1;
   DenseMatrix &E_E_i = E_T_mtmp1;

   E_E_i.SetSize(p - 1, 3);
   E_E(p - 1, s, sds, E_E_i);

   L_j.SetSize(p);
   for (int i=0; i<p-1; i++)
   {
      const real_t alpha = 2.0 * i + 1.0;
      CalcHomogenizedIntJacobi(p - 1, alpha, s[0] + s[1], s[2], L_j);

      u(i, 0, 0) = 0.0; u(i, 0, 1) = 0.0; u(i, 0, 2) = 0.0;
      for (int j=1; i+j<p; j++)
         for (int k=0; k<3; k++)
         {
            u(i, j, k) = L_j(j) * E_E_i(i, k);
         }
   }
}

void FuentesPyramid::E_T(int p, Vector s, const DenseMatrix & grad_s,
                         DenseTensor &u, DenseTensor &curl_u) const
{
   MFEM_ASSERT(p >= 2, "Polynomial order must be two or larger");
   MFEM_ASSERT(s.Size() >= 3, "Size of s must be 3 or larger");
   MFEM_ASSERT(grad_s.Height() >= 3,
               "First dimension of grad_s must be 3");
   MFEM_ASSERT(grad_s.Width() >= 3,
               "Second dimension of grad_s must be 3");
   MFEM_ASSERT(u.SizeI() >= p - 1, "First dimension of u is too small");
   MFEM_ASSERT(u.SizeJ() >= p, "Second dimension of u is too small");
   MFEM_ASSERT(u.SizeK() >= 3, "Third dimension of u must be 3 or larger");
   MFEM_ASSERT(curl_u.SizeI() >= p - 1,
               "First dimension of curl_u is too small");
   MFEM_ASSERT(curl_u.SizeJ() >= p,
               "Second dimension of curl_u is too small");
   MFEM_ASSERT(curl_u.SizeK() >= 3,
               "Third dimension of curl_u must be 3 or larger");

#ifdef MFEM_THREAD_SAFE
   Vector      E_T_vtmp1;
   Vector      E_T_vtmp2;
   Vector      E_T_vtmp3;
   DenseMatrix E_T_mtmp1;
   DenseMatrix E_T_mtmp2;
#endif
   Vector &L_j = E_T_vtmp1;
   Vector &dL_j_dx = E_T_vtmp2;
   Vector &dL_j_dt = E_T_vtmp3;
   DenseMatrix & E_E_i = E_T_mtmp1;
   DenseMatrix &dE_E_i = E_T_mtmp2;

   Vector dL(3), grad_L(3);

   E_E_i.SetSize(p - 1, 3);
   dE_E_i.SetSize(p - 1, 3);
   E_E(p - 1, s, grad_s, E_E_i, dE_E_i);

   L_j.SetSize(p);
   dL_j_dx.SetSize(p);
   dL_j_dt.SetSize(p);
   for (int i=0; i<p-1; i++)
   {
      const real_t alpha = 2.0 * i + 1.0;
      CalcHomogenizedIntJacobi(p - 1, alpha, s[0] + s[1], s[2], L_j,
                               dL_j_dx, dL_j_dt);

      u(i, 0, 0) = 0.0; u(i, 0, 1) = 0.0; u(i, 0, 2) = 0.0;
      curl_u(i, 0, 0) = 0.0; curl_u(i, 0, 1) = 0.0; curl_u(i, 0, 2) = 0.0;
      for (int j=1; i+j<p; j++)
      {
         dL(0) = dL_j_dx(j); dL(1) = dL_j_dx(j); dL(2) = dL_j_dt(j);

         grad_s.MultTranspose(dL, grad_L);

         for (int k=0; k<3; k++)
         {
            u(i, j, k) = L_j(j) * E_E_i(i, k);
            curl_u(i, j, k) = L_j(j) * dE_E_i(i, k);
         }
         curl_u(i, j, 0) += grad_L(1) * E_E_i(i, 2) - grad_L(2) * E_E_i(i, 1);
         curl_u(i, j, 1) += grad_L(2) * E_E_i(i, 0) - grad_L(0) * E_E_i(i, 2);
         curl_u(i, j, 2) += grad_L(0) * E_E_i(i, 1) - grad_L(1) * E_E_i(i, 0);
      }
   }
}

void FuentesPyramid::V_Q(int p, Vector s, Vector sds,
                         Vector t, Vector tdt, DenseTensor &u) const
{
   MFEM_ASSERT(p >= 1, "Polynomial order must be one or larger");
   MFEM_ASSERT(s.Size() >= 2, "Size of s must be 2 or larger");
   MFEM_ASSERT(sds.Size() >= 3, "Size of sds must be 3 or larger");
   MFEM_ASSERT(t.Size() >= 2, "Size of t must be 2 or larger");
   MFEM_ASSERT(tdt.Size() >= 3, "Size of tdt must be 3 or larger");
   MFEM_ASSERT(u.SizeI() >= p, "First dimension of u is too small");
   MFEM_ASSERT(u.SizeJ() >= p, "Second dimension of u is too small");
   MFEM_ASSERT(u.SizeK() >= 3, "Third dimension of u must be 3 or larger");

#ifdef MFEM_THREAD_SAFE
   DenseMatrix V_Q_mtmp1;
   DenseMatrix V_Q_mtmp2;
#endif
   DenseMatrix &E_E_i = V_Q_mtmp1;
   DenseMatrix &E_E_j = V_Q_mtmp2;

   E_E_i.SetSize(p, 3);
   E_E(p, s, sds, E_E_i);

   E_E_j.SetSize(p, 3);
   E_E(p, t, tdt, E_E_j);

   for (int j=0; j<p; j++)
      for (int i=0; i<p; i++)
      {
         u(i, j, 0) = E_E_i(i, 1) * E_E_j(j, 2) - E_E_i(i, 2) * E_E_j(j, 1);
         u(i, j, 1) = E_E_i(i, 2) * E_E_j(j, 0) - E_E_i(i, 0) * E_E_j(j, 2);
         u(i, j, 2) = E_E_i(i, 0) * E_E_j(j, 1) - E_E_i(i, 1) * E_E_j(j, 0);
      }
}

void FuentesPyramid::V_T(int p, Vector s, Vector sdsxds, DenseTensor &u) const
{
   MFEM_ASSERT(p >= 1, "Polynomial order must be one or larger");
   MFEM_ASSERT(s.Size() >= 2, "Size of s must be 2 or larger");
   MFEM_ASSERT(sdsxds.Size() >= 3, "Size of sdsxds must be 3 or larger");
   MFEM_ASSERT(u.SizeI() >= p, "First dimension of u is too small");
   MFEM_ASSERT(u.SizeJ() >= p, "Second dimension of u is too small");
   MFEM_ASSERT(u.SizeK() >= 3, "Third dimension of u must be 3 or larger");

#ifdef MFEM_THREAD_SAFE
   Vector V_T_vtmp1;
   Vector V_T_vtmp2;
#endif
   Vector &P_i = V_T_vtmp1;
   Vector &P_j = V_T_vtmp2;

   P_i.SetSize(p);
   CalcHomogenizedScaLegendre(p-1, s[0], s[1], P_i);

   P_j.SetSize(p);
   for (int i=0; i<p; i++)
   {
      const real_t alpha = 2.0 * i + 1.0;
      CalcHomogenizedScaJacobi(p-1, alpha, s[0] + s[1], s[2], P_j);
      for (int j=0; i + j < p; j++)
      {
         const real_t vij = P_i(i) * P_j(j);
         u(i,j,0) = vij * sdsxds(0);
         u(i,j,1) = vij * sdsxds(1);
         u(i,j,2) = vij * sdsxds(2);
      }
   }
}

void FuentesPyramid::V_T(int p, Vector s, Vector sdsxds, real_t dsdsxds,
                         DenseTensor &u, DenseMatrix &du) const
{
   MFEM_ASSERT(p >= 1, "Polynomial order must be one or larger");
   MFEM_ASSERT(s.Size() >= 2, "Size of s must be 2 or larger");
   MFEM_ASSERT(sdsxds.Size() >= 3, "Size of sdsxds must be 3 or larger");
   MFEM_ASSERT(u.SizeI() >= p, "First dimension of u is too small");
   MFEM_ASSERT(u.SizeJ() >= p, "Second dimension of u is too small");
   MFEM_ASSERT(u.SizeK() >= 3, "Third dimension of u must be 3 or larger");
   MFEM_ASSERT(du.Height() >= p, "First dimension of du is too small");
   MFEM_ASSERT(du.Width() >= p, "Second dimension of du is too small");

#ifdef MFEM_THREAD_SAFE
   Vector V_T_vtmp1;
   Vector V_T_vtmp2;
#endif
   Vector &P_i = V_T_vtmp1;
   Vector &P_j = V_T_vtmp2;

   P_i.SetSize(p);
   CalcHomogenizedScaLegendre(p-1, s[0], s[1], P_i);

   P_j.SetSize(p);
   for (int i=0; i<p; i++)
   {
      const real_t alpha = 2.0 * i + 1.0;
      CalcHomogenizedScaJacobi(p-1, alpha, s[0] + s[1], s[2], P_j);
      for (int j=0; i + j < p; j++)
      {
         const real_t vij = P_i(i) * P_j(j);
         u(i,j,0) = vij * sdsxds(0);
         u(i,j,1) = vij * sdsxds(1);
         u(i,j,2) = vij * sdsxds(2);

         du(i,j) = (i+j+3) * vij * dsdsxds;
      }
   }
}

void FuentesPyramid::VT_T(int p, Vector s, Vector sds, Vector sdsxds,
                          real_t mu, Vector grad_mu, DenseTensor &u) const
{
   MFEM_ASSERT(p >= 1, "Polynomial order must be one or larger");
   MFEM_ASSERT(s.Size() >= 2, "Size of s must be 2 or larger");
   MFEM_ASSERT(sds.Size() >= 3, "Size of sds must be 3 or larger");
   MFEM_ASSERT(sdsxds.Size() >= 3, "Size of sdsxds must be 3 or larger");
   MFEM_ASSERT(grad_mu.Size() >= 3, "Size of grad_mu must be 3 or larger");
   MFEM_ASSERT(u.SizeI() >= p, "First dimension of u is too small");
   MFEM_ASSERT(u.SizeJ() >= p, "Second dimension of u is too small");
   MFEM_ASSERT(u.SizeK() >= 3, "Third dimension of u must be 3 or larger");

#ifdef MFEM_THREAD_SAFE
   Vector      VT_T_vtmp1;
   Vector      VT_T_vtmp2;
   DenseMatrix VT_T_mtmp1;
   DenseTensor VT_T_ttmp1;
#endif

   Vector ms({mu * s(0), mu * s(1), s(2)});
   Vector s2(s.GetData(), 2);

   Vector &P_i = VT_T_vtmp1;
   P_i.SetSize(p);
   CalcHomogenizedScaLegendre(p-1, ms[0], ms[1], P_i);

   DenseMatrix &EE0 = VT_T_mtmp1;
   EE0.SetSize(1,3);
   Vector EE(EE0.GetData(), 3);
   E_E(1, s2, sds, EE0);

   Vector dmuxEE(3);
   grad_mu.cross3D(EE, dmuxEE);

   DenseTensor &VT00 = VT_T_ttmp1;
   VT00.SetSize(1,1,3);
   V_T(1, s, sdsxds, VT00);

   Vector &J_j = VT_T_vtmp2;
   J_j.SetSize(p);

   u = 0.0;

   for (int i=0; i<p; i++)
   {
      CalcHomogenizedScaJacobi(p-i-1, 2*i+1, ms[0] + ms[1], ms[2], J_j);
      for (int j=0; i+j<p; j++)
         for (int k=0; k<3; k++)
            u(i, j, k) = P_i(i) * J_j(j) *
                         (mu * VT00(0,0,k) + s(2) * dmuxEE(k));
   }
}

void FuentesPyramid::VT_T(int p, Vector s, Vector sds, Vector sdsxds,
                          Vector grad_s2, real_t mu, Vector grad_mu,
                          DenseTensor &u, DenseMatrix &du) const
{
   MFEM_ASSERT(p >= 1, "Polynomial order must be one or larger");
   MFEM_ASSERT(s.Size() >= 2, "Size of s must be 2 or larger");
   MFEM_ASSERT(sds.Size() >= 3, "Size of sds must be 3 or larger");
   MFEM_ASSERT(sdsxds.Size() >= 3, "Size of sdsxds must be 3 or larger");
   MFEM_ASSERT(grad_s2.Size() >= 3, "Size of grad_s2 must be 3 or larger");
   MFEM_ASSERT(grad_mu.Size() >= 3, "Size of grad_mu must be 3 or larger");
   MFEM_ASSERT(u.SizeI() >= p, "First dimension of u is too small");
   MFEM_ASSERT(u.SizeJ() >= p, "Second dimension of u is too small");
   MFEM_ASSERT(u.SizeK() >= 3, "Third dimension of u must be 3 or larger");
   MFEM_ASSERT(du.Height() >= p, "First dimension of du is too small");
   MFEM_ASSERT(du.Width() >= p, "Second dimension of du is too small");

#ifdef MFEM_THREAD_SAFE
   Vector      VT_T_vtmp1;
   Vector      VT_T_vtmp2;
   DenseMatrix VT_T_mtmp1;
   DenseTensor VT_T_ttmp1;
#endif

   Vector ms({mu * s(0), mu * s(1), s(2)});
   Vector s2(s.GetData(), 2);

   Vector &P_i = VT_T_vtmp1;
   P_i.SetSize(p);
   CalcHomogenizedScaLegendre(p-1, ms[0], ms[1], P_i);

   DenseMatrix &EE0 = VT_T_mtmp1;
   EE0.SetSize(1,3);
   Vector EE(EE0.GetData(), 3);
   E_E(1, s2, sds, EE0);

   Vector dmuxEE(3);
   grad_mu.cross3D(EE, dmuxEE);

   Vector EExds2(3);
   EE.cross3D(grad_s2, EExds2);

   DenseTensor &VT00 = VT_T_ttmp1;
   VT00.SetSize(1,1,3);
   V_T(1, s, sdsxds, VT00);

   Vector &J_j = VT_T_vtmp2;
   J_j.SetSize(p);

   Vector EV(3);

   u = 0.0;
   du = 0.0;

   for (int i=0; i<p; i++)
   {
      CalcHomogenizedScaJacobi(p-i-1, 2*i+1, ms[0] + ms[1], ms[2], J_j);
      for (int j=0; i+j<p; j++)
      {
         for (int k=0; k<3; k++)
         {
            u(i, j, k) = P_i(i) * J_j(j) *
                         (mu * VT00(0, 0, k) + s(2) * dmuxEE(k));

            EV(k) = (i+j+3) * EExds2(k) - VT00(0, 0, k);
         }

         du(i, j) = P_i(i) * J_j(j) * (grad_mu * EV);
      }
   }
}

void FuentesPyramid::V_L(int p, Vector sx, const DenseMatrix &grad_sx,
                         Vector sy, const DenseMatrix &grad_sy,
                         real_t t, Vector grad_t,
                         DenseTensor &u) const
{
   MFEM_ASSERT(p >= 2, "Polynomial order must be two or larger");
   MFEM_ASSERT(sx.Size() >= 2, "Size of sx must be 2 or larger");
   MFEM_ASSERT(grad_sx.Height() >= 2,
               "First dimension of grad_sx must be 2 or larger");
   MFEM_ASSERT(grad_sx.Width() >= 3,
               "Second dimension of grad_sx must be 3 or larger");
   MFEM_ASSERT(sy.Size() >= 2, "Size of sy must be 2 or larger");
   MFEM_ASSERT(grad_sy.Height() >= 2,
               "First dimension of grad_sy must be 2 or larger");
   MFEM_ASSERT(grad_sy.Width() >= 3,
               "Second dimension of grad_sy must be 3 or larger");
   MFEM_ASSERT(grad_t.Size() >= 3, "Size of grad_t must be 3 or larger");
   MFEM_ASSERT(u.SizeI() >= p+1, "First dimension of u is too small");
   MFEM_ASSERT(u.SizeJ() >= p+1, "Second dimension of u is too small");
   MFEM_ASSERT(u.SizeK() >= 3, "Third dimension of u must be 3 or larger");

#ifdef MFEM_THREAD_SAFE
   Vector      V_L_vtmp1;
   Vector      V_L_vtmp2;
   DenseMatrix V_L_mtmp1;
   DenseMatrix V_L_mtmp2;
#endif
   Vector       &phi_E_i = V_L_vtmp1;
   Vector       &phi_E_j = V_L_vtmp2;
   DenseMatrix &dphi_E_i = V_L_mtmp1;
   DenseMatrix &dphi_E_j = V_L_mtmp2;

   Vector grad_t3(grad_t.GetData(), 3);

   phi_E_i.SetSize(p+1);
   dphi_E_i.SetSize(p+1, grad_sx.Width());
   phi_E(p, sx, grad_sx, phi_E_i, dphi_E_i);

   phi_E_j.SetSize(p+1);
   dphi_E_j.SetSize(p+1, grad_sy.Width());
   phi_E(p, sy, grad_sy, phi_E_j, dphi_E_j);

   Vector dphii(3);
   Vector dphij(3);
   Vector dphidphi(3);
   Vector phidphi(3);
   Vector dtphidphi(3);

   for (int j=2; j<=p; j++)
   {
      for (int l=0; l<3; l++) { dphij[l] = dphi_E_j(j, l); }
      for (int i=2; i<=p; i++)
      {
         for (int l=0; l<3; l++) { dphii[l] = dphi_E_i(i, l); }

         dphii.cross3D(dphij, dphidphi);

         add(phi_E_i(i), dphij, -phi_E_j(j), dphii, phidphi);

         grad_t3.cross3D(phidphi, dtphidphi);

         for (int l=0; l<3; l++)
         {
            u(i, j, l) = t * (t * dphidphi(l) + dtphidphi(l));
         }
      }
   }
}

void FuentesPyramid::V_R(int p, Vector s, const DenseMatrix &grad_s,
                         real_t mu, Vector dmu, real_t t, Vector dt,
                         DenseMatrix &u) const
{
   MFEM_ASSERT(p >= 2, "Polynomial order must be two or larger");
   MFEM_ASSERT(s.Size() >= 2, "Size of s must be 2 or larger");
   MFEM_ASSERT(grad_s.Height() >= 2,
               "First dimension of grad_s must be 2");
   MFEM_ASSERT(grad_s.Width() >= 3,
               "Second dimension of grad_s must be 3");
   MFEM_ASSERT(dmu.Size() >= 3, "Size of dmu must be 3 or larger");
   MFEM_ASSERT(dt.Size() >= 3, "Size of dt must be 3 or larger");
   MFEM_ASSERT(u.Height() >= p+1, "First dimension of u is too small");
   MFEM_ASSERT(u.Width() >= 3, "Second dimension of u is too small");

#ifdef MFEM_THREAD_SAFE
   Vector      V_R_vtmp;
   DenseMatrix V_R_mtmp;
#endif
   Vector       &phi_E_i = V_R_vtmp;
   DenseMatrix &dphi_E_i = V_R_mtmp;

   phi_E_i.SetSize(p+1);
   dphi_E_i.SetSize(p+1, grad_s.Width());
   phi_E(p, s, grad_s, phi_E_i, dphi_E_i);

   u.SetRow(0, 0.0);
   u.SetRow(1, 0.0);

   Vector dmu3(dmu.GetData(), 3);
   Vector dt3(dt.GetData(), 3);
   Vector dphit2(3);
   Vector dphixdmu(3);
   Vector dphi(3);

   for (int i=2; i<=p; i++)
   {
      // dphi_E_i.GetRow(i, dphi);
      for (int l=0; l<3; l++) { dphi[l] = dphi_E_i(i, l); }
      add(t * t, dphi, 2.0 * t * phi_E_i(i), dt3, dphit2);
      dphit2.cross3D(dmu3, dphixdmu);
      // u.SetRow(i, dphixdmu);
      for (int l=0; l<3; l++) { u(i, l) = dphixdmu(l); }
   }
}

} // namespace mfem
