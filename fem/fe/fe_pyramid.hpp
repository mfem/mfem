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

#ifndef MFEM_FE_PYRAMID
#define MFEM_FE_PYRAMID

#include "fe_base.hpp"

namespace mfem
{

class FuentesPyramid
{
public:
   FuentesPyramid() = default;

   static double lam1(const double x, const double y, const double z)
   { return (z < 1.0) ? (1.0 - x - z) * (1.0 - y - z) / (1.0 - z): 0.0; }
   static double lam2(const double x, const double y, const double z)
   { return (z < 1.0) ? x * (1.0 - y - z) / (1.0 - z): 0.0; }
   static double lam3(const double x, const double y, const double z)
   { return (z < 1.0) ? x * y / (1.0 - z): 0.0; }
   static double lam4(const double x, const double y, const double z)
   { return (z < 1.0) ? (1.0 - x - z) * y / (1.0 - z): 0.0; }
   static double lam5(const double x, const double y, const double z)
   { return z; }

   static void grad_lam1(const double x, const double y, const double z,
                         double du[]);
   static void grad_lam2(const double x, const double y, const double z,
                         double du[]);
   static void grad_lam3(const double x, const double y, const double z,
                         double du[]);
   static void grad_lam4(const double x, const double y, const double z,
                         double du[]);
   static void grad_lam5(const double x, const double y, const double z,
                         double du[]);

   static double mu0(const double x) { return 1.0 - x; }
   static double mu1(const double x) { return x; }

   static double dmu0(const double x) { return -1.0; }
   static double dmu1(const double x) { return 1.0; }


   static double nu0(const double x, const double y)
   { return 1.0 - x - y; }
   static double nu1(const double x, const double y) { return x; }
   static double nu2(const double x, const double y) { return y; }

   static void grad_nu0(const double x, const double y, double dnu[])
   { dnu[0] = -1.0; dnu[1] = -1.0;}
   static void grad_nu1(const double x, const double y, double dnu[])
   { dnu[0] = 1.0; dnu[1] = 0.0;}
   static void grad_nu2(const double x, const double y, double dnu[])
   { dnu[0] = 0.0; dnu[1] = 1.0;}

   static void phi_E(const int p, const double s0, double s1, double *u);
   static void phi_E(const int p, const double s0, double s1, double *u,
                     double *duds0, double *duds1);

   static void phi_E(const int p, const double s0, double s1, Vector &u)
   { phi_E(p, s0, s1, u.GetData()); }
   static void phi_E(const int p, const double s0, double s1, Vector &u,
                     Vector &duds0, Vector &duds1)
   { phi_E(p, s0, s1, u.GetData(), duds0.GetData(), duds1.GetData()); }

   static void E_E(const int p, const double s0, double s1, double *u);
   static void E_E(const int p, const double s0, double s1, double *u,
                   double *duds0, double *duds1);

   static void CalcScaledLegendre(const int p, const double x, const double t,
                                  double *u);
   static void CalcScaledLegendre(const int p, const double x, const double t,
                                  double *u, double *dudx, double *dudt);

   static void CalcScaledLegendre(const int p, const double x, const double t,
                                  Vector &u)
   { CalcScaledLegendre(p, x, t, u.GetData()); }
   static void CalcScaledLegendre(const int p, const double x, const double t,
                                  Vector &u, Vector &dudx, Vector &dudt)
   { CalcScaledLegendre(p, x, t, u.GetData(), dudx.GetData(), dudt.GetData()); }

   static void CalcIntegratedLegendre(const int p, const double x,
                                      const double t, double *u);
   static void CalcIntegratedLegendre(const int p, const double x,
                                      const double t, double *u,
                                      double *dudx, double *dudt);

   static void CalcIntegratedLegendre(const int p, const double x,
                                      const double t, Vector &u)
   { CalcIntegratedLegendre(p, x, t, u.GetData()); }
   static void CalcIntegratedLegendre(const int p, const double x,
                                      const double t, Vector &u,
                                      Vector &dudx, Vector &dudt)
   {
      CalcIntegratedLegendre(p, x, t, u.GetData(),
                             dudx.GetData(), dudt.GetData());
   }

   static void CalcScaledJacobi(const int p, const double alpha,
                                const double x, const double t,
                                double *u);
   static void CalcScaledJacobi(const int p, const double alpha,
                                const double x, const double t,
                                double *u, double *dudx, double *dudt);

   static void CalcScaledJacobi(const int p, const double alpha,
                                const double x, const double t,
                                Vector &u)
   { CalcScaledJacobi(p, alpha, x, t, u.GetData()); }
   static void CalcScaledJacobi(const int p, const double alpha,
                                const double x, const double t,
                                Vector &u, Vector &dudx, Vector &dudt)
   {
      CalcScaledJacobi(p, alpha, x, t, u.GetData(),
                       dudx.GetData(), dudt.GetData());
   }

   static void CalcIntegratedJacobi(const int p, const double alpha,
                                    const double x, const double t,
                                    double *u);
   static void CalcIntegratedJacobi(const int p, const double alpha,
                                    const double x, const double t,
                                    double *u, double *dudx, double *dudt);

   static void CalcIntegratedJacobi(const int p, const double alpha,
                                    const double x, const double t,
                                    Vector &u)
   { CalcIntegratedJacobi(p, alpha, x, t, u.GetData()); }
   static void CalcIntegratedJacobi(const int p, const double alpha,
                                    const double x, const double t,
                                    Vector &u, Vector &dudx, Vector &dudt)
   {
      CalcIntegratedJacobi(p, alpha, x, t, u.GetData(),
                           dudx.GetData(), dudt.GetData());
   }

};

} // namespace mfem

#endif

