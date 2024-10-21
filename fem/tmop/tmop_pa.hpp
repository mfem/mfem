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

#ifndef MFEM_TMOP_PA_HPP
#define MFEM_TMOP_PA_HPP

#include "../../config/config.hpp"
#include "../../fem/kernels.hpp"
#include "../../general/forall.hpp"
#include "../../linalg/dinvariants.hpp"
#include "../../linalg/kernels.hpp"
#include "../tmop.hpp"
#include "../tmop_tools.hpp"

namespace mfem
{

/// Abstract base class for the 2D metric TMOP PA kernels.
struct TMOP_PA_Metric_2D
{
   static constexpr int DIM = 2;
   using Args = kernels::InvariantsEvaluator2D::Buffers;

   virtual MFEM_HOST_DEVICE void EvalP(const double (&Jpt)[4], const double *w,
                                       double (&P)[4])
   {
   }

   virtual MFEM_HOST_DEVICE void
   AssembleH(const int qx, const int qy, const int e, const double weight,
             const double (&Jpt)[4], const double *w, const DeviceTensor<7> &H)
   {
   }
};

struct TMOP_PA_Metric_001 : TMOP_PA_Metric_2D
{
   MFEM_HOST_DEVICE
   void EvalP(const double (&Jpt)[4], const double *w, double (&P)[4]) override
   {
      double dI1[4];
      kernels::InvariantsEvaluator2D ie(Args().J(Jpt).dI1(dI1));
      kernels::Set(2, 2, 1.0, ie.Get_dI1(), P);
   }

   MFEM_HOST_DEVICE
   void AssembleH(const int qx, const int qy, const int e, const double weight,
                  const double (&Jpt)[4], const double *w,
                  const DeviceTensor<7> &H) override
   {
      // weight * ddI1
      double ddI1[4];
      kernels::InvariantsEvaluator2D ie(Args().J(Jpt).ddI1(ddI1));
      for (int i = 0; i < DIM; i++)
      {
         for (int j = 0; j < DIM; j++)
         {
            ConstDeviceMatrix ddi1(ie.Get_ddI1(i, j), DIM, DIM);
            for (int r = 0; r < DIM; r++)
            {
               for (int c = 0; c < DIM; c++)
               {
                  const double h = ddi1(r, c);
                  H(r, c, i, j, qx, qy, e) = weight * h;
               }
            }
         }
      }
   }
};

struct TMOP_PA_Metric_002 : TMOP_PA_Metric_2D
{
   MFEM_HOST_DEVICE
   void EvalP(const double (&Jpt)[4], const double *w, double (&P)[4]) override
   {
      double dI1b[4], dI2b[4];
      kernels::InvariantsEvaluator2D ie(Args().J(Jpt).dI1b(dI1b).dI2b(dI2b));
      kernels::Set(2, 2, 1. / 2., ie.Get_dI1b(), P);
   }

   MFEM_HOST_DEVICE
   void AssembleH(const int qx, const int qy, const int e, const double weight,
                  const double (&Jpt)[4], const double *w,
                  const DeviceTensor<7> &H) override
   {
      // 0.5 * weight * dI1b
      double ddI1[4], ddI1b[4], dI2b[4];
      kernels::InvariantsEvaluator2D ie(
         Args().J(Jpt).ddI1(ddI1).ddI1b(ddI1b).dI2b(dI2b));
      const double half_weight = 0.5 * weight;
      for (int i = 0; i < DIM; i++)
      {
         for (int j = 0; j < DIM; j++)
         {
            ConstDeviceMatrix ddi1b(ie.Get_ddI1b(i, j), DIM, DIM);
            for (int r = 0; r < DIM; r++)
            {
               for (int c = 0; c < DIM; c++)
               {
                  const double h = ddi1b(r, c);
                  H(r, c, i, j, qx, qy, e) = half_weight * h;
               }
            }
         }
      }
   }
};

struct TMOP_PA_Metric_007 : TMOP_PA_Metric_2D
{
   MFEM_HOST_DEVICE
   void EvalP(const double (&Jpt)[4], const double *w, double (&P)[4]) override
   {
      double dI1[4], dI2[4], dI2b[4];
      kernels::InvariantsEvaluator2D ie(
         Args().J(Jpt).dI1(dI1).dI2(dI2).dI2b(dI2b));
      const double I2 = ie.Get_I2();
      kernels::Add(2, 2, 1.0 + 1.0 / I2, ie.Get_dI1(), -ie.Get_I1() / (I2 * I2),
                   ie.Get_dI2(), P);
   }

   MFEM_HOST_DEVICE
   void AssembleH(const int qx, const int qy, const int e, const double weight,
                  const double (&Jpt)[4], const double *w,
                  const DeviceTensor<7> &H) override
   {
      double ddI1[4], ddI2[4], dI1[4], dI2[4], dI2b[4];
      kernels::InvariantsEvaluator2D ie(
         Args().J(Jpt).ddI1(ddI1).ddI2(ddI2).dI1(dI1).dI2(dI2).dI2b(dI2b));
      const double c1 = 1. / ie.Get_I2();
      const double c2 = weight * c1 * c1;
      const double c3 = ie.Get_I1() * c2;
      ConstDeviceMatrix di1(ie.Get_dI1(), DIM, DIM);
      ConstDeviceMatrix di2(ie.Get_dI2(), DIM, DIM);

      for (int i = 0; i < DIM; i++)
      {
         for (int j = 0; j < DIM; j++)
         {
            ConstDeviceMatrix ddi1(ie.Get_ddI1(i, j), DIM, DIM);
            ConstDeviceMatrix ddi2(ie.Get_ddI2(i, j), DIM, DIM);
            for (int r = 0; r < DIM; r++)
            {
               for (int c = 0; c < DIM; c++)
               {
                  H(r, c, i, j, qx, qy, e) =
                     weight * (1.0 + c1) * ddi1(r, c) - c3 * ddi2(r, c) -
                     c2 * (di1(i, j) * di2(r, c) + di2(i, j) * di1(r, c)) +
                     2.0 * c1 * c3 * di2(r, c) * di2(i, j);
               }
            }
         }
      }
   }
};

struct TMOP_PA_Metric_056 : TMOP_PA_Metric_2D
{
   MFEM_HOST_DEVICE
   void EvalP(const double (&Jpt)[4], const double *w, double (&P)[4]) override
   {
      // 0.5*(1 - 1/I2b^2)*dI2b
      double dI2b[4];
      kernels::InvariantsEvaluator2D ie(Args().J(Jpt).dI2b(dI2b));
      const double I2b = ie.Get_I2b();
      kernels::Set(2, 2, 0.5 * (1.0 - 1.0 / (I2b * I2b)), ie.Get_dI2b(), P);
   }

   MFEM_HOST_DEVICE
   void AssembleH(const int qx, const int qy, const int e, const double weight,
                  const double (&Jpt)[4], const double *w,
                  const DeviceTensor<7> &H) override
   {
      // (0.5 - 0.5/I2b^2)*ddI2b + (1/I2b^3)*(dI2b x dI2b)
      double dI2b[4], ddI2b[4];
      kernels::InvariantsEvaluator2D ie(Args().J(Jpt).dI2b(dI2b).ddI2b(ddI2b));
      const double I2b = ie.Get_I2b();
      ConstDeviceMatrix di2b(ie.Get_dI2b(), DIM, DIM);
      for (int i = 0; i < DIM; i++)
      {
         for (int j = 0; j < DIM; j++)
         {
            ConstDeviceMatrix ddi2b(ie.Get_ddI2b(i, j), DIM, DIM);
            for (int r = 0; r < DIM; r++)
            {
               for (int c = 0; c < DIM; c++)
               {
                  H(r, c, i, j, qx, qy, e) =
                     weight * (0.5 - 0.5 / (I2b * I2b)) * ddi2b(r, c) +
                     weight / (I2b * I2b * I2b) * di2b(r, c) * di2b(i, j);
               }
            }
         }
      }
   }
};

struct TMOP_PA_Metric_077 : TMOP_PA_Metric_2D
{
   MFEM_HOST_DEVICE
   void EvalP(const double (&Jpt)[4], const double *w, double (&P)[4]) override
   {
      double dI2[4], dI2b[4];
      kernels::InvariantsEvaluator2D ie(Args().J(Jpt).dI2(dI2).dI2b(dI2b));
      const double I2 = ie.Get_I2();
      kernels::Set(2, 2, 0.5 * (1.0 - 1.0 / (I2 * I2)), ie.Get_dI2(), P);
   }

   MFEM_HOST_DEVICE
   void AssembleH(const int qx, const int qy, const int e, const double weight,
                  const double (&Jpt)[4], const double *w,
                  const DeviceTensor<7> &H) override
   {
      double dI2[4], dI2b[4], ddI2[4];
      kernels::InvariantsEvaluator2D ie(
         Args().J(Jpt).dI2(dI2).dI2b(dI2b).ddI2(ddI2));
      const double I2 = ie.Get_I2(), I2inv_sq = 1.0 / (I2 * I2);
      ConstDeviceMatrix di2(ie.Get_dI2(), DIM, DIM);
      for (int i = 0; i < DIM; i++)
      {
         for (int j = 0; j < DIM; j++)
         {
            ConstDeviceMatrix ddi2(ie.Get_ddI2(i, j), DIM, DIM);
            for (int r = 0; r < DIM; r++)
            {
               for (int c = 0; c < DIM; c++)
               {
                  H(r, c, i, j, qx, qy, e) =
                     weight * 0.5 * (1.0 - I2inv_sq) * ddi2(r, c) +
                     weight * (I2inv_sq / I2) * di2(r, c) * di2(i, j);
               }
            }
         }
      }
   }
};

struct TMOP_PA_Metric_080 : TMOP_PA_Metric_2D
{
   MFEM_HOST_DEVICE
   void EvalP(const double (&Jpt)[4], const double *w, double (&P)[4]) override
   {
      // w0 P_2 + w1 P_77
      double dI1b[4], dI2[4], dI2b[4];
      kernels::InvariantsEvaluator2D ie(
         Args().J(Jpt).dI1b(dI1b).dI2(dI2).dI2b(dI2b));
      kernels::Set(2, 2, w[0] * 0.5, ie.Get_dI1b(), P);
      const double I2 = ie.Get_I2();
      kernels::Add(2, 2, w[1] * 0.5 * (1.0 - 1.0 / (I2 * I2)), ie.Get_dI2(), P);
   }

   MFEM_HOST_DEVICE
   void AssembleH(const int qx, const int qy, const int e, const double weight,
                  const double (&Jpt)[4], const double *w,
                  const DeviceTensor<7> &H) override
   {
      // w0 H_2 + w1 H_77
      double ddI1[4], ddI1b[4], dI2[4], dI2b[4], ddI2[4];
      kernels::InvariantsEvaluator2D ie(
         Args().J(Jpt).dI2(dI2).ddI1(ddI1).ddI1b(ddI1b).dI2b(dI2b).ddI2(ddI2));

      const double I2 = ie.Get_I2(), I2inv_sq = 1.0 / (I2 * I2);
      ConstDeviceMatrix di2(ie.Get_dI2(), DIM, DIM);
      for (int i = 0; i < DIM; i++)
      {
         for (int j = 0; j < DIM; j++)
         {
            ConstDeviceMatrix ddi1b(ie.Get_ddI1b(i, j), DIM, DIM);
            ConstDeviceMatrix ddi2(ie.Get_ddI2(i, j), DIM, DIM);
            for (int r = 0; r < DIM; r++)
            {
               for (int c = 0; c < DIM; c++)
               {
                  H(r, c, i, j, qx, qy, e) =
                     w[0] * 0.5 * weight * ddi1b(r, c) +
                     w[1] * (weight * 0.5 * (1.0 - I2inv_sq) * ddi2(r, c) +
                             weight * (I2inv_sq / I2) * di2(r, c) * di2(i, j));
               }
            }
         }
      }
   }
};

struct TMOP_PA_Metric_094 : TMOP_PA_Metric_2D
{
   MFEM_HOST_DEVICE
   void EvalP(const double (&Jpt)[4], const double *w, double (&P)[4]) override
   {
      // w0 P_2 + w1 P_56
      double dI1b[4], dI2b[4];
      kernels::InvariantsEvaluator2D ie(Args().J(Jpt).dI1b(dI1b).dI2b(dI2b));
      kernels::Set(2, 2, w[0] * 0.5, ie.Get_dI1b(), P);
      const double I2b = ie.Get_I2b();
      kernels::Add(2, 2, w[1] * 0.5 * (1.0 - 1.0 / (I2b * I2b)), ie.Get_dI2b(),
                   P);
   }

   MFEM_HOST_DEVICE
   void AssembleH(const int qx, const int qy, const int e, const double weight,
                  const double (&Jpt)[4], const double *w,
                  const DeviceTensor<7> &H) override
   {
      // w0 H_2 + w1 H_56
      double ddI1[4], ddI1b[4], dI2b[4], ddI2b[4];
      kernels::InvariantsEvaluator2D ie(
         Args().J(Jpt).ddI1(ddI1).ddI1b(ddI1b).dI2b(dI2b).ddI2b(ddI2b));
      const double I2b = ie.Get_I2b();
      ConstDeviceMatrix di2b(ie.Get_dI2b(), DIM, DIM);
      for (int i = 0; i < DIM; i++)
      {
         for (int j = 0; j < DIM; j++)
         {
            ConstDeviceMatrix ddi1b(ie.Get_ddI1b(i, j), DIM, DIM);
            ConstDeviceMatrix ddi2b(ie.Get_ddI2b(i, j), DIM, DIM);
            for (int r = 0; r < DIM; r++)
            {
               for (int c = 0; c < DIM; c++)
               {
                  H(r, c, i, j, qx, qy, e) =
                     w[0] * 0.5 * weight * ddi1b(r, c) +
                     w[1] *
                     (weight * (0.5 - 0.5 / (I2b * I2b)) * ddi2b(r, c) +
                      weight / (I2b * I2b * I2b) * di2b(r, c) * di2b(i, j));
               }
            }
         }
      }
   }
};

/// Abstract base class for the 3D metric TMOP PA kernels.
struct TMOP_PA_Metric_3D
{
   static constexpr int DIM = 3;
   using Args = kernels::InvariantsEvaluator3D::Buffers;

   virtual MFEM_HOST_DEVICE void EvalP(const double (&Jpt)[DIM * DIM],
                                       const double *w, double (&P)[DIM * DIM])
   {
   }

   virtual MFEM_HOST_DEVICE void
   AssembleH(const int qx, const int qy, const int qz, const int e,
             const double weight, double *Jrt, double *Jpr,
             const double (&Jpt)[DIM * DIM], const double *w,
             const DeviceTensor<5 + DIM> &H) const
   {
   }
};

struct TMOP_PA_Metric_302 : TMOP_PA_Metric_3D
{
   MFEM_HOST_DEVICE
   void EvalP(const double (&Jpt)[9], const double *w, double (&P)[9]) override
   {
      // (I1b/9)*dI2b + (I2b/9)*dI1b
      double B[9];
      double dI1b[9], dI2[9], dI2b[9], dI3b[9];
      kernels::InvariantsEvaluator3D ie(
         Args().J(Jpt).B(B).dI1b(dI1b).dI2(dI2).dI2b(dI2b).dI3b(dI3b));
      const double alpha = ie.Get_I1b() / 9.;
      const double beta = ie.Get_I2b() / 9.;
      kernels::Add(3, 3, alpha, ie.Get_dI2b(), beta, ie.Get_dI1b(), P);
   }

   MFEM_HOST_DEVICE
   void AssembleH(const int qx, const int qy, const int qz, const int e,
                  const double weight, double *Jrt, double *Jpr,
                  const double (&Jpt)[9], const double *w,
                  const DeviceTensor<8> &H) const override
   {
      double B[9];
      double dI1b[9], ddI1b[9];
      double dI2[9], dI2b[9], ddI2[9], ddI2b[9];
      double dI3b[9]; // = Jrt;
      // (dI2b*dI1b + dI1b*dI2b)/9 + (I1b/9)*ddI2b + (I2b/9)*ddI1b
      kernels::InvariantsEvaluator3D ie(Args()
                                        .J(Jpt)
                                        .B(B)
                                        .dI1b(dI1b)
                                        .ddI1b(ddI1b)
                                        .dI2(dI2)
                                        .dI2b(dI2b)
                                        .ddI2(ddI2)
                                        .ddI2b(ddI2b)
                                        .dI3b(dI3b));

      const double c1 = weight / 9.;
      const double I1b = ie.Get_I1b();
      const double I2b = ie.Get_I2b();
      ConstDeviceMatrix di1b(ie.Get_dI1b(), DIM, DIM);
      ConstDeviceMatrix di2b(ie.Get_dI2b(), DIM, DIM);
      for (int i = 0; i < DIM; i++)
      {
         for (int j = 0; j < DIM; j++)
         {
            ConstDeviceMatrix ddi1b(ie.Get_ddI1b(i, j), DIM, DIM);
            ConstDeviceMatrix ddi2b(ie.Get_ddI2b(i, j), DIM, DIM);
            for (int r = 0; r < DIM; r++)
            {
               for (int c = 0; c < DIM; c++)
               {
                  const double dp =
                     (di2b(r, c) * di1b(i, j) + di1b(r, c) * di2b(i, j)) +
                     ddi2b(r, c) * I1b + ddi1b(r, c) * I2b;
                  H(r, c, i, j, qx, qy, qz, e) = c1 * dp;
               }
            }
         }
      }
   }
};

struct TMOP_PA_Metric_303 : TMOP_PA_Metric_3D
{
   MFEM_HOST_DEVICE
   void EvalP(const double (&Jpt)[9], const double *w, double (&P)[9]) override
   {
      // dI1b/3
      double B[9];
      double dI1b[9], dI3b[9];
      kernels::InvariantsEvaluator3D ie(
         Args().J(Jpt).B(B).dI1b(dI1b).dI3b(dI3b));
      kernels::Set(3, 3, 1. / 3., ie.Get_dI1b(), P);
   }

   MFEM_HOST_DEVICE
   void AssembleH(const int qx, const int qy, const int qz, const int e,
                  const double weight, double *Jrt, double *Jpr,
                  const double (&Jpt)[9], const double *w,
                  const DeviceTensor<8> &H) const override
   {
      double B[9];
      double dI1b[9], ddI1[9], ddI1b[9];
      double dI2[9], dI2b[9], ddI2[9], ddI2b[9];
      double *dI3b = Jrt, *ddI3b = Jpr;

      // ddI1b/3
      kernels::InvariantsEvaluator3D ie(Args()
                                        .J(Jpt)
                                        .B(B)
                                        .dI1b(dI1b)
                                        .ddI1(ddI1)
                                        .ddI1b(ddI1b)
                                        .dI2(dI2)
                                        .dI2b(dI2b)
                                        .ddI2(ddI2)
                                        .ddI2b(ddI2b)
                                        .dI3b(dI3b)
                                        .ddI3b(ddI3b));

      const double c1 = weight / 3.;
      for (int i = 0; i < DIM; i++)
      {
         for (int j = 0; j < DIM; j++)
         {
            ConstDeviceMatrix ddi1b(ie.Get_ddI1b(i, j), DIM, DIM);
            for (int r = 0; r < DIM; r++)
            {
               for (int c = 0; c < DIM; c++)
               {
                  const double dp = ddi1b(r, c);
                  H(r, c, i, j, qx, qy, qz, e) = c1 * dp;
               }
            }
         }
      }
   }
};

struct TMOP_PA_Metric_315 : TMOP_PA_Metric_3D
{
   MFEM_HOST_DEVICE
   void EvalP(const double (&Jpt)[9], const double *w, double (&P)[9]) override
   {
      // 2*(I3b - 1)*dI3b
      double dI3b[9];
      kernels::InvariantsEvaluator3D ie(Args().J(Jpt).dI3b(dI3b));
      double sign_detJ;
      const double I3b = ie.Get_I3b(sign_detJ);
      kernels::Set(3, 3, 2.0 * (I3b - 1.0), ie.Get_dI3b(sign_detJ), P);
   }

   MFEM_HOST_DEVICE
   void AssembleH(const int qx, const int qy, const int qz, const int e,
                  const double weight, double *Jrt, double *Jpr,
                  const double (&Jpt)[9], const double *w,
                  const DeviceTensor<8> &H) const override
   {
      double *dI3b = Jrt, *ddI3b = Jpr;
      // 2*(dI3b x dI3b) + 2*(I3b - 1)*ddI3b
      kernels::InvariantsEvaluator3D ie(Args().J(Jpt).dI3b(dI3b).ddI3b(ddI3b));
      double sign_detJ;
      const double I3b = ie.Get_I3b(sign_detJ);
      ConstDeviceMatrix di3b(ie.Get_dI3b(sign_detJ), DIM, DIM);
      for (int i = 0; i < DIM; i++)
      {
         for (int j = 0; j < DIM; j++)
         {
            ConstDeviceMatrix ddi3b(ie.Get_ddI3b(i, j), DIM, DIM);
            for (int r = 0; r < DIM; r++)
            {
               for (int c = 0; c < DIM; c++)
               {
                  const double dp = 2.0 * weight * (I3b - 1.0) * ddi3b(r, c) +
                                    2.0 * weight * di3b(r, c) * di3b(i, j);
                  H(r, c, i, j, qx, qy, qz, e) = dp;
               }
            }
         }
      }
   }
};

struct TMOP_PA_Metric_318 : TMOP_PA_Metric_3D
{
   // P_318 = (I3b - 1/I3b^3)*dI3b.
   // Uses the I3b form, as dI3 and ddI3 were not implemented at the time
   MFEM_HOST_DEVICE
   void EvalP(const double (&Jpt)[9], const double *w, double (&P)[9]) override
   {
      double dI3b[9];
      kernels::InvariantsEvaluator3D ie(Args().J(Jpt).dI3b(dI3b));

      double sign_detJ;
      const double I3b = ie.Get_I3b(sign_detJ);
      kernels::Set(3, 3, I3b - 1.0 / (I3b * I3b * I3b), ie.Get_dI3b(sign_detJ),
                   P);
   }

   MFEM_HOST_DEVICE
   void AssembleH(const int qx, const int qy, const int qz, const int e,
                  const double weight, double *Jrt, double *Jpr,
                  const double (&Jpt)[9], const double *w,
                  const DeviceTensor<8> &H) const override
   {
      double *dI3b = Jrt, *ddI3b = Jpr;
      // dP_318 = (I3b - 1/I3b^3)*ddI3b + (1 + 3/I3b^4)*(dI3b x dI3b)
      // Uses the I3b form, as dI3 and ddI3 were not implemented at the time
      kernels::InvariantsEvaluator3D ie(Args().J(Jpt).dI3b(dI3b).ddI3b(ddI3b));
      double sign_detJ;
      const double I3b = ie.Get_I3b(sign_detJ);
      ConstDeviceMatrix di3b(ie.Get_dI3b(sign_detJ), DIM, DIM);
      for (int i = 0; i < DIM; i++)
      {
         for (int j = 0; j < DIM; j++)
         {
            ConstDeviceMatrix ddi3b(ie.Get_ddI3b(i, j), DIM, DIM);
            for (int r = 0; r < DIM; r++)
            {
               for (int c = 0; c < DIM; c++)
               {
                  const double dp =
                     weight * (I3b - 1.0 / (I3b * I3b * I3b)) * ddi3b(r, c) +
                     weight * (1.0 + 3.0 / (I3b * I3b * I3b * I3b)) *
                     di3b(r, c) * di3b(i, j);
                  H(r, c, i, j, qx, qy, qz, e) = dp;
               }
            }
         }
      }
   }
};

struct TMOP_PA_Metric_321 : TMOP_PA_Metric_3D
{
   MFEM_HOST_DEVICE
   void EvalP(const double (&Jpt)[9], const double *w, double (&P)[9]) override
   {
      // dI1 + (1/I3)*dI2 - (2*I2/I3b^3)*dI3b
      double B[9];
      double dI1[9], dI2[9], dI3b[9];
      kernels::InvariantsEvaluator3D ie(
         Args().J(Jpt).B(B).dI1(dI1).dI2(dI2).dI3b(dI3b));
      double sign_detJ;
      const double I3 = ie.Get_I3();
      const double alpha = 1.0 / I3;
      const double beta = -2. * ie.Get_I2() / (I3 * ie.Get_I3b(sign_detJ));
      kernels::Add(3, 3, alpha, ie.Get_dI2(), beta, ie.Get_dI3b(sign_detJ), P);
      kernels::Add(3, 3, ie.Get_dI1(), P);
   }

   MFEM_HOST_DEVICE
   void AssembleH(const int qx, const int qy, const int qz, const int e,
                  const double weight, double *Jrt, double *Jpr,
                  const double (&Jpt)[9], const double *w,
                  const DeviceTensor<8> &H) const override
   {
      double B[9];
      double dI1b[9], ddI1[9], ddI1b[9];
      double dI2[9], dI2b[9], ddI2[9], ddI2b[9];
      double *dI3b = Jrt, *ddI3b = Jpr;

      // ddI1 + (-2/I3b^3)*(dI2 x dI3b + dI3b x dI2)
      //      + (1/I3)*ddI2
      //      + (6*I2/I3b^4)*(dI3b x dI3b)
      //      + (-2*I2/I3b^3)*ddI3b
      kernels::InvariantsEvaluator3D ie(Args()
                                        .J(Jpt)
                                        .B(B)
                                        .dI1b(dI1b)
                                        .ddI1(ddI1)
                                        .ddI1b(ddI1b)
                                        .dI2(dI2)
                                        .dI2b(dI2b)
                                        .ddI2(ddI2)
                                        .ddI2b(ddI2b)
                                        .dI3b(dI3b)
                                        .ddI3b(ddI3b));
      double sign_detJ;
      const double I2 = ie.Get_I2();
      const double I3b = ie.Get_I3b(sign_detJ);
      ConstDeviceMatrix di2(ie.Get_dI2(), DIM, DIM);
      ConstDeviceMatrix di3b(ie.Get_dI3b(sign_detJ), DIM, DIM);
      const double c0 = 1.0 / I3b;
      const double c1 = weight * c0 * c0;
      const double c2 = -2 * c0 * c1;
      const double c3 = c2 * I2;
      for (int i = 0; i < DIM; i++)
      {
         for (int j = 0; j < DIM; j++)
         {
            ConstDeviceMatrix ddi1(ie.Get_ddI1(i, j), DIM, DIM);
            ConstDeviceMatrix ddi2(ie.Get_ddI2(i, j), DIM, DIM);
            ConstDeviceMatrix ddi3b(ie.Get_ddI3b(i, j), DIM, DIM);
            for (int r = 0; r < DIM; r++)
            {
               for (int c = 0; c < DIM; c++)
               {
                  const double dp =
                     weight * ddi1(r, c) + c1 * ddi2(r, c) + c3 * ddi3b(r, c) +
                     c2 * ((di2(r, c) * di3b(i, j) + di3b(r, c) * di2(i, j))) -
                     3 * c0 * c3 * di3b(r, c) * di3b(i, j);
                  H(r, c, i, j, qx, qy, qz, e) = dp;
               }
            }
         }
      }
   }
};

struct TMOP_PA_Metric_332 : TMOP_PA_Metric_3D
{
   MFEM_HOST_DEVICE
   void EvalP(const double (&Jpt)[9], const double *w, double (&P)[9]) override
   {
      // w0 P_302 + w1 P_315
      double B[9];
      double dI1b[9], dI2[9], dI2b[9], dI3b[9];
      kernels::InvariantsEvaluator3D ie(
         Args().J(Jpt).B(B).dI1b(dI1b).dI2(dI2).dI2b(dI2b).dI3b(dI3b));
      const double alpha = w[0] * ie.Get_I1b() / 9.;
      const double beta = w[0] * ie.Get_I2b() / 9.;
      kernels::Add(3, 3, alpha, ie.Get_dI2b(), beta, ie.Get_dI1b(), P);
      double sign_detJ;
      const double I3b = ie.Get_I3b(sign_detJ);
      kernels::Add(3, 3, w[1] * 2.0 * (I3b - 1.0), ie.Get_dI3b(sign_detJ), P);
   }

   MFEM_HOST_DEVICE
   void AssembleH(const int qx, const int qy, const int qz, const int e,
                  const double weight, double *Jrt, double *Jpr,
                  const double (&Jpt)[9], const double *w,
                  const DeviceTensor<8> &H) const override
   {
      double B[9];
      double dI1b[9], /*ddI1[9],*/ ddI1b[9];
      double dI2[9], dI2b[9], ddI2[9], ddI2b[9];
      double *dI3b = Jrt, *ddI3b = Jpr;
      // w0 H_302 + w1 H_315
      kernels::InvariantsEvaluator3D ie(Args()
                                        .J(Jpt)
                                        .B(B)
                                        .dI1b(dI1b)
                                        .ddI1b(ddI1b)
                                        .dI2(dI2)
                                        .dI2b(dI2b)
                                        .ddI2(ddI2)
                                        .ddI2b(ddI2b)
                                        .dI3b(dI3b)
                                        .ddI3b(ddI3b));
      double sign_detJ;
      const double c1 = weight / 9.0;
      const double I1b = ie.Get_I1b();
      const double I2b = ie.Get_I2b();
      const double I3b = ie.Get_I3b(sign_detJ);
      ConstDeviceMatrix di1b(ie.Get_dI1b(), DIM, DIM);
      ConstDeviceMatrix di2b(ie.Get_dI2b(), DIM, DIM);
      ConstDeviceMatrix di3b(ie.Get_dI3b(sign_detJ), DIM, DIM);
      for (int i = 0; i < DIM; i++)
      {
         for (int j = 0; j < DIM; j++)
         {
            ConstDeviceMatrix ddi1b(ie.Get_ddI1b(i, j), DIM, DIM);
            ConstDeviceMatrix ddi2b(ie.Get_ddI2b(i, j), DIM, DIM);
            ConstDeviceMatrix ddi3b(ie.Get_ddI3b(i, j), DIM, DIM);
            for (int r = 0; r < DIM; r++)
            {
               for (int c = 0; c < DIM; c++)
               {
                  const double dp_302 =
                     (di2b(r, c) * di1b(i, j) + di1b(r, c) * di2b(i, j)) +
                     ddi2b(r, c) * I1b + ddi1b(r, c) * I2b;
                  const double dp_315 =
                     2.0 * weight * (I3b - 1.0) * ddi3b(r, c) +
                     2.0 * weight * di3b(r, c) * di3b(i, j);
                  H(r, c, i, j, qx, qy, qz, e) =
                     w[0] * c1 * dp_302 + w[1] * dp_315;
               }
            }
         }
      }
   }
};

struct TMOP_PA_Metric_338 : TMOP_PA_Metric_3D
{
   MFEM_HOST_DEVICE
   void EvalP(const double (&Jpt)[9], const double *w, double (&P)[9]) override
   {
      // w0 P_302 + w1 P_318
      double B[9];
      double dI1b[9], dI2[9], dI2b[9], dI3b[9];
      kernels::InvariantsEvaluator3D ie(
         Args().J(Jpt).B(B).dI1b(dI1b).dI2(dI2).dI2b(dI2b).dI3b(dI3b));
      const double alpha = w[0] * ie.Get_I1b() / 9.;
      const double beta = w[0] * ie.Get_I2b() / 9.;
      kernels::Add(3, 3, alpha, ie.Get_dI2b(), beta, ie.Get_dI1b(), P);
      double sign_detJ;
      const double I3b = ie.Get_I3b(sign_detJ);
      kernels::Add(3, 3, w[1] * (I3b - 1.0 / (I3b * I3b * I3b)),
                   ie.Get_dI3b(sign_detJ), P);
   }

   MFEM_HOST_DEVICE
   void AssembleH(const int qx, const int qy, const int qz, const int e,
                  const double weight, double *Jrt, double *Jpr,
                  const double (&Jpt)[9], const double *w,
                  const DeviceTensor<8> &H) const override
   {
      double B[9];
      double dI1b[9], ddI1b[9];
      double dI2[9], dI2b[9], ddI2[9], ddI2b[9];
      double *dI3b = Jrt, *ddI3b = Jpr;
      // w0 H_302 + w1 H_318
      kernels::InvariantsEvaluator3D ie(Args()
                                        .J(Jpt)
                                        .B(B)
                                        .dI1b(dI1b)
                                        .ddI1b(ddI1b)
                                        .dI2(dI2)
                                        .dI2b(dI2b)
                                        .ddI2(ddI2)
                                        .ddI2b(ddI2b)
                                        .dI3b(dI3b)
                                        .ddI3b(ddI3b));
      double sign_detJ;
      const double c1 = weight / 9.;
      const double I1b = ie.Get_I1b();
      const double I2b = ie.Get_I2b();
      const double I3b = ie.Get_I3b(sign_detJ);
      ConstDeviceMatrix di1b(ie.Get_dI1b(), DIM, DIM);
      ConstDeviceMatrix di2b(ie.Get_dI2b(), DIM, DIM);
      ConstDeviceMatrix di3b(ie.Get_dI3b(sign_detJ), DIM, DIM);
      for (int i = 0; i < DIM; i++)
      {
         for (int j = 0; j < DIM; j++)
         {
            ConstDeviceMatrix ddi1b(ie.Get_ddI1b(i, j), DIM, DIM);
            ConstDeviceMatrix ddi2b(ie.Get_ddI2b(i, j), DIM, DIM);
            ConstDeviceMatrix ddi3b(ie.Get_ddI3b(i, j), DIM, DIM);
            for (int r = 0; r < DIM; r++)
            {
               for (int c = 0; c < DIM; c++)
               {
                  const double dp_302 =
                     (di2b(r, c) * di1b(i, j) + di1b(r, c) * di2b(i, j)) +
                     ddi2b(r, c) * I1b + ddi1b(r, c) * I2b;
                  const double dp_318 =
                     weight * (I3b - 1.0 / (I3b * I3b * I3b)) * ddi3b(r, c) +
                     weight * (1.0 + 3.0 / (I3b * I3b * I3b * I3b)) *
                     di3b(r, c) * di3b(i, j);
                  H(r, c, i, j, qx, qy, qz, e) =
                     w[0] * c1 * dp_302 + w[1] * dp_318;
               }
            }
         }
      }
   }
};

template <typename M /* metric */, typename K /* kernel */>
static void TMOPKernelLaunch(K &ker)
{
   const int d = ker.Ndof(), q = ker.Nqpt();

   if (d == 2 && q == 2) { return ker.template operator()<M, 2, 2>(); }
   if (d == 2 && q == 3) { return ker.template operator()<M, 2, 3>(); }
   if (d == 2 && q == 4) { return ker.template operator()<M, 2, 4>(); }
   if (d == 2 && q == 5) { return ker.template operator()<M, 2, 5>(); }
   if (d == 2 && q == 6) { return ker.template operator()<M, 2, 6>(); }

   if (d == 3 && q == 3) { return ker.template operator()<M, 3, 3>(); }
   if (d == 3 && q == 4) { return ker.template operator()<M, 3, 4>(); }
   if (d == 3 && q == 5) { return ker.template operator()<M, 3, 5>(); }
   if (d == 3 && q == 6) { return ker.template operator()<M, 3, 6>(); }

   if (d == 4 && q == 4) { return ker.template operator()<M, 4, 4>(); }
   if (d == 4 && q == 5) { return ker.template operator()<M, 4, 5>(); }
   if (d == 4 && q == 6) { return ker.template operator()<M, 4, 6>(); }

   if (d == 5 && q == 5) { return ker.template operator()<M, 5, 5>(); }
   if (d == 5 && q == 6) { return ker.template operator()<M, 5, 6>(); }

   ker.template operator()<M, 0, 0>();
}

} // namespace mfem

#endif // MFEM_TMOP_PA_HPP
