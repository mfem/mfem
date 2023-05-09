// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "tmop_pa_h2s.hpp"

namespace mfem
{

struct MetricTMOP_1 : MetricTMOPKer2D
{
   MFEM_HOST_DEVICE
   void AssembleH(const int qx, const int qy, const int e,
                  const double weight, const double (&Jpt)[4],
                  const double *w, const DeviceTensor<7> &H) override
   {
      // weight * ddI1
      double ddI1[4];
      kernels::InvariantsEvaluator2D ie(Args().J(Jpt).ddI1(ddI1));
      for (int i = 0; i < DIM; i++)
      {
         for (int j = 0; j < DIM; j++)
         {
            ConstDeviceMatrix ddi1(ie.Get_ddI1(i,j),DIM,DIM);
            for (int r = 0; r < DIM; r++)
            {
               for (int c = 0; c < DIM; c++)
               {
                  const double h = ddi1(r,c);
                  H(r,c,i,j,qx,qy,e) = weight * h;
               }
            }
         }
      }
   }
};


struct MetricTMOP_2 : MetricTMOPKer2D
{
   MFEM_HOST_DEVICE
   void AssembleH(const int qx, const int qy, const int e,
                  const double weight, const double (&Jpt)[4],
                  const double *w, const DeviceTensor<7> &H) override
   {
      // 0.5 * weight * dI1b
      double ddI1[4], ddI1b[4], dI2b[4];
      kernels::InvariantsEvaluator2D ie
      (Args().J(Jpt).ddI1(ddI1).ddI1b(ddI1b).dI2b(dI2b));
      const double half_weight = 0.5 * weight;
      for (int i = 0; i < DIM; i++)
      {
         for (int j = 0; j < DIM; j++)
         {
            ConstDeviceMatrix ddi1b(ie.Get_ddI1b(i,j),DIM,DIM);
            for (int r = 0; r < DIM; r++)
            {
               for (int c = 0; c < DIM; c++)
               {
                  const double h = ddi1b(r,c);
                  H(r,c,i,j,qx,qy,e) = half_weight * h;
               }
            }
         }
      }
   }
};

struct MetricTMOP_7 : MetricTMOPKer2D
{
   MFEM_HOST_DEVICE
   void AssembleH(const int qx, const int qy, const int e,
                  const double weight, const double (&Jpt)[4],
                  const double *w, const DeviceTensor<7> &H) override
   {
      double ddI1[4], ddI2[4], dI1[4], dI2[4], dI2b[4];
      kernels::InvariantsEvaluator2D ie
      (Args().J(Jpt).ddI1(ddI1).ddI2(ddI2)
       .dI1(dI1).dI2(dI2).dI2b(dI2b));
      const double c1 = 1./ie.Get_I2();
      const double c2 = weight*c1*c1;
      const double c3 = ie.Get_I1()*c2;
      ConstDeviceMatrix di1(ie.Get_dI1(),DIM,DIM);
      ConstDeviceMatrix di2(ie.Get_dI2(),DIM,DIM);

      for (int i = 0; i < DIM; i++)
      {
         for (int j = 0; j < DIM; j++)
         {
            ConstDeviceMatrix ddi1(ie.Get_ddI1(i,j),DIM,DIM);
            ConstDeviceMatrix ddi2(ie.Get_ddI2(i,j),DIM,DIM);
            for (int r = 0; r < DIM; r++)
            {
               for (int c = 0; c < DIM; c++)
               {
                  H(r,c,i,j,qx,qy,e) =
                     weight * (1.0 + c1) * ddi1(r,c)
                     - c3 * ddi2(r,c)
                     - c2 * (di1(i,j) * di2(r,c) + di2(i,j) * di1(r,c))
                     + 2.0 * c1 * c3 * di2(r,c) * di2(i,j);
               }
            }
         }
      }
   }
};

struct MetricTMOP_56 : MetricTMOPKer2D
{
   MFEM_HOST_DEVICE
   void AssembleH(const int qx, const int qy, const int e,
                  const double weight, const double (&Jpt)[4],
                  const double *w, const DeviceTensor<7> &H) override
   {
      // (0.5 - 0.5/I2b^2)*ddI2b + (1/I2b^3)*(dI2b x dI2b)
      double dI2b[4], ddI2b[4];
      kernels::InvariantsEvaluator2D ie
      (Args().J(Jpt).dI2b(dI2b).ddI2b(ddI2b));
      const double I2b = ie.Get_I2b();
      ConstDeviceMatrix di2b(ie.Get_dI2b(),DIM,DIM);
      for (int i = 0; i < DIM; i++)
      {
         for (int j = 0; j < DIM; j++)
         {
            ConstDeviceMatrix ddi2b(ie.Get_ddI2b(i,j),DIM,DIM);
            for (int r = 0; r < DIM; r++)
            {
               for (int c = 0; c < DIM; c++)
               {
                  H(r,c,i,j,qx,qy,e) =
                     weight * (0.5 - 0.5/(I2b*I2b)) * ddi2b(r,c) +
                     weight / (I2b*I2b*I2b) * di2b(r,c) * di2b(i,j);
               }
            }
         }
      }
   }
};

struct MetricTMOP_77 : MetricTMOPKer2D
{
   MFEM_HOST_DEVICE
   void AssembleH(const int qx, const int qy, const int e,
                  const double weight, const double (&Jpt)[4],
                  const double *w, const DeviceTensor<7> &H) override
   {
      double dI2[4], dI2b[4], ddI2[4];
      kernels::InvariantsEvaluator2D ie
      (Args().J(Jpt).dI2(dI2).dI2b(dI2b).ddI2(ddI2));
      const double I2 = ie.Get_I2(), I2inv_sq = 1.0 / (I2 * I2);
      ConstDeviceMatrix di2(ie.Get_dI2(),DIM,DIM);
      for (int i = 0; i < DIM; i++)
      {
         for (int j = 0; j < DIM; j++)
         {
            ConstDeviceMatrix ddi2(ie.Get_ddI2(i,j),DIM,DIM);
            for (int r = 0; r < DIM; r++)
            {
               for (int c = 0; c < DIM; c++)
               {
                  H(r,c,i,j,qx,qy,e) =
                     weight * 0.5 * (1.0 - I2inv_sq) * ddi2(r,c) +
                     weight * (I2inv_sq / I2) * di2(r,c) * di2(i,j);
               }
            }
         }
      }
   }
};

struct MetricTMOP_80 : MetricTMOPKer2D
{
   MFEM_HOST_DEVICE
   void AssembleH(const int qx, const int qy, const int e,
                  const double weight, const double (&Jpt)[4],
                  const double *w, const DeviceTensor<7> &H) override
   {
      // w0 H_2 + w1 H_77
      double ddI1[4], ddI1b[4], dI2[4], dI2b[4], ddI2[4];
      kernels::InvariantsEvaluator2D ie
      (Args().J(Jpt).dI2(dI2).ddI1(ddI1).ddI1b(ddI1b).dI2b(dI2b).ddI2(ddI2));

      const double I2 = ie.Get_I2(), I2inv_sq = 1.0 / (I2 * I2);
      ConstDeviceMatrix di2(ie.Get_dI2(),DIM,DIM);
      for (int i = 0; i < DIM; i++)
      {
         for (int j = 0; j < DIM; j++)
         {
            ConstDeviceMatrix ddi1b(ie.Get_ddI1b(i,j),DIM,DIM);
            ConstDeviceMatrix ddi2(ie.Get_ddI2(i,j),DIM,DIM);
            for (int r = 0; r < DIM; r++)
            {
               for (int c = 0; c < DIM; c++)
               {
                  H(r,c,i,j,qx,qy,e) =
                     w[0] * 0.5 * weight * ddi1b(r,c) +
                     w[1] * (weight * 0.5 * (1.0 - I2inv_sq) * ddi2(r,c) +
                             weight * (I2inv_sq / I2) * di2(r,c) * di2(i,j));
               }
            }
         }
      }
   }
};

struct MetricTMOP_94 : MetricTMOPKer2D
{
   MFEM_HOST_DEVICE
   void AssembleH(const int qx, const int qy, const int e,
                  const double weight, const double (&Jpt)[4],
                  const double *w, const DeviceTensor<7> &H) override
   {
      // w0 H_2 + w1 H_56
      double ddI1[4], ddI1b[4], dI2b[4], ddI2b[4];
      kernels::InvariantsEvaluator2D ie
      (Args().J(Jpt).ddI1(ddI1).ddI1b(ddI1b).dI2b(dI2b).ddI2b(ddI2b));
      const double I2b = ie.Get_I2b();
      ConstDeviceMatrix di2b(ie.Get_dI2b(),DIM,DIM);
      for (int i = 0; i < DIM; i++)
      {
         for (int j = 0; j < DIM; j++)
         {
            ConstDeviceMatrix ddi1b(ie.Get_ddI1b(i,j),DIM,DIM);
            ConstDeviceMatrix ddi2b(ie.Get_ddI2b(i,j),DIM,DIM);
            for (int r = 0; r < DIM; r++)
            {
               for (int c = 0; c < DIM; c++)
               {
                  H(r,c,i,j,qx,qy,e) =
                     w[0] * 0.5 * weight * ddi1b(r,c) +
                     w[1] * (weight * (0.5 - 0.5/(I2b*I2b)) * ddi2b(r,c) +
                             weight / (I2b*I2b*I2b) * di2b(r,c) * di2b(i,j) );
               }
            }
         }
      }
   }
};

void TMOP_Integrator::AssembleGradPA_2D(const Vector &x) const
{
   constexpr int DIM = 2;
   const int NE = PA.ne, M = metric->Id();
   const int d = PA.maps->ndof, q = PA.maps->nqpt;
   const double mn = metric_normal;

   Array<double> mp;
   if (auto m = dynamic_cast<TMOP_Combo_QualityMetric *>(metric))
   {
      m->GetWeights(mp);
   }
   const double *w = mp.Read();

   const auto B = Reshape(PA.maps->B.Read(), q,d);
   const auto G = Reshape(PA.maps->G.Read(), q,d);
   const auto W = Reshape(PA.ir->GetWeights().Read(), q,q);
   const auto J = Reshape(PA.Jtr.Read(), DIM,DIM, q,q, NE);
   const auto X = Reshape(x.Read(), d,d, DIM, NE);
   auto H = Reshape(PA.H.Write(), DIM,DIM, DIM,DIM, q,q, NE);

   if (M == 1) { return Launch<MetricTMOP_1>(d,q,X,mn,w,NE,W,B,G,J,H); }
   if (M == 2) { return Launch<MetricTMOP_2>(d,q,X,mn,w,NE,W,B,G,J,H); }
   if (M == 7) { return Launch<MetricTMOP_7>(d,q,X,mn,w,NE,W,B,G,J,H); }
   if (M == 56) { return Launch<MetricTMOP_56>(d,q,X,mn,w,NE,W,B,G,J,H); }
   if (M == 77) { return Launch<MetricTMOP_77>(d,q,X,mn,w,NE,W,B,G,J,H); }
   if (M == 80) { return Launch<MetricTMOP_80>(d,q,X,mn,w,NE,W,B,G,J,H); }
   if (M == 94) { return Launch<MetricTMOP_94>(d,q,X,mn,w,NE,W,B,G,J,H); }
   MFEM_ABORT("Unsupported kernel!");
}

} // namespace mfem
