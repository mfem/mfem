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

#ifndef MFEM_DINVARIANTS_HPP
#define MFEM_DINVARIANTS_HPP

#include "../config/config.hpp"
#include "dtensor.hpp"
#include <cmath>

namespace mfem
{

namespace kernels
{

class InvariantsEvaluator2D
{
public:
   class Buffers
   {
      friend class InvariantsEvaluator2D;
   private:
      const real_t * J_ = nullptr;
      real_t * dI1_ = nullptr;
      real_t * dI1b_ = nullptr;
      real_t * ddI1_ = nullptr;
      real_t * ddI1b_ = nullptr;
      real_t * dI2_ = nullptr;
      real_t * dI2b_ = nullptr;
      real_t * ddI2_ = nullptr;
      real_t * ddI2b_ = nullptr;
   public:
      MFEM_HOST_DEVICE Buffers() {}
      MFEM_HOST_DEVICE Buffers &J(const real_t *b) { J_     = b; return *this; }
      MFEM_HOST_DEVICE Buffers &dI1(real_t *b)     { dI1_   = b; return *this; }
      MFEM_HOST_DEVICE Buffers &dI1b(real_t *b)    { dI1b_  = b; return *this; }
      MFEM_HOST_DEVICE Buffers &ddI1(real_t *b)    { ddI1_  = b; return *this; }
      MFEM_HOST_DEVICE Buffers &ddI1b(real_t *b)   { ddI1b_ = b; return *this; }
      MFEM_HOST_DEVICE Buffers &dI2(real_t *b)     { dI2_   = b; return *this; }
      MFEM_HOST_DEVICE Buffers &dI2b(real_t *b)    { dI2b_  = b; return *this; }
      MFEM_HOST_DEVICE Buffers &ddI2(real_t *b)    { ddI2_  = b; return *this; }
      MFEM_HOST_DEVICE Buffers &ddI2b(real_t *b)   { ddI2b_ = b; return *this; }
   };

private:
   real_t const * const J;
   real_t * const dI1, * const dI1b, * const ddI1, * const ddI1b;
   real_t * const dI2, * const dI2b, * const ddI2, * const ddI2b;

public:
   MFEM_HOST_DEVICE
   InvariantsEvaluator2D(Buffers &b):
      J(b.J_),
      dI1(b.dI1_), dI1b(b.dI1b_), ddI1(b.ddI1_), ddI1b(b.ddI1b_),
      dI2(b.dI2_), dI2b(b.dI2b_), ddI2(b.ddI2_), ddI2b(b.ddI2b_) { }

   MFEM_HOST_DEVICE inline real_t Get_I2b(real_t &sign_detJ) // det(J) + sign
   {
      const real_t I2b = J[0]*J[3] - J[1]*J[2];
      sign_detJ = I2b >= 0.0 ? 1.0 : -1.0;
      return sign_detJ * I2b;
   }

   MFEM_HOST_DEVICE inline real_t Get_I2b() // det(J)
   {
      real_t sign_detJ;
      return Get_I2b(sign_detJ);
   }

   MFEM_HOST_DEVICE inline real_t Get_I2() // det(J)^{2}
   {
      const real_t I2b = Get_I2b();
      return I2b * I2b;
   }

   MFEM_HOST_DEVICE inline real_t Get_I1() // I1 = ||J||_F^2
   {
      return J[0]*J[0] + J[1]*J[1] + J[2]*J[2] + J[3]*J[3];
   }

   MFEM_HOST_DEVICE inline real_t Get_I1b() // I1b = I1/det(J)
   {
      return Get_I1() / Get_I2b();
   }

   MFEM_HOST_DEVICE inline real_t *Get_dI1()
   {
      dI1[0] = 2*J[0]; dI1[2] = 2*J[2];
      dI1[1] = 2*J[1]; dI1[3] = 2*J[3];
      return dI1;
   }

   // Requires dI2b.
   MFEM_HOST_DEVICE inline real_t *Get_dI1b()
   {
      // I1b = I1/I2b
      // dI1b = (1/I2b)*dI1 - (I1/I2b^2)*dI2b = (2/I2b)*[J - (I1b/2)*dI2b]
      const real_t c1 = 2.0/Get_I2b();
      const real_t c2 = Get_I1b()/2.0;
      Get_dI2b();
      dI1b[0] = c1*(J[0] - c2*dI2b[0]);
      dI1b[1] = c1*(J[1] - c2*dI2b[1]);
      dI1b[2] = c1*(J[2] - c2*dI2b[2]);
      dI1b[3] = c1*(J[3] - c2*dI2b[3]);
      return dI1b;
   }

   // Requires dI2b.
   MFEM_HOST_DEVICE inline real_t *Get_dI2()
   {
      // I2 = I2b^2
      // dI2 = 2*I2b*dI2b = 2*det(J)*adj(J)^T
      const real_t c1 = 2*Get_I2b();
      Get_dI2b();
      dI2[0] = c1*dI2b[0];
      dI2[1] = c1*dI2b[1];
      dI2[2] = c1*dI2b[2];
      dI2[3] = c1*dI2b[3];
      return dI2;
   }

   MFEM_HOST_DEVICE inline real_t *Get_dI2b()
   {
      // I2b = det(J)
      // dI2b = adj(J)^T
      real_t sign_detJ;
      Get_I2b(sign_detJ);
      dI2b[0] =  sign_detJ*J[3];
      dI2b[1] = -sign_detJ*J[2];
      dI2b[2] = -sign_detJ*J[1];
      dI2b[3] =  sign_detJ*J[0];
      return dI2b;
   }

   // ddI1_ijkl = 2 I_ijkl = 2 δ_ik δ_jl
   MFEM_HOST_DEVICE inline real_t *Get_ddI1(int i, int j)
   {
      // ddI1_ijkl = 2 I_ijkl = 2 δ_ik δ_jl
      DeviceMatrix ddi1(ddI1,2,2);
      for (int k=0; k<2; k++)
      {
         for (int l=0; l<2; l++)
         {
            ddi1(k,l) = (i==k && j==l) ? 2.0 : 0.0;
         }
      }
      return ddI1;
   }

   // Requires dI2b + ddI1.
   // ddI1b = X1 + X2 + X3, where
   // X1_ijkl = (I1b/I2) [ dI2b_ij dI2b_kl + dI2b_kj dI2b_il ]
   // X2_ijkl = (1/I2b) ddI1_ijkl
   // X3_ijkl = -(2/I2) (J_ij dI2b_kl + dI2b_ij J_kl)
   MFEM_HOST_DEVICE inline real_t *Get_ddI1b(int i, int j)
   {
      real_t X1_p[4], X2_p[4], X3_p[4];

      // X1_ijkl = (I1b/I2) [ dI2b_ij dI2b_kl + dI2b_kj dI2b_il ]
      const real_t I2 = Get_I2();
      const real_t I1b = Get_I1b();
      ConstDeviceMatrix di2b(Get_dI2b(),2,2);
      const real_t alpha = I1b / I2;
      DeviceMatrix X1(X1_p,2,2);
      for (int k=0; k<2; k++)
      {
         for (int l=0; l<2; l++)
         {
            X1(k,l) = alpha * (di2b(i,j)*di2b(k,l) + di2b(k,j)*di2b(i,l));
         }
      }
      // X2_ijkl = (1/I2b) ddI1_ijkl
      DeviceMatrix X2(X2_p,2,2);
      const real_t beta = 1.0 / Get_I2b();
      ConstDeviceMatrix ddi1(Get_ddI1(i,j),2,2);
      for (int k=0; k<2; k++)
      {
         for (int l=0; l<2; l++)
         {
            X2(k,l) = beta * ddi1(k,l);
         }
      }
      // X3_ijkl = -(2/I2) (J_ij dI2b_kl + dI2b_ij J_kl)
      DeviceMatrix X3(X3_p,2,2);
      const real_t gamma = -2.0/Get_I2();
      ConstDeviceMatrix Jpt(J,2,2);
      for (int k=0; k<2; k++)
      {
         for (int l=0; l<2; l++)
         {
            X3(k,l) = gamma * (Jpt(i,j)*di2b(k,l) + di2b(i,j)*Jpt(k,l));
         }
      }
      DeviceMatrix ddi1b(ddI1b,2,2);
      for (int k=0; k<2; k++)
      {
         for (int l=0; l<2; l++)
         {
            ddi1b(k,l) = X1(k,l) + X2(k,l) + X3(k,l);
         }
      }
      return ddI1b;
   }

   // Requires dI2b.
   // ddI2_ijkl = 2 dI2b_ij dI2b_kl + 2 (dI2b_ij dI2b_kl - dI2b_kj dI2b_il)
   MFEM_HOST_DEVICE inline real_t *Get_ddI2(int i, int j)
   {
      DeviceMatrix ddi2(ddI2,2,2);
      ConstDeviceMatrix di2b(Get_dI2b(),2,2);
      for (int k=0; k<2; k++)
      {
         for (int l=0; l<2; l++)
         {
            ddi2(k,l) = 2*di2b(i,j)*di2b(k,l)
                        + 2*(di2b(i,j)*di2b(k,l) - di2b(k,j)*di2b(i,l));
         }
      }
      return ddI2;
   }

   // Requires dI2b.
   // ddI2b_ijkl = (1/I2b) (δ_ks δ_it - δ_kt δ_si) dI2b_tj dI2b_sl
   MFEM_HOST_DEVICE inline real_t *Get_ddI2b(int i, int j)
   {
      DeviceMatrix ddi2b(ddI2b,2,2);
      const real_t alpha = 1.0/Get_I2b();
      ConstDeviceMatrix di2b(Get_dI2b(),2,2);
      for (int k=0; k<2; k++)
      {
         for (int l=0; l<2; l++)
         {
            ddi2b(k,l) = 0.0;
            for (int s=0; s<2; s++)
            {
               for (int t=0; t<2; t++)
               {
                  const real_t ks_it = k==s && i==t ? 1.0 : 0.0;
                  const real_t kt_si = k==t && s==i ? 1.0 : 0.0;
                  ddi2b(k,l) += alpha * (ks_it - kt_si) * di2b(t,j) * di2b(s,l);
               }
            }
         }
      }
      return ddI2b;
   }
};


class InvariantsEvaluator3D
{
public:
   class Buffers
   {
      friend class InvariantsEvaluator3D;
   private:
      const real_t * J_ = nullptr;
      real_t * B_ = nullptr;
      real_t * dI1_ = nullptr;
      real_t * dI1b_ = nullptr;
      real_t * ddI1_ = nullptr;
      real_t * ddI1b_ = nullptr;
      real_t * dI2_ = nullptr;
      real_t * dI2b_ = nullptr;
      real_t * ddI2_ = nullptr;
      real_t * ddI2b_ = nullptr;
      real_t * dI3b_ = nullptr;
      real_t * ddI3b_ = nullptr;
   public:
      MFEM_HOST_DEVICE Buffers() {}
      MFEM_HOST_DEVICE Buffers &J(const real_t *b) { J_     = b; return *this; }
      MFEM_HOST_DEVICE Buffers &B(real_t *b)       { B_     = b; return *this; }
      MFEM_HOST_DEVICE Buffers &dI1(real_t *b)     { dI1_   = b; return *this; }
      MFEM_HOST_DEVICE Buffers &dI1b(real_t *b)    { dI1b_  = b; return *this; }
      MFEM_HOST_DEVICE Buffers &ddI1(real_t *b)    { ddI1_  = b; return *this; }
      MFEM_HOST_DEVICE Buffers &ddI1b(real_t *b)   { ddI1b_ = b; return *this; }
      MFEM_HOST_DEVICE Buffers &dI2(real_t *b)     { dI2_   = b; return *this; }
      MFEM_HOST_DEVICE Buffers &dI2b(real_t *b)    { dI2b_  = b; return *this; }
      MFEM_HOST_DEVICE Buffers &ddI2(real_t *b)    { ddI2_  = b; return *this; }
      MFEM_HOST_DEVICE Buffers &ddI2b(real_t *b)   { ddI2b_ = b; return *this; }
      MFEM_HOST_DEVICE Buffers &dI3b(real_t *b)    { dI3b_  = b; return *this; }
      MFEM_HOST_DEVICE Buffers &ddI3b(real_t *b)   { ddI3b_ = b; return *this; }
   };

private:
   real_t const * const J;
   real_t * const B;
   real_t * const dI1, * const dI1b, * const ddI1, * const ddI1b;
   real_t * const dI2, * const dI2b, * const ddI2, * const ddI2b;
   real_t * const dI3b, * const ddI3b;

public:
   MFEM_HOST_DEVICE
   InvariantsEvaluator3D(Buffers &b):
      J(b.J_), B(b.B_),
      dI1(b.dI1_), dI1b(b.dI1b_), ddI1(b.ddI1_), ddI1b(b.ddI1b_),
      dI2(b.dI2_), dI2b(b.dI2b_), ddI2(b.ddI2_), ddI2b(b.ddI2b_),
      dI3b(b.dI3b_), ddI3b(b.ddI3b_) { }

   MFEM_HOST_DEVICE inline real_t Get_I3b(real_t &sign_detJ) // det(J) + sign
   {
      const real_t I3b = + J[0]*(J[4]*J[8] - J[7]*J[5])
                         - J[1]*(J[3]*J[8] - J[5]*J[6])
                         + J[2]*(J[3]*J[7] - J[4]*J[6]);
      sign_detJ = I3b >= 0.0 ? 1.0 : -1.0;
      return sign_detJ * I3b;
   }

   MFEM_HOST_DEVICE inline real_t Get_I3b() // det(J)
   {
      const real_t I3b = + J[0]*(J[4]*J[8] - J[7]*J[5])
                         - J[1]*(J[3]*J[8] - J[5]*J[6])
                         + J[2]*(J[3]*J[7] - J[4]*J[6]);
      return I3b;
   }

   MFEM_HOST_DEVICE inline real_t Get_I3() // det(J)^{2}
   {
      const real_t I3b = Get_I3b();
      return I3b * I3b;
   }

   MFEM_HOST_DEVICE inline real_t Get_I3b_p() // I3b^{-2/3}
   {
      real_t sign_detJ;
      const real_t i3b = Get_I3b(sign_detJ);
      return sign_detJ * std::pow(i3b, -2./3.);
   }

   MFEM_HOST_DEVICE inline real_t Get_I3b_p(real_t &sign_detJ) // I3b^{-2/3}
   {
      const real_t i3b = Get_I3b(sign_detJ);
      return sign_detJ * std::pow(i3b, -2./3.);
   }

   MFEM_HOST_DEVICE inline real_t Get_I1()
   {
      B[0] = J[0]*J[0] + J[3]*J[3] + J[6]*J[6];
      B[1] = J[1]*J[1] + J[4]*J[4] + J[7]*J[7];
      B[2] = J[2]*J[2] + J[5]*J[5] + J[8]*J[8];
      const real_t I1 = B[0] + B[1] + B[2];
      return I1;
   }

   MFEM_HOST_DEVICE inline
   real_t Get_I1b() // det(J)^{-2/3}*I_1 = I_1/I_3^{1/3}
   {
      const real_t I1b = Get_I1() * Get_I3b_p();
      return I1b;
   }

   MFEM_HOST_DEVICE inline void Get_B_offd()
   {
      // B = J J^t
      // B[3]=B(0,1), B[4]=B(0,2), B[5]=B(1,2)
      B[3] = J[0]*J[1] + J[3]*J[4] + J[6]*J[7]; // B(0,1)
      B[4] = J[0]*J[2] + J[3]*J[5] + J[6]*J[8]; // B(0,2)
      B[5] = J[1]*J[2] + J[4]*J[5] + J[7]*J[8]; // B(1,2)
   }

   MFEM_HOST_DEVICE inline real_t Get_I2()
   {
      Get_B_offd();
      const real_t I1 = Get_I1();
      const real_t BF2 = B[0]*B[0] + B[1]*B[1] + B[2]*B[2] +
                         2*(B[3]*B[3] + B[4]*B[4] + B[5]*B[5]);
      const real_t I2 = (I1*I1 - BF2)/2;
      return I2;
   }

   MFEM_HOST_DEVICE inline real_t Get_I2b() // I2b = I2*I3b^{-4/3}
   {
      const real_t I3b_p = Get_I3b_p();
      return Get_I2() * I3b_p * I3b_p;
   }

   MFEM_HOST_DEVICE inline real_t *Get_dI1()
   {
      for (int i = 0; i < 9; i++) { dI1[i] = 2*J[i]; }
      return dI1;
   }

   MFEM_HOST_DEVICE inline real_t *Get_dI1b()
   {
      // I1b = I3b^{-2/3}*I1
      // dI1b = 2*I3b^{-2/3}*(J - (1/3)*I1/I3b*dI3b)
      real_t sign_detJ;
      const real_t I3b = Get_I3b(sign_detJ);
      const real_t I3b_p = Get_I3b_p();
      const real_t c1 = 2.0 * I3b_p;
      const real_t c2 = Get_I1()/(3.0 * I3b);
      Get_dI3b(sign_detJ);
      for (int i = 0; i < 9; i++) { dI1b[i] = c1*(J[i] - c2*dI3b[i]); }
      return dI1b;
   }

   MFEM_HOST_DEVICE inline real_t *Get_dI2()
   {
      // dI2 = 2 I_1 J - 2 J J^t J = 2 (I_1 I - B) J
      const real_t I1 = Get_I1();
      Get_B_offd();
      // B[0]=B(0,0), B[1]=B(1,1), B[2]=B(2,2)
      // B[3]=B(0,1), B[4]=B(0,2), B[5]=B(1,2)
      const real_t C[6] =
      {
         2*(I1 - B[0]), 2*(I1 - B[1]), 2*(I1 - B[2]),
         -2*B[3], -2*B[4], -2*B[5]
      };
      //       | C[0]  C[3]  C[4] |  | J[0]  J[3]  J[6] |
      // dI2 = | C[3]  C[1]  C[5] |  | J[1]  J[4]  J[7] |
      //       | C[4]  C[5]  C[2] |  | J[2]  J[5]  J[8] |
      dI2[0] = C[0]*J[0] + C[3]*J[1] + C[4]*J[2];
      dI2[1] = C[3]*J[0] + C[1]*J[1] + C[5]*J[2];
      dI2[2] = C[4]*J[0] + C[5]*J[1] + C[2]*J[2];

      dI2[3] = C[0]*J[3] + C[3]*J[4] + C[4]*J[5];
      dI2[4] = C[3]*J[3] + C[1]*J[4] + C[5]*J[5];
      dI2[5] = C[4]*J[3] + C[5]*J[4] + C[2]*J[5];

      dI2[6] = C[0]*J[6] + C[3]*J[7] + C[4]*J[8];
      dI2[7] = C[3]*J[6] + C[1]*J[7] + C[5]*J[8];
      dI2[8] = C[4]*J[6] + C[5]*J[7] + C[2]*J[8];
      return dI2;
   }

   MFEM_HOST_DEVICE inline real_t *Get_dI2b()
   {
      // I2b = det(J)^{-4/3}*I2 = I3b^{-4/3}*I2
      // dI2b = (-4/3)*I3b^{-7/3}*I2*dI3b + I3b^{-4/3}*dI2
      //      = I3b^{-4/3} * [ dI2 - (4/3)*I2/I3b*dI3b ]
      real_t sign_detJ;
      const real_t I2 = Get_I2();
      const real_t I3b_p = Get_I3b_p();
      const real_t I3b = Get_I3b(sign_detJ);
      const real_t c1 = I3b_p*I3b_p;
      const real_t c2 = (4*I2/I3b)/3;
      Get_dI2();
      Get_dI3b(sign_detJ);
      for (int i = 0; i < 9; i++) { dI2b[i] = c1*(dI2[i] - c2*dI3b[i]); }
      return dI2b;
   }

   MFEM_HOST_DEVICE inline real_t *Get_dI3b(const real_t sign_detJ)
   {
      // I3b = det(J)
      // dI3b = adj(J)^T
      dI3b[0] = sign_detJ*(J[4]*J[8] - J[5]*J[7]); // 0  3  6
      dI3b[1] = sign_detJ*(J[5]*J[6] - J[3]*J[8]); // 1  4  7
      dI3b[2] = sign_detJ*(J[3]*J[7] - J[4]*J[6]); // 2  5  8
      dI3b[3] = sign_detJ*(J[2]*J[7] - J[1]*J[8]);
      dI3b[4] = sign_detJ*(J[0]*J[8] - J[2]*J[6]);
      dI3b[5] = sign_detJ*(J[1]*J[6] - J[0]*J[7]);
      dI3b[6] = sign_detJ*(J[1]*J[5] - J[2]*J[4]);
      dI3b[7] = sign_detJ*(J[2]*J[3] - J[0]*J[5]);
      dI3b[8] = sign_detJ*(J[0]*J[4] - J[1]*J[3]);
      return dI3b;
   }

   // ddI1_ijkl = 2 I_ijkl = 2 δ_ik δ_jl
   MFEM_HOST_DEVICE inline real_t *Get_ddI1(int i, int j)
   {
      DeviceMatrix ddi1(ddI1,3,3);
      for (int k=0; k<3; k++)
      {
         for (int l=0; l<3; l++)
         {
            const real_t I_ijkl = (i==k && j==l) ? 1.0 : 0.0;
            ddi1(k,l) = 2.0 * I_ijkl;
         }
      }
      return ddI1;
   }

   // ddI1b = X1 + X2 + X3, where
   // X1_ijkl = (2/3*I1b/I3) [ 2/3 dI3b_ij dI3b_kl + dI3b_kj dI3b_il ]
   // X2_ijkl = (I3b^{-2/3}) ddI1_ijkl
   // X3_ijkl = -(4/3*I3b^{-5/3}) (J_ij dI3b_kl + dI3b_ij J_kl)
   MFEM_HOST_DEVICE inline real_t *Get_ddI1b(int i, int j)
   {
      // X1_ijkl = (2/3*I1b/I3) [ 2/3 dI3b_ij dI3b_kl + dI3b_kj dI3b_il ]
      real_t sign_detJ;
      Get_I3b(sign_detJ);
      real_t X1_p[9], X2_p[9], X3_p[9];
      DeviceMatrix X1(X1_p,3,3);
      const real_t I3 = Get_I3();
      const real_t I1b = Get_I1b();
      const real_t alpha = (2./3.)*I1b/I3;
      ConstDeviceMatrix di3b(Get_dI3b(sign_detJ),3,3);
      for (int k=0; k<3; k++)
      {
         for (int l=0; l<3; l++)
         {
            X1(k,l) = alpha * ((2./3.)*di3b(i,j) * di3b(k,l) +
                               di3b(k,j)*di3b(i,l));
         }
      }
      // ddI1_ijkl = 2 δ_ik δ_jl
      // X2_ijkl = (I3b^{-2/3}) ddI1_ijkl
      DeviceMatrix X2(X2_p,3,3);
      const real_t beta = Get_I3b_p();
      for (int k=0; k<3; k++)
      {
         for (int l=0; l<3; l++)
         {
            const real_t ddI1_ijkl = (i==k && j==l) ? 2.0 : 0.0;
            X2(k,l) = beta * ddI1_ijkl;
         }
      }
      // X3_ijkl = -(4/3*I3b^{-5/3}) (J_ij dI3b_kl + dI3b_ij J_kl)
      DeviceMatrix X3(X3_p,3,3);
      const real_t I3b = Get_I3b();
      const real_t gamma = -(4./3.)*Get_I3b_p()/I3b;
      ConstDeviceMatrix Jpt(J,3,3);
      for (int k=0; k<3; k++)
      {
         for (int l=0; l<3; l++)
         {
            X3(k,l) = gamma * (Jpt(i,j) * di3b(k,l) + di3b(i,j) * Jpt(k,l));
         }
      }
      DeviceMatrix ddi1b(ddI1b,3,3);
      for (int k=0; k<3; k++)
      {
         for (int l=0; l<3; l++)
         {
            ddi1b(k,l) = X1(k,l) + X2(k,l) + X3(k,l);
         }
      }
      return ddI1b;
   }

   // ddI2 = x1 + x2 + x3
   //    x1_ijkl = (2 I1) δ_ik δ_jl
   //    x2_ijkl = 2 ( 2 δ_ku δ_iv - δ_ik δ_uv - δ_kv δ_iu ) J_vj J_ul
   //    x3_ijkl = -2 (J J^t)_ik δ_jl = -2 B_ik δ_jl
   MFEM_HOST_DEVICE inline real_t *Get_ddI2(int i, int j)
   {
      real_t x1_p[9], x2_p[9], x3_p[9];
      DeviceMatrix x1(x1_p,3,3), x2(x2_p,3,3), x3(x3_p,3,3);
      // x1_ijkl = (2 I1) δ_ik δ_jl
      const real_t I1 = Get_I1();
      for (int k=0; k<3; k++)
      {
         for (int l=0; l<3; l++)
         {
            const real_t ik_jl = (i==k && j==l) ? 1.0 : 0.0;
            x1(k,l) = 2.0 * I1 * ik_jl;
         }
      }
      // x2_ijkl = 2 ( 2 δ_ku δ_iv - δ_ik δ_uv - δ_kv δ_iu ) J_vj J_ul
      ConstDeviceMatrix Jpt(J,3,3);
      for (int k=0; k<3; k++)
      {
         for (int l=0; l<3; l++)
         {
            x2(k,l) = 0.0;
            for (int u=0; u<3; u++)
            {
               for (int v=0; v<3; v++)
               {
                  const real_t ku_iv = k==u && i==v ? 1.0 : 0.0;
                  const real_t ik_uv = i==k && u==v ? 1.0 : 0.0;
                  const real_t kv_iu = k==v && i==u ? 1.0 : 0.0;
                  x2(k,l) += 2.0*(2.*ku_iv-ik_uv-kv_iu)*Jpt(v,j)*Jpt(u,l);
               }
            }
         }
      }
      // x3_ijkl = -2 B_ik δ_jl
      B[0] = J[0]*J[0] + J[3]*J[3] + J[6]*J[6];
      B[1] = J[1]*J[1] + J[4]*J[4] + J[7]*J[7];
      B[2] = J[2]*J[2] + J[5]*J[5] + J[8]*J[8];
      B[3] = J[0]*J[1] + J[3]*J[4] + J[6]*J[7]; // B(0,1)
      B[4] = J[0]*J[2] + J[3]*J[5] + J[6]*J[8]; // B(0,2)
      B[5] = J[1]*J[2] + J[4]*J[5] + J[7]*J[8]; // B(1,2)
      const real_t b_p[9] =
      {
         B[0], B[3], B[4],
         B[3], B[1], B[5],
         B[4], B[5], B[2]
      };
      ConstDeviceMatrix b(b_p,3,3);
      for (int k=0; k<3; k++)
      {
         for (int l=0; l<3; l++)
         {
            const real_t jl = j==l ? 1.0 : 0.0;
            x3(k,l) = -2.0 * b(i,k) * jl;
         }
      }
      // ddI2 = x1 + x2 + x3
      DeviceMatrix ddi2(ddI2,3,3);
      for (int k=0; k<3; k++)
      {
         for (int l=0; l<3; l++)
         {
            ddi2(k,l) = x1(k,l) + x2(k,l) + x3(k,l);
         }
      }
      return ddI2;
   }

   // ddI2b = X1 + X2 + X3
   //    X1_ijkl = 16/9 det(J)^{-10/3} I2 dI3b_ij dI3b_kl +
   //               4/3 det(J)^{-10/3} I2 dI3b_il dI3b_kj
   //    X2_ijkl = -4/3 det(J)^{-7/3} (dI2_ij dI3b_kl + dI2_kl dI3b_ij)
   //    X3_ijkl =      det(J)^{-4/3} ddI2_ijkl
   MFEM_HOST_DEVICE inline real_t *Get_ddI2b(int i, int j)
   {
      real_t X1_p[9], X2_p[9], X3_p[9];
      // X1_ijkl = 16/9 det(J)^{-10/3} I2 dI3b_ij dI3b_kl +
      //               4/3 det(J)^{-10/3} I2 dI3b_il dI3b_kj
      real_t sign_detJ;
      DeviceMatrix X1(X1_p,3,3);
      const real_t I3b_p = Get_I3b_p(); // I3b^{-2/3}
      const real_t I3b = Get_I3b(sign_detJ); // det(J)
      const real_t I2 = Get_I2();
      const real_t I3b_p43 = I3b_p*I3b_p;
      const real_t I3b_p73 = I3b_p*I3b_p/I3b;
      const real_t I3b_p103 = I3b_p*I3b_p/(I3b*I3b);
      ConstDeviceMatrix di3b(Get_dI3b(sign_detJ),3,3);
      for (int k=0; k<3; k++)
      {
         for (int l=0; l<3; l++)
         {
            const real_t up = (16./9.)*I3b_p103*I2*di3b(i,j)*di3b(k,l);
            const real_t down = (4./3.)*I3b_p103*I2*di3b(i,l)*di3b(k,j);
            X1(k,l) = up + down;
         }
      }
      // X2_ijkl = -4/3 det(J)^{-7/3} (dI2_ij dI3b_kl + dI2_kl dI3b_ij)
      DeviceMatrix X2(X2_p,3,3);
      ConstDeviceMatrix di2(Get_dI2(),3,3);
      for (int k=0; k<3; k++)
      {
         for (int l=0; l<3; l++)
         {
            X2(k,l) = -(4./3.)*I3b_p73*(di2(i,j)*di3b(k,l)+di2(k,l)*di3b(i,j));
         }
      }
      // X3_ijkl =  det(J)^{-4/3} ddI2_ijkl
      DeviceMatrix X3(X3_p,3,3);
      ConstDeviceMatrix ddi2(Get_ddI2(i,j),3,3);
      for (int k=0; k<3; k++)
      {
         for (int l=0; l<3; l++)
         {
            X3(k,l) = I3b_p43 * ddi2(k,l);
         }
      }
      // ddI2b = X1 + X2 + X3
      DeviceMatrix ddi2b(ddI2b,3,3);
      for (int k=0; k<3; k++)
      {
         for (int l=0; l<3; l++)
         {
            ddi2b(k,l) = X1(k,l) + X2(k,l) + X3(k,l);
         }
      }
      return ddI2b;
   }

   // dI3b = adj(J)^T
   // ddI3b_ijkl = (1/I3b) (δ_ks δ_it - δ_kt δ_si) dI3b_tj dI3b_sl
   MFEM_HOST_DEVICE inline real_t *Get_ddI3b(int i, int j)
   {
      const real_t c1 = 1./Get_I3b();
      ConstDeviceMatrix di3b(dI3b,3,3);
      DeviceMatrix ddi3b(ddI3b,3,3);
      for (int k=0; k<3; k++)
      {
         for (int l=0; l<3; l++)
         {
            ddi3b(k,l) = 0.0;
            for (int s=0; s<3; s++)
            {
               for (int t=0; t<3; t++)
               {
                  const real_t ks_it = k==s && i==t ? 1.0 : 0.0;
                  const real_t kt_si = k==t && s==i ? 1.0 : 0.0;
                  ddi3b(k,l) += c1*(ks_it-kt_si)*di3b(t,j)*di3b(s,l);
               }
            }
         }
      }
      return ddI3b;
   }
};

} // namespace kernels

} // namespace mfem

#endif // MFEM_DINVARIANTS_HPP
