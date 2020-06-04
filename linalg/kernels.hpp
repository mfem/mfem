// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_KERNELS_HPP
#define MFEM_KERNELS_HPP

#ifdef _WIN32
#define _USE_MATH_DEFINES
#include <cmath>
#endif

#include "../config/config.hpp"
#include "../general/cuda.hpp"
#include "../general/globals.hpp"

#include "matrix.hpp"
#include "tmatrix.hpp"
#include "tlayout.hpp"
#include "ttensor.hpp"
#include "dtensor.hpp"

// This header contains stand-alone functions for "small" dense linear algebra
// (at quadrature point or element-level) designed to be inlined directly into
// device kernels.

// Many methods of the DenseMatrix class and some of the Vector class call these
// kernels directly on the host, see the implementations in linalg/densemat.cpp
// and linalag.vector.cpp.

namespace mfem
{

namespace kernels
{

class InvariantsEvaluator3D
{
public:
   // Transformation Jacobian
   const double *J;

   // Invariants:
   //    I_1 = ||J||_F^2, \bar{I}_1 = det(J)^{-2/3}*I_1,
   //    I_2 = (1/2)*(||J||_F^4-||J J^t||_F^2) = (1/2)*(I_1^2-||J J^t||_F^2),
   //    \bar{I}_2 = det(J)^{-4/3}*I_2,
   //    I_3 = det(J)^2, \bar{I}_3 = det(J).
   double I1, I1b, I2, I2b, I3b;
   double I3b_p; // I3b^{-2/3}

   // Derivatives of I1, I1b, I2, I2b, I3, and I3b using column-major storage.
   double dI1[9], dI1b[9], dI2[9], dI2b[9], dI3[9], dI3b[9];
   double B[6]; // B = J J^t (diagonal entries first, then off-diagonal)

   double sign_detJ;

   double ddI1b_ij[9], ddI2_ij[9], ddI2b_ij[9];

   enum EvalMasks
   {
      HAVE_I1     = 1,
      HAVE_I1b    = 2,
      HAVE_B_offd = 4,
      HAVE_I2     = 8,
      HAVE_I2b    = 16,
      HAVE_I3b    = 1<<5,
      HAVE_I3b_p  = 1<<6,
      HAVE_dI1    = 1<<7,
      HAVE_dI1b   = 1<<8,
      HAVE_dI2    = 1<<9,
      HAVE_dI2b   = 1<<10,
      HAVE_dI3    = 1<<11,
      HAVE_dI3b   = 1<<12,
   };

   // Bitwise OR of EvalMasks
   int eval_state;

   MFEM_HOST_DEVICE bool dont(int have_mask) const { return !(eval_state & have_mask); }

   MFEM_HOST_DEVICE void Eval_I1()
   {
      eval_state |= HAVE_I1;
      B[0] = J[0]*J[0] + J[3]*J[3] + J[6]*J[6];
      B[1] = J[1]*J[1] + J[4]*J[4] + J[7]*J[7];
      B[2] = J[2]*J[2] + J[5]*J[5] + J[8]*J[8];
      I1 = B[0] + B[1] + B[2];
   }

   MFEM_HOST_DEVICE void Eval_I1b() // det(J)^{-2/3}*I_1 = I_1/I_3^{1/3}
   {
      eval_state |= HAVE_I1b;
      I1b = Get_I1()*Get_I3b_p();
   }

   MFEM_HOST_DEVICE void Eval_B_offd()
   {
      eval_state |= HAVE_B_offd;
      // B = J J^t
      // B[3]=B(0,1), B[4]=B(0,2), B[5]=B(1,2)
      B[3] = J[0]*J[1] + J[3]*J[4] + J[6]*J[7]; // B(0,1)
      B[4] = J[0]*J[2] + J[3]*J[5] + J[6]*J[8]; // B(0,2)
      B[5] = J[1]*J[2] + J[4]*J[5] + J[7]*J[8]; // B(1,2)
   }
   MFEM_HOST_DEVICE void Eval_I2()
   {
      eval_state |= HAVE_I2;
      Get_I1();
      if (dont(HAVE_B_offd)) { Eval_B_offd(); }
      const double BF2 = B[0]*B[0] + B[1]*B[1] + B[2]*B[2] +
                         2*(B[3]*B[3] + B[4]*B[4] + B[5]*B[5]);
      I2 = (I1*I1 - BF2)/2;
   }
   MFEM_HOST_DEVICE void Eval_I2b() // I2b = I2*I3b^{-4/3}
   {
      eval_state |= HAVE_I2b;
      Get_I3b_p();
      I2b = Get_I2()*I3b_p*I3b_p;
   }
   MFEM_HOST_DEVICE void Eval_I3b() // det(J)
   {
      eval_state |= HAVE_I3b;
      I3b = J[0]*(J[4]*J[8] - J[7]*J[5]) - J[1]*(J[3]*J[8] - J[5]*J[6]) +
            J[2]*(J[3]*J[7] - J[4]*J[6]);
      sign_detJ = I3b >= 0.0 ? 1.0 : -1.0;
      I3b = sign_detJ*I3b;
   }
   MFEM_HOST_DEVICE double Get_I3b_p()  // I3b^{-2/3}
   {
      if (dont(HAVE_I3b_p))
      {
         eval_state |= HAVE_I3b_p;
         const double i3b = Get_I3b();
         I3b_p = sign_detJ * std::pow(i3b, -2./3.);
      }
      return I3b_p;
   }
   MFEM_HOST_DEVICE void Eval_dI1()
   {
      eval_state |= HAVE_dI1;
      for (int i = 0; i < 9; i++)
      {
         dI1[i] = 2*J[i];
      }
   }
   MFEM_HOST_DEVICE void Eval_dI1b()
   {
      eval_state |= HAVE_dI1b;
      // I1b = I3b^{-2/3}*I1
      // dI1b = 2*I3b^{-2/3}*(J - (1/3)*I1/I3b*dI3b)
      const double c1 = 2*Get_I3b_p();
      const double c2 = Get_I1()/(3*I3b);
      Get_dI3b();
      for (int i = 0; i < 9; i++)
      {
         dI1b[i] = c1*(J[i] - c2*dI3b[i]);
      }
   }
   MFEM_HOST_DEVICE void Eval_dI2()
   {
      eval_state |= HAVE_dI2;
      // dI2 = 2 I_1 J - 2 J J^t J = 2 (I_1 I - B) J
      Get_I1();
      if (dont(HAVE_B_offd)) { Eval_B_offd(); }
      // B[0]=B(0,0), B[1]=B(1,1), B[2]=B(2,2)
      // B[3]=B(0,1), B[4]=B(0,2), B[5]=B(1,2)
      const double C[6] =
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
   }
   MFEM_HOST_DEVICE void Eval_dI2b()
   {
      eval_state |= HAVE_dI2b;
      // I2b = det(J)^{-4/3}*I2 = I3b^{-4/3}*I2
      // dI2b = (-4/3)*I3b^{-7/3}*I2*dI3b + I3b^{-4/3}*dI2
      //      = I3b^{-4/3} * [ dI2 - (4/3)*I2/I3b*dI3b ]
      Get_I3b_p();
      const double c1 = I3b_p*I3b_p;
      const double c2 = (4*Get_I2()/I3b)/3;
      Get_dI2();
      Get_dI3b();
      for (int i = 0; i < 9; i++)
      {
         dI2b[i] = c1*(dI2[i] - c2*dI3b[i]);
      }
   }
   MFEM_HOST_DEVICE void Eval_dI3()
   {
      eval_state |= HAVE_dI3;
      // I3 = I3b^2
      // dI3 = 2*I3b*dI3b = 2*det(J)*adj(J)^T
      const double c1 = 2*Get_I3b();
      Get_dI3b();
      for (int i = 0; i < 9; i++)
      {
         dI3[i] = c1*dI3b[i];
      }
   }
   MFEM_HOST_DEVICE void Eval_dI3b()
   {
      eval_state |= HAVE_dI3b;
      // I3b = det(J)
      // dI3b = adj(J)^T
      dI3b[0] = sign_detJ*(J[4]*J[8] - J[5]*J[7]);  // 0  3  6
      dI3b[1] = sign_detJ*(J[5]*J[6] - J[3]*J[8]);  // 1  4  7
      dI3b[2] = sign_detJ*(J[3]*J[7] - J[4]*J[6]);  // 2  5  8
      dI3b[3] = sign_detJ*(J[2]*J[7] - J[1]*J[8]);
      dI3b[4] = sign_detJ*(J[0]*J[8] - J[2]*J[6]);
      dI3b[5] = sign_detJ*(J[1]*J[6] - J[0]*J[7]);
      dI3b[6] = sign_detJ*(J[1]*J[5] - J[2]*J[4]);
      dI3b[7] = sign_detJ*(J[2]*J[3] - J[0]*J[5]);
      dI3b[8] = sign_detJ*(J[0]*J[4] - J[1]*J[3]);
   }

public:
   /// The Jacobian should use column-major storage.
   MFEM_HOST_DEVICE  InvariantsEvaluator3D(const double *J): J(J), eval_state(0) { }

   MFEM_HOST_DEVICE double Get_I1()  { if (dont(HAVE_I1 )) { Eval_I1();  } return I1; }
   MFEM_HOST_DEVICE double Get_I1b() { if (dont(HAVE_I1b)) { Eval_I1b(); } return I1b; }
   MFEM_HOST_DEVICE double Get_I2()  { if (dont(HAVE_I2 )) { Eval_I2();  } return I2; }
   MFEM_HOST_DEVICE double Get_I2b() { if (dont(HAVE_I2b)) { Eval_I2b(); } return I2b; }
   MFEM_HOST_DEVICE double Get_I3()  { if (dont(HAVE_I3b)) { Eval_I3b(); } return I3b*I3b; }
   MFEM_HOST_DEVICE double Get_I3b() { if (dont(HAVE_I3b)) { Eval_I3b(); } return I3b; }

   MFEM_HOST_DEVICE const double *Get_dI1()
   {
      if (dont(HAVE_dI1 )) { Eval_dI1();  } return dI1;
   }
   MFEM_HOST_DEVICE const double *Get_dI1b()
   {
      if (dont(HAVE_dI1b)) { Eval_dI1b(); } return dI1b;
   }
   MFEM_HOST_DEVICE const double *Get_dI2()
   {
      if (dont(HAVE_dI2)) { Eval_dI2(); } return dI2;
   }
   MFEM_HOST_DEVICE const double *Get_dI2b()
   {
      if (dont(HAVE_dI2b)) { Eval_dI2b(); } return dI2b;
   }
   MFEM_HOST_DEVICE const double *Get_dI3()
   {
      if (dont(HAVE_dI3)) { Eval_dI3(); } return dI3;
   }
   MFEM_HOST_DEVICE const double *Get_dI3b()
   {
      if (dont(HAVE_dI3b)) { Eval_dI3b(); } return dI3b;
   }

   // *****************************************************************************
   // ddI1b = X1 + X2 + X3, where
   // X1_ijkl = (2/3*I1b/I3) [ 2/3 dI3b_ij dI3b_kl + dI3b_kj dI3b_il ]
   // X2_ijkl = (I3b^{-2/3}) ddI1_ijkl
   // X3_ijkl = -(4/3*I3b^{-5/3}) (J_ij dI3b_kl + dI3b_ij J_kl)
   MFEM_HOST_DEVICE const double *Get_ddI1b_ij(int i, int j)
   {
      // X1_ijkl = (2/3*I1b/I3) [ 2/3 dI3b_ij dI3b_kl + dI3b_kj dI3b_il ]
      double X1_p[9], X2_p[9], X3_p[9];
      DeviceMatrix X1(X1_p,3,3);
      const double I3 = Get_I3();
      const double I1b = Get_I1b();
      const double alpha = (2./3.)*I1b/I3;

      ConstDeviceMatrix dI3b(Get_dI3b(),3,3);
      for (int k=0; k<3; k++)
      {
         for (int l=0; l<3; l++)
         {
            X1(k,l) = alpha * ((2./3.)*dI3b(i,j) * dI3b(k,l) +
                               dI3b(k,j)*dI3b(i,l));
         }
      }

      // ddI1_ijkl = 2 δ_ik δ_jl
      // X2_ijkl = (I3b^{-2/3}) ddI1_ijkl
      DeviceMatrix X2(X2_p,3,3);
      const double beta = Get_I3b_p();
      for (int k=0; k<3; k++)
      {
         for (int l=0; l<3; l++)
         {
            const double ddI1_ijkl = (i==k && j==l) ? 2.0 : 0.0;
            X2(k,l) = beta * ddI1_ijkl;
         }
      }

      // X3_ijkl = -(4/3*I3b^{-5/3}) (J_ij dI3b_kl + dI3b_ij J_kl)
      DeviceMatrix X3(X3_p,3,3);
      const double I3b = Get_I3b();
      const double gamma = -(4./3.)*Get_I3b_p()/I3b;
      ConstDeviceMatrix Jpt(J,3,3);
      for (int k=0; k<3; k++)
      {
         for (int l=0; l<3; l++)
         {
            X3(k,l) = gamma * (Jpt(i,j) * dI3b(k,l) + dI3b(i,j) * Jpt(k,l));
         }
      }

      DeviceMatrix ddI1b(ddI1b_ij,3,3);
      for (int k=0; k<3; k++)
      {
         for (int l=0; l<3; l++)
         {
            ddI1b(k,l) = X1(k,l) + X2(k,l) + X3(k,l);
         }
      }
      return ddI1b_ij;
   }

   // *****************************************************************************
   // ddI2 = x1 + x2 + x3
   //    x1_ijkl = (2 I1) δ_ik δ_jl
   //    x2_ijkl = 2 ( 2 δ_ku δ_iv - δ_ik δ_uv - δ_kv δ_iu ) J_vj J_ul
   //    x3_ijkl = -2 (J J^t)_ik δ_jl = -2 B_ik δ_jl
   MFEM_HOST_DEVICE const double *Get_ddI2_ij(int i, int j)
   {
      double x1_p[9], x2_p[9], x3_p[9];
      DeviceMatrix x1(x1_p,3,3), x2(x2_p,3,3), x3(x3_p,3,3);

      // x1_ijkl = (2 I1) δ_ik δ_jl
      const double I1 = Get_I1();
      for (int k=0; k<3; k++)
      {
         for (int l=0; l<3; l++)
         {
            const double ik_jl = (i==k && j==l) ? 1.0 : 0.0;
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
                  const double ku_iv = k==u && i==v ? 1.0 : 0.0;
                  const double ik_uv = i==k && u==v ? 1.0 : 0.0;
                  const double kv_iu = k==v && i==u ? 1.0 : 0.0;
                  x2(k,l) += 2.0 * (2.*ku_iv - ik_uv - kv_iu) * Jpt(v,j) * Jpt(u,l);
               }
            }
         }
      }

      //    x3_ijkl = -2 B_ik δ_jl
      double b[9];
      //const double *J = M;
      b[0] = J[0]*J[0] + J[3]*J[3] + J[6]*J[6];
      b[1] = J[1]*J[1] + J[4]*J[4] + J[7]*J[7];
      b[2] = J[2]*J[2] + J[5]*J[5] + J[8]*J[8];
      b[3] = J[0]*J[1] + J[3]*J[4] + J[6]*J[7]; // B(0,1)
      b[4] = J[0]*J[2] + J[3]*J[5] + J[6]*J[8]; // B(0,2)
      b[5] = J[1]*J[2] + J[4]*J[5] + J[7]*J[8]; // B(1,2)
      double b_p[9] =
      {
         b[0], b[3], b[4],
         b[3], b[1], b[5],
         b[4], b[5], b[2]
      };
      DeviceMatrix B(b_p,3,3);
      for (int k=0; k<3; k++)
      {
         for (int l=0; l<3; l++)
         {
            const double jl = j==l ? 1.0 : 0.0;
            x3(k,l) = -2.0 * B(i,k) * jl;
         }
      }

      // ddI2 = x1 + x2 + x3
      DeviceMatrix ddI2(ddI2_ij,3,3);
      for (int k=0; k<3; k++)
      {
         for (int l=0; l<3; l++)
         {
            ddI2(k,l) = x1(k,l) + x2(k,l) + x3(k,l);
         }
      }
      return ddI2_ij;
   }

   // *****************************************************************************
   // ddI2b = X1 + X2 + X3
   //    X1_ijkl = 16/9 det(J)^{-10/3} I2 dI3b_ij dI3b_kl +
   //               4/3 det(J)^{-10/3} I2 dI3b_il dI3b_kj
   //    X2_ijkl = -4/3 det(J)^{-7/3} (dI2_ij dI3b_kl + dI2_kl dI3b_ij)
   //    X3_ijkl =      det(J)^{-4/3} ddI2_ijkl
   MFEM_HOST_DEVICE const double *Get_ddI2b_ij(int i, int j)
   {
      double X1_p[9], X2_p[9], X3_p[9];
      // X1_ijkl = 16/9 det(J)^{-10/3} I2 dI3b_ij dI3b_kl +
      //               4/3 det(J)^{-10/3} I2 dI3b_il dI3b_kj
      DeviceMatrix X1(X1_p,3,3);
      const double I3b_p = Get_I3b_p(); // I3b^{-2/3}
      const double I3b = Get_I3b();     // det(J)
      const double I2 = Get_I2();
      const double I3b_p43 = I3b_p*I3b_p;
      const double I3b_p73 = I3b_p*I3b_p/I3b;
      const double I3b_p103 = I3b_p*I3b_p/(I3b*I3b);
      ConstDeviceMatrix dI3b(Get_dI3b(),3,3);
      for (int k=0; k<3; k++)
      {
         for (int l=0; l<3; l++)
         {
            const double up = (16./9.)*I3b_p103*I2*dI3b(i,j)*dI3b(k,l);
            const double down = (4./3.)*I3b_p103*I2*dI3b(i,l)*dI3b(k,j);
            X1(k,l) = up + down;
         }
      }

      // X2_ijkl = -4/3 det(J)^{-7/3} (dI2_ij dI3b_kl + dI2_kl dI3b_ij)
      DeviceMatrix X2(X2_p,3,3);
      ConstDeviceMatrix dI2(Get_dI2(),3,3);
      for (int k=0; k<3; k++)
      {
         for (int l=0; l<3; l++)
         {
            X2(k,l) = -(4./3.)*I3b_p73*(dI2(i,j)*dI3b(k,l) + dI2(k,l)*dI3b(i,j));
         }
      }

      ConstDeviceMatrix ddI2(Get_ddI2_ij(i,j),3,3);

      // X3_ijkl =  det(J)^{-4/3} ddI2_ijkl
      DeviceMatrix X3(X3_p,3,3);
      for (int k=0; k<3; k++)
      {
         for (int l=0; l<3; l++)
         {
            X3(k,l) = I3b_p43 * ddI2(k,l);
         }
      }

      // ddI2b = X1 + X2 + X3
      DeviceMatrix ddI2b(ddI2b_ij,3,3);
      for (int k=0; k<3; k++)
      {
         for (int l=0; l<3; l++)
         {
            ddI2b(k,l) = X1(k,l) + X2(k,l) + X3(k,l);
         }
      }
      return ddI2b_ij;
   }
};

template<int H, int W, typename T>
MFEM_HOST_DEVICE inline
void FNorm(double &scale_factor, double &scaled_fnorm2, const T *data)
{
   int i, hw = H * W;
   T max_norm = 0.0, entry, fnorm2;

   for (i = 0; i < hw; i++)
   {
      entry = fabs(data[i]);
      if (entry > max_norm)
      {
         max_norm = entry;
      }
   }

   if (max_norm == 0.0)
   {
      scale_factor = scaled_fnorm2 = 0.0;
      return;
   }

   fnorm2 = 0.0;
   for (i = 0; i < hw; i++)
   {
      entry = data[i] / max_norm;
      fnorm2 += entry * entry;
   }

   scale_factor = max_norm;
   scaled_fnorm2 = fnorm2;
}

/// Compute the Frobenius norm of the matrix
template<int H, int W, typename T>
MFEM_HOST_DEVICE inline
double FNorm(const T *data)
{
   double s, n2;
   kernels::FNorm<H,W>(s, n2, data);
   return s*sqrt(n2);
}

/// Compute the square of the Frobenius norm of the matrix
template<int H, int W, typename T>
MFEM_HOST_DEVICE inline
double FNorm2(const T *data)
{
   double s, n2;
   kernels::FNorm<H,W>(s, n2, data);
   return s*s*n2;
}

/// Returns the l2 norm of the Vector with given @a size and @a data.
template<typename T>
MFEM_HOST_DEVICE inline
double Norml2(const int size, const T *data)
{
   if (0 == size) { return 0.0; }
   if (1 == size) { return std::abs(data[0]); }
   T scale = 0.0;
   T sum = 0.0;
   for (int i = 0; i < size; i++)
   {
      if (data[i] != 0.0)
      {
         const T absdata = fabs(data[i]);
         if (scale <= absdata)
         {
            const T sqr_arg = scale / absdata;
            sum = 1.0 + sum * (sqr_arg * sqr_arg);
            scale = absdata;
            continue;
         } // end if scale <= absdata
         const T sqr_arg = absdata / scale;
         sum += (sqr_arg * sqr_arg); // else scale > absdata
      } // end if data[i] != 0
   }
   return scale * sqrt(sum);
}

/** @brief Matrix vector multiplication: y = A x, where the matrix A is of size
    @a height x @a width with given @a data, while @a x and @a y specify the
    data of the input and output vectors. */
template<typename TA, typename TX, typename TY>
MFEM_HOST_DEVICE inline
void Mult(const int height, const int width, TA *data, const TX *x, TY *y)
{
   if (width == 0)
   {
      for (int row = 0; row < height; row++)
      {
         y[row] = 0.0;
      }
      return;
   }
   TA *d_col = data;
   TX x_col = x[0];
   for (int row = 0; row < height; row++)
   {
      y[row] = x_col*d_col[row];
   }
   d_col += height;
   for (int col = 1; col < width; col++)
   {
      x_col = x[col];
      for (int row = 0; row < height; row++)
      {
         y[row] += x_col*d_col[row];
      }
      d_col += height;
   }
}

/// Symmetrize a square matrix with given @a size and @a data: A -> (A+A^T)/2.
template<typename T>
MFEM_HOST_DEVICE inline
void Symmetrize(const int size, T *data)
{
   for (int i = 0; i < size; i++)
   {
      for (int j = 0; j < i; j++)
      {
         const T a = 0.5 * (data[i*size+j] + data[j*size+i]);
         data[j*size+i] = data[i*size+j] = a;
      }
   }
}

/// Compute the determinant of a square matrix of size dim with given @a data.
template<int dim, typename T>
MFEM_HOST_DEVICE inline T Det(const T *data)
{
   return TDetHD<T>(ColumnMajorLayout2D<dim,dim>(), data);
}

/** @brief Return the inverse of a matrix with given @a size and @a data into
   the matrix with data @a inv_data. */
template<int dim, typename T>
MFEM_HOST_DEVICE inline
void CalcInverse(const T *data, T *inv_data)
{
   typedef ColumnMajorLayout2D<dim,dim> layout_t;
   const T det = TAdjDetHD<T>(layout_t(), data, layout_t(), inv_data);
   TAssignHD<AssignOp::Mult>(layout_t(), inv_data, static_cast<T>(1.0)/det);
}

/** @brief Return the adjugate of a matrix */
template<int dim, typename T>
MFEM_HOST_DEVICE inline
void CalcAdjugate(const T *data, T *adj_data)
{
   typedef ColumnMajorLayout2D<dim,dim> layout_t;
   TAdjugateHD<T>(layout_t(), data, layout_t(), adj_data);
}

/** @brief Compute C = A + alpha*B, where the matrices A, B and C are of size @a
    height x @a width with data @a Adata, @a Bdata and @a Cdata. */
template<typename TALPHA, typename TA, typename TB, typename TC>
MFEM_HOST_DEVICE inline
void Add(const int height, const int width, const TALPHA alpha,
         const TA *Adata, const TB *Bdata, TC *Cdata)
{
   for (int j = 0; j < width; j++)
   {
      for (int i = 0; i < height; i++)
      {
         const int n = i*width+j;
         Cdata[n] = Adata[n] + alpha * Bdata[n];
      }
   }
}

/** @brief Compute C = alpha*A + beta*B, where the matrices A, B and C are of
    size @a height x @a width with data @a Adata, @a Bdata and @a Cdata. */
template<typename TALPHA, typename TBETA, typename TA, typename TB, typename TC>
MFEM_HOST_DEVICE inline
void Add(const int height, const int width,
         const TALPHA alpha, const TA *Adata,
         const TBETA beta, const TB *Bdata,
         TC *Cdata)
{
   const int m = height * width;
   for (int i = 0; i < m; i++)
   {
      Cdata[i] = alpha * Adata[i] + beta * Bdata[i];
   }
}

/** @brief Compute B += A, where the matrices A and B are of size
    @a height x @a width with data @a Adata and @a Bdata. */
template<typename TA, typename TB>
MFEM_HOST_DEVICE inline
void Add(const int height, const int width, const TA *Adata, TB *Bdata)
{
   const int m = height * width;
   for (int i = 0; i < m; i++)
   {
      Bdata[i] += Adata[i];
   }
}


/** @brief Matrix-matrix multiplication: A = B * C, where the matrices A, B and
    C are of sizes @a Aheight x @a Awidth, @a Aheight x @a Bwidth and @a Bwidth
    x @a Awidth, respectively. */
template<typename TA, typename TB, typename TC>
MFEM_HOST_DEVICE inline
void Mult(const int Aheight, const int Awidth, const int Bwidth,
          const TB *Bdata, const TC *Cdata, TA *Adata)
{
   const int ah_x_aw = Aheight * Awidth;
   for (int i = 0; i < ah_x_aw; i++) { Adata[i] = 0.0; }
   for (int j = 0; j < Awidth; j++)
   {
      for (int k = 0; k < Bwidth; k++)
      {
         for (int i = 0; i < Aheight; i++)
         {
            Adata[i+j*Aheight] += Bdata[i+k*Aheight] * Cdata[k+j*Bwidth];
         }
      }
   }
}

/** @brief Multiply a matrix of size @a Aheight x @a Awidth and data @a Adata
    with the transpose of a matrix of size @a Bheight x @a Awidth and data @a
    Bdata: A * Bt. Return the result in a matrix with data @a ABtdata. */
template<typename TA, typename TB, typename TC>
MFEM_HOST_DEVICE inline
void MultABt(const int Aheight, const int Awidth, const int Bheight,
             const TA *Adata, const TB *Bdata, TC *ABtdata)
{
   const int ah_x_bh = Aheight * Bheight;
   for (int i = 0; i < ah_x_bh; i++) { ABtdata[i] = 0.0; }
   for (int k = 0; k < Awidth; k++)
   {
      TC *c = ABtdata;
      for (int j = 0; j < Bheight; j++)
      {
         const double bjk = Bdata[j];
         for (int i = 0; i < Aheight; i++)
         {
            c[i] += Adata[i] * bjk;
         }
         c += Aheight;
      }
      Adata += Aheight;
      Bdata += Bheight;
   }
}

/// Compute the spectrum of the matrix of size dim with given @a data, returning
/// the eigenvalues in the array @a lambda and the eigenvectors in the array @a
/// vec (listed consecutively).
template<int dim> MFEM_HOST_DEVICE
void CalcEigenvalues(const double *data, double *lambda, double *vec);

/// Return the i'th singular value of the matrix of size dim with given @a data.
template<int dim> MFEM_HOST_DEVICE
double CalcSingularvalue(const double *data, const int i);


// Utility functions for CalcEigenvalues and CalcSingularvalue
namespace internal
{

/// Utility function to swap the values of @a a and @a b.
template<typename T>
MFEM_HOST_DEVICE static inline
void Swap(T &a, T &b)
{
   T tmp = a;
   a = b;
   b = tmp;
}

const double Epsilon = std::numeric_limits<double>::epsilon();

/// Utility function used in CalcSingularvalue<3>.
MFEM_HOST_DEVICE static inline
void Eigenvalues2S(const double &d12, double &d1, double &d2)
{
   const double sqrt_1_eps = sqrt(1./Epsilon);
   if (d12 != 0.)
   {
      // "The Symmetric Eigenvalue Problem", B. N. Parlett, pp.189-190
      double t;
      const double zeta = (d2 - d1)/(2*d12); // inf/inf from overflows?
      if (fabs(zeta) < sqrt_1_eps)
      {
         t = d12*copysign(1./(fabs(zeta) + sqrt(1. + zeta*zeta)), zeta);
      }
      else
      {
         t = d12*copysign(0.5/fabs(zeta), zeta);
      }
      d1 -= t;
      d2 += t;
   }
}

/// Utility function used in CalcEigenvalues().
MFEM_HOST_DEVICE static inline
void Eigensystem2S(const double &d12, double &d1, double &d2,
                   double &c, double &s)
{
   const double sqrt_1_eps = sqrt(1./Epsilon);
   if (d12 == 0.0)
   {
      c = 1.;
      s = 0.;
   }
   else
   {
      // "The Symmetric Eigenvalue Problem", B. N. Parlett, pp.189-190
      double t;
      const double zeta = (d2 - d1)/(2*d12);
      const double azeta = fabs(zeta);
      if (azeta < sqrt_1_eps)
      {
         t = copysign(1./(azeta + sqrt(1. + zeta*zeta)), zeta);
      }
      else
      {
         t = copysign(0.5/azeta, zeta);
      }
      c = sqrt(1./(1. + t*t));
      s = c*t;
      t *= d12;
      d1 -= t;
      d2 += t;
   }
}


/// Utility function used in CalcEigenvalues<3>.
MFEM_HOST_DEVICE static inline
void GetScalingFactor(const double &d_max, double &mult)
{
   int d_exp;
   if (d_max > 0.)
   {
      mult = frexp(d_max, &d_exp);
      if (d_exp == std::numeric_limits<double>::max_exponent)
      {
         mult *= std::numeric_limits<double>::radix;
      }
      mult = d_max/mult;
   }
   else
   {
      mult = 1.;
   }
   // mult = 2^d_exp is such that d_max/mult is in [0.5,1) or in other words
   // d_max is in the interval [0.5,1)*mult
}

/// Utility function used in CalcEigenvalues<3>.
MFEM_HOST_DEVICE static inline
bool KernelVector2G(const int &mode,
                    double &d1, double &d12, double &d21, double &d2)
{
   // Find a vector (z1,z2) in the "near"-kernel of the matrix
   // |  d1  d12 |
   // | d21   d2 |
   // using QR factorization.
   // The vector (z1,z2) is returned in (d1,d2). Return 'true' if the matrix
   // is zero without setting (d1,d2).
   // Note: in the current implementation |z1| + |z2| = 1.

   // l1-norms of the columns
   double n1 = fabs(d1) + fabs(d21);
   double n2 = fabs(d2) + fabs(d12);

   bool swap_columns = (n2 > n1);
   double mu;

   if (!swap_columns)
   {
      if (n1 == 0.)
      {
         return true;
      }

      if (mode == 0) // eliminate the larger entry in the column
      {
         if (fabs(d1) > fabs(d21))
         {
            Swap(d1, d21);
            Swap(d12, d2);
         }
      }
      else // eliminate the smaller entry in the column
      {
         if (fabs(d1) < fabs(d21))
         {
            Swap(d1, d21);
            Swap(d12, d2);
         }
      }
   }
   else
   {
      // n2 > n1, swap columns 1 and 2
      if (mode == 0) // eliminate the larger entry in the column
      {
         if (fabs(d12) > fabs(d2))
         {
            Swap(d1, d2);
            Swap(d12, d21);
         }
         else
         {
            Swap(d1, d12);
            Swap(d21, d2);
         }
      }
      else // eliminate the smaller entry in the column
      {
         if (fabs(d12) < fabs(d2))
         {
            Swap(d1, d2);
            Swap(d12, d21);
         }
         else
         {
            Swap(d1, d12);
            Swap(d21, d2);
         }
      }
   }

   n1 = hypot(d1, d21);

   if (d21 != 0.)
   {
      // v = (n1, n2)^t,  |v| = 1
      // Q = I - 2 v v^t,  Q (d1, d21)^t = (mu, 0)^t
      mu = copysign(n1, d1);
      n1 = -d21*(d21/(d1 + mu)); // = d1 - mu
      d1 = mu;
      // normalize (n1,d21) to avoid overflow/underflow
      // normalize (n1,d21) by the max-norm to avoid the sqrt call
      if (fabs(n1) <= fabs(d21))
      {
         // (n1,n2) <-- (n1/d21,1)
         n1 = n1/d21;
         mu = (2./(1. + n1*n1))*(n1*d12 + d2);
         d2  = d2  - mu;
         d12 = d12 - mu*n1;
      }
      else
      {
         // (n1,n2) <-- (1,d21/n1)
         n2 = d21/n1;
         mu = (2./(1. + n2*n2))*(d12 + n2*d2);
         d2  = d2  - mu*n2;
         d12 = d12 - mu;
      }
   }

   // Solve:
   // | d1 d12 | | z1 | = | 0 |
   // |  0  d2 | | z2 |   | 0 |

   // choose (z1,z2) to minimize |d1*z1 + d12*z2| + |d2*z2|
   // under the condition |z1| + |z2| = 1, z2 >= 0 (for uniqueness)
   // set t = z1, z2 = 1 - |t|, -1 <= t <= 1
   // objective function is:
   // |d1*t + d12*(1 - |t|)| + |d2|*(1 - |t|) -- piecewise linear with
   // possible minima are -1,0,1,t1 where t1: d1*t1 + d12*(1 - |t1|) = 0
   // values: @t=+/-1 -> |d1|, @t=0 -> |n1| + |d2|, @t=t1 -> |d2|*(1 - |t1|)

   // evaluate z2 @t=t1
   mu = -d12/d1;
   // note: |mu| <= 1,       if using l2-norm for column pivoting
   //       |mu| <= sqrt(2), if using l1-norm
   n2 = 1./(1. + fabs(mu));
   // check if |d1|<=|d2|*z2
   if (fabs(d1) <= n2*fabs(d2))
   {
      d2 = 0.;
      d1 = 1.;
   }
   else
   {
      d2 = n2;
      // d1 = (n2 < 0.5) ? copysign(1. - n2, mu) : mu*n2;
      d1 = mu*n2;
   }

   if (swap_columns)
   {
      Swap(d1, d2);
   }

   return false;
}

/// Utility function used in CalcEigenvalues<3>.
MFEM_HOST_DEVICE static inline
void Vec_normalize3_aux(const double &x1, const double &x2,
                        const double &x3,
                        double &n1, double &n2, double &n3)
{
   double t, r;

   const double m = fabs(x1);
   r = x2/m;
   t = 1. + r*r;
   r = x3/m;
   t = sqrt(1./(t + r*r));
   n1 = copysign(t, x1);
   t /= m;
   n2 = x2*t;
   n3 = x3*t;
}

/// Utility function used in CalcEigenvalues<3>.
MFEM_HOST_DEVICE static inline
void Vec_normalize3(const double &x1, const double &x2, const double &x3,
                    double &n1, double &n2, double &n3)
{
   // should work ok when xk is the same as nk for some or all k
   if (fabs(x1) >= fabs(x2))
   {
      if (fabs(x1) >= fabs(x3))
      {
         if (x1 != 0.)
         {
            Vec_normalize3_aux(x1, x2, x3, n1, n2, n3);
         }
         else
         {
            n1 = n2 = n3 = 0.;
         }
         return;
      }
   }
   else if (fabs(x2) >= fabs(x3))
   {
      Vec_normalize3_aux(x2, x1, x3, n2, n1, n3);
      return;
   }
   Vec_normalize3_aux(x3, x1, x2, n3, n1, n2);
}

/// Utility function used in CalcEigenvalues<3>.
MFEM_HOST_DEVICE static inline
int KernelVector3G_aux(const int &mode,
                       double &d1, double &d2, double &d3,
                       double &c12, double &c13, double &c23,
                       double &c21, double &c31, double &c32)
{
   int kdim;
   double mu, n1, n2, n3, s1, s2, s3;

   s1 = hypot(c21, c31);
   n1 = hypot(d1, s1);

   if (s1 != 0.)
   {
      // v = (s1, s2, s3)^t,  |v| = 1
      // Q = I - 2 v v^t,  Q (d1, c12, c13)^t = (mu, 0, 0)^t
      mu = copysign(n1, d1);
      n1 = -s1*(s1/(d1 + mu)); // = d1 - mu
      d1 = mu;

      // normalize (n1,c21,c31) to avoid overflow/underflow
      // normalize (n1,c21,c31) by the max-norm to avoid the sqrt call
      if (fabs(n1) >= fabs(c21))
      {
         if (fabs(n1) >= fabs(c31))
         {
            // n1 is max, (s1,s2,s3) <-- (1,c21/n1,c31/n1)
            s2 = c21/n1;
            s3 = c31/n1;
            mu = 2./(1. + s2*s2 + s3*s3);
            n2  = mu*(c12 + s2*d2  + s3*c32);
            n3  = mu*(c13 + s2*c23 + s3*d3);
            c12 = c12 -    n2;
            d2  = d2  - s2*n2;
            c32 = c32 - s3*n2;
            c13 = c13 -    n3;
            c23 = c23 - s2*n3;
            d3  = d3  - s3*n3;
            goto done_column_1;
         }
      }
      else if (fabs(c21) >= fabs(c31))
      {
         // c21 is max, (s1,s2,s3) <-- (n1/c21,1,c31/c21)
         s1 = n1/c21;
         s3 = c31/c21;
         mu = 2./(1. + s1*s1 + s3*s3);
         n2  = mu*(s1*c12 + d2  + s3*c32);
         n3  = mu*(s1*c13 + c23 + s3*d3);
         c12 = c12 - s1*n2;
         d2  = d2  -    n2;
         c32 = c32 - s3*n2;
         c13 = c13 - s1*n3;
         c23 = c23 -    n3;
         d3  = d3  - s3*n3;
         goto done_column_1;
      }
      // c31 is max, (s1,s2,s3) <-- (n1/c31,c21/c31,1)
      s1 = n1/c31;
      s2 = c21/c31;
      mu = 2./(1. + s1*s1 + s2*s2);
      n2  = mu*(s1*c12 + s2*d2  + c32);
      n3  = mu*(s1*c13 + s2*c23 + d3);
      c12 = c12 - s1*n2;
      d2  = d2  - s2*n2;
      c32 = c32 -    n2;
      c13 = c13 - s1*n3;
      c23 = c23 - s2*n3;
      d3  = d3  -    n3;
   }

done_column_1:

   // Solve:
   // |  d2 c23 | | z2 | = | 0 |
   // | c32  d3 | | z3 |   | 0 |
   if (KernelVector2G(mode, d2, c23, c32, d3))
   {
      // Have two solutions:
      // two vectors in the kernel are P (-c12/d1, 1, 0)^t and
      // P (-c13/d1, 0, 1)^t where P is the permutation matrix swapping
      // entries 1 and col.

      // A vector orthogonal to both these vectors is P (1, c12/d1, c13/d1)^t
      d2 = c12/d1;
      d3 = c13/d1;
      d1 = 1.;
      kdim = 2;
   }
   else
   {
      // solve for z1:
      // note: |z1| <= a since |z2| + |z3| = 1, and
      // max{|c12|,|c13|} <= max{norm(col. 2),norm(col. 3)}
      //                  <= norm(col. 1) <= a |d1|
      // a = 1,       if using l2-norm for column pivoting
      // a = sqrt(3), if using l1-norm
      d1 = -(c12*d2 + c13*d3)/d1;
      kdim = 1;
   }

   Vec_normalize3(d1, d2, d3, d1, d2, d3);

   return kdim;
}

/// Utility function used in CalcEigenvalues<3>.
MFEM_HOST_DEVICE static inline
int KernelVector3S(const int &mode, const double &d12,
                   const double &d13, const double &d23,
                   double &d1, double &d2, double &d3)
{
   // Find a unit vector (z1,z2,z3) in the "near"-kernel of the matrix
   // |  d1  d12  d13 |
   // | d12   d2  d23 |
   // | d13  d23   d3 |
   // using QR factorization.
   // The vector (z1,z2,z3) is returned in (d1,d2,d3).
   // Returns the dimension of the kernel, kdim, but never zero.
   // - if kdim == 3, then (d1,d2,d3) is not defined,
   // - if kdim == 2, then (d1,d2,d3) is a vector orthogonal to the kernel,
   // - otherwise kdim == 1 and (d1,d2,d3) is a vector in the "near"-kernel.

   double c12 = d12, c13 = d13, c23 = d23;
   double c21, c31, c32;
   int col, row;

   // l1-norms of the columns:
   c32 = fabs(d1) + fabs(c12) + fabs(c13);
   c31 = fabs(d2) + fabs(c12) + fabs(c23);
   c21 = fabs(d3) + fabs(c13) + fabs(c23);

   // column pivoting: choose the column with the largest norm
   if (c32 >= c21)
   {
      col = (c32 >= c31) ? 1 : 2;
   }
   else
   {
      col = (c31 >= c21) ? 2 : 3;
   }
   switch (col)
   {
      case 1:
         if (c32 == 0.) // zero matrix
         {
            return 3;
         }
         break;

      case 2:
         if (c31 == 0.) // zero matrix
         {
            return 3;
         }
         Swap(c13, c23);
         Swap(d1, d2);
         break;

      case 3:
         if (c21 == 0.) // zero matrix
         {
            return 3;
         }
         Swap(c12, c23);
         Swap(d1, d3);
   }

   // row pivoting depending on 'mode'
   if (mode == 0)
   {
      if (fabs(d1) <= fabs(c13))
      {
         row = (fabs(d1) <= fabs(c12)) ? 1 : 2;
      }
      else
      {
         row = (fabs(c12) <= fabs(c13)) ? 2 : 3;
      }
   }
   else
   {
      if (fabs(d1) >= fabs(c13))
      {
         row = (fabs(d1) >= fabs(c12)) ? 1 : 2;
      }
      else
      {
         row = (fabs(c12) >= fabs(c13)) ? 2 : 3;
      }
   }
   switch (row)
   {
      case 1:
         c21 = c12;
         c31 = c13;
         c32 = c23;
         break;

      case 2:
         c21 = d1;
         c31 = c13;
         c32 = c23;
         d1 = c12;
         c12 = d2;
         d2 = d1;
         c13 = c23;
         c23 = c31;
         break;

      case 3:
         c21 = c12;
         c31 = d1;
         c32 = c12;
         d1 = c13;
         c12 = c23;
         c13 = d3;
         d3 = d1;
   }
   row = KernelVector3G_aux(mode, d1, d2, d3, c12, c13, c23, c21, c31, c32);
   // row is kdim

   switch (col)
   {
      case 2:
         Swap(d1, d2);
         break;

      case 3:
         Swap(d1, d3);
   }
   return row;
}

/// Utility function used in CalcEigenvalues<3>.
MFEM_HOST_DEVICE static inline
int Reduce3S(const int &mode,
             double &d1, double &d2, double &d3,
             double &d12, double &d13, double &d23,
             double &z1, double &z2, double &z3,
             double &v1, double &v2, double &v3,
             double &g)
{
   // Given the matrix
   //     |  d1  d12  d13 |
   // A = | d12   d2  d23 |
   //     | d13  d23   d3 |
   // and a unit eigenvector z=(z1,z2,z3), transform the matrix A into the
   // matrix B = Q P A P Q that has the form
   //                 | b1   0   0 |
   // B = Q P A P Q = | 0   b2 b23 |
   //                 | 0  b23  b3 |
   // where P is the permutation matrix switching entries 1 and k, and
   // Q is the reflection matrix Q = I - g v v^t, defined by: set y = P z and
   // v = c(y - e_1); if y = e_1, then v = 0 and Q = I.
   // Note: Q y = e_1, Q e_1 = y ==> Q P A P Q e_1 = ... = lambda e_1.
   // The entries (b1,b2,b3,b23) are returned in (d1,d2,d3,d23), and the
   // return value of the function is k. The variable g = 2/(v1^2+v2^2+v3^3).

   int k;
   double s, w1, w2, w3;

   if (mode == 0)
   {
      // choose k such that z^t e_k = zk has the smallest absolute value, i.e.
      // the angle between z and e_k is closest to pi/2
      if (fabs(z1) <= fabs(z3))
      {
         k = (fabs(z1) <= fabs(z2)) ? 1 : 2;
      }
      else
      {
         k = (fabs(z2) <= fabs(z3)) ? 2 : 3;
      }
   }
   else
   {
      // choose k such that zk is the largest by absolute value
      if (fabs(z1) >= fabs(z3))
      {
         k = (fabs(z1) >= fabs(z2)) ? 1 : 2;
      }
      else
      {
         k = (fabs(z2) >= fabs(z3)) ? 2 : 3;
      }
   }
   switch (k)
   {
      case 2:
         Swap(d13, d23);
         Swap(d1, d2);
         Swap(z1, z2);
         break;

      case 3:
         Swap(d12, d23);
         Swap(d1, d3);
         Swap(z1, z3);
   }

   s = hypot(z2, z3);

   if (s == 0.)
   {
      // s can not be zero, if zk is the smallest (mode == 0)
      v1 = v2 = v3 = 0.;
      g = 1.;
   }
   else
   {
      g = copysign(1., z1);
      v1 = -s*(s/(z1 + g)); // = z1 - g
      // normalize (v1,z2,z3) by its max-norm, avoiding the sqrt call
      g = fabs(v1);
      if (fabs(z2) > g) { g = fabs(z2); }
      if (fabs(z3) > g) { g = fabs(z3); }
      v1 = v1/g;
      v2 = z2/g;
      v3 = z3/g;
      g = 2./(v1*v1 + v2*v2 + v3*v3);

      // Compute Q A Q = A - v w^t - w v^t, where
      // w = u - (g/2)(v^t u) v, and u = g A v
      // set w = g A v
      w1 = g*( d1*v1 + d12*v2 + d13*v3);
      w2 = g*(d12*v1 +  d2*v2 + d23*v3);
      w3 = g*(d13*v1 + d23*v2 +  d3*v3);
      // w := w - (g/2)(v^t w) v
      s = (g/2)*(v1*w1 + v2*w2 + v3*w3);
      w1 -= s*v1;
      w2 -= s*v2;
      w3 -= s*v3;
      // dij -= vi*wj + wi*vj
      d1  -= 2*v1*w1;
      d2  -= 2*v2*w2;
      d23 -= v2*w3 + v3*w2;
      d3  -= 2*v3*w3;
      // compute the offdiagonal entries on the first row/column of B which
      // should be zero (for debugging):
#if 0
      s = d12 - v1*w2 - v2*w1;  // b12 = 0
      s = d13 - v1*w3 - v3*w1;  // b13 = 0
#endif
   }

   switch (k)
   {
      case 2:
         Swap(z1, z2);
         break;
      case 3:
         Swap(z1, z3);
   }
   return k;
}

} // namespace kernels::internal


// Implementations of CalcEigenvalues and CalcSingularvalue for dim = 2, 3.

/// Compute the spectrum of the matrix of size 2 with given @a data, returning
/// the eigenvalues in the array @a lambda and the eigenvectors in the array @a
/// vec (listed consecutively).
template<> MFEM_HOST_DEVICE inline
void CalcEigenvalues<2>(const double *data, double *lambda, double *vec)
{
   double d0 = data[0];
   double d2 = data[2]; // use the upper triangular entry
   double d3 = data[3];
   double c, s;
   internal::Eigensystem2S(d2, d0, d3, c, s);
   if (d0 <= d3)
   {
      lambda[0] = d0;
      lambda[1] = d3;
      vec[0] =  c;
      vec[1] = -s;
      vec[2] =  s;
      vec[3] =  c;
   }
   else
   {
      lambda[0] = d3;
      lambda[1] = d0;
      vec[0] =  s;
      vec[1] =  c;
      vec[2] =  c;
      vec[3] = -s;
   }
}

/// Compute the spectrum of the matrix of size 3 with given @a data, returning
/// the eigenvalues in the array @a lambda and the eigenvectors in the array @a
/// vec (listed consecutively).
template<> MFEM_HOST_DEVICE inline
void CalcEigenvalues<3>(const double *data, double *lambda, double *vec)
{
   double d11 = data[0];
   double d12 = data[3]; // use the upper triangular entries
   double d22 = data[4];
   double d13 = data[6];
   double d23 = data[7];
   double d33 = data[8];

   double mult;
   {
      double d_max = fabs(d11);
      if (d_max < fabs(d22)) { d_max = fabs(d22); }
      if (d_max < fabs(d33)) { d_max = fabs(d33); }
      if (d_max < fabs(d12)) { d_max = fabs(d12); }
      if (d_max < fabs(d13)) { d_max = fabs(d13); }
      if (d_max < fabs(d23)) { d_max = fabs(d23); }

      internal::GetScalingFactor(d_max, mult);
   }

   d11 /= mult;  d22 /= mult;  d33 /= mult;
   d12 /= mult;  d13 /= mult;  d23 /= mult;

   double aa = (d11 + d22 + d33)/3;  // aa = tr(A)/3
   double c1 = d11 - aa;
   double c2 = d22 - aa;
   double c3 = d33 - aa;

   double Q, R;

   Q = (2*(d12*d12 + d13*d13 + d23*d23) + c1*c1 + c2*c2 + c3*c3)/6;
   R = (c1*(d23*d23 - c2*c3)+ d12*(d12*c3 - 2*d13*d23) + d13*d13*c2)/2;

   if (Q <= 0.)
   {
      lambda[0] = lambda[1] = lambda[2] = aa;
      vec[0] = 1.; vec[3] = 0.; vec[6] = 0.;
      vec[1] = 0.; vec[4] = 1.; vec[7] = 0.;
      vec[2] = 0.; vec[5] = 0.; vec[8] = 1.;
   }
   else
   {
      double sqrtQ = sqrt(Q);
      double sqrtQ3 = Q*sqrtQ;
      // double sqrtQ3 = sqrtQ*sqrtQ*sqrtQ;
      // double sqrtQ3 = pow(Q, 1.5);
      double r;
      if (fabs(R) >= sqrtQ3)
      {
         if (R < 0.)
         {
            // R = -1.;
            r = 2*sqrtQ;
         }
         else
         {
            // R = 1.;
            r = -2*sqrtQ;
         }
      }
      else
      {
         R = R/sqrtQ3;

         if (R < 0.)
         {
            r = -2*sqrtQ*cos((acos(R) + 2.0*M_PI)/3); // max
         }
         else
         {
            r = -2*sqrtQ*cos(acos(R)/3); // min
         }
      }

      aa += r;
      c1 = d11 - aa;
      c2 = d22 - aa;
      c3 = d33 - aa;

      // Type of Householder reflections: z --> mu ek, where k is the index
      // of the entry in z with:
      // mode == 0: smallest absolute value --> angle closest to pi/2
      // mode == 1: largest absolute value --> angle farthest from pi/2
      // Observations:
      // mode == 0 produces better eigenvectors, less accurate eigenvalues?
      // mode == 1 produces better eigenvalues, less accurate eigenvectors?
      const int mode = 0;

      // Find a unit vector z = (z1,z2,z3) in the "near"-kernel of
      //  |  c1  d12  d13 |
      //  | d12   c2  d23 | = A - aa*I
      //  | d13  d23   c3 |
      // This vector is also an eigenvector for A corresponding to aa.
      // The vector z overwrites (c1,c2,c3).
      switch (internal::KernelVector3S(mode, d12, d13, d23, c1, c2, c3))
      {
         case 3:
            // 'aa' is a triple eigenvalue
            lambda[0] = lambda[1] = lambda[2] = aa;
            vec[0] = 1.; vec[3] = 0.; vec[6] = 0.;
            vec[1] = 0.; vec[4] = 1.; vec[7] = 0.;
            vec[2] = 0.; vec[5] = 0.; vec[8] = 1.;
            goto done_3d;

         case 2:
         // ok, continue with the returned vector orthogonal to the kernel
         case 1:
            // ok, continue with the returned vector in the "near"-kernel
            ;
      }

      // Using the eigenvector c=(c1,c2,c3) transform A into
      //                   | d11   0   0 |
      // A <-- Q P A P Q = |  0  d22 d23 |
      //                   |  0  d23 d33 |
      double v1, v2, v3, g;
      int k = internal::Reduce3S(mode, d11, d22, d33, d12, d13, d23,
                                 c1, c2, c3, v1, v2, v3, g);
      // Q = I - 2 v v^t
      // P - permutation matrix switching entries 1 and k

      // find the eigenvalues and eigenvectors for
      // | d22 d23 |
      // | d23 d33 |
      double c, s;
      internal::Eigensystem2S(d23, d22, d33, c, s);
      // d22 <-> P Q (0, c, -s), d33 <-> P Q (0, s, c)

      double *vec_1, *vec_2, *vec_3;
      if (d11 <= d22)
      {
         if (d22 <= d33)
         {
            lambda[0] = d11;  vec_1 = vec;
            lambda[1] = d22;  vec_2 = vec + 3;
            lambda[2] = d33;  vec_3 = vec + 6;
         }
         else if (d11 <= d33)
         {
            lambda[0] = d11;  vec_1 = vec;
            lambda[1] = d33;  vec_3 = vec + 3;
            lambda[2] = d22;  vec_2 = vec + 6;
         }
         else
         {
            lambda[0] = d33;  vec_3 = vec;
            lambda[1] = d11;  vec_1 = vec + 3;
            lambda[2] = d22;  vec_2 = vec + 6;
         }
      }
      else
      {
         if (d11 <= d33)
         {
            lambda[0] = d22;  vec_2 = vec;
            lambda[1] = d11;  vec_1 = vec + 3;
            lambda[2] = d33;  vec_3 = vec + 6;
         }
         else if (d22 <= d33)
         {
            lambda[0] = d22;  vec_2 = vec;
            lambda[1] = d33;  vec_3 = vec + 3;
            lambda[2] = d11;  vec_1 = vec + 6;
         }
         else
         {
            lambda[0] = d33;  vec_3 = vec;
            lambda[1] = d22;  vec_2 = vec + 3;
            lambda[2] = d11;  vec_1 = vec + 6;
         }
      }

      vec_1[0] = c1;
      vec_1[1] = c2;
      vec_1[2] = c3;
      d22 = g*(v2*c - v3*s);
      d33 = g*(v2*s + v3*c);
      vec_2[0] =    - v1*d22;  vec_3[0] =   - v1*d33;
      vec_2[1] =  c - v2*d22;  vec_3[1] = s - v2*d33;
      vec_2[2] = -s - v3*d22;  vec_3[2] = c - v3*d33;
      switch (k)
      {
         case 2:
            internal::Swap(vec_2[0], vec_2[1]);
            internal::Swap(vec_3[0], vec_3[1]);
            break;

         case 3:
            internal::Swap(vec_2[0], vec_2[2]);
            internal::Swap(vec_3[0], vec_3[2]);
      }
   }

done_3d:
   lambda[0] *= mult;
   lambda[1] *= mult;
   lambda[2] *= mult;
}

/// Return the i'th singular value of the matrix of size 2 with given @a data.
template<> MFEM_HOST_DEVICE inline
double CalcSingularvalue<2>(const double *data, const int i)
{
   double d0, d1, d2, d3;
   d0 = data[0];
   d1 = data[1];
   d2 = data[2];
   d3 = data[3];
   double mult;

   {
      double d_max = fabs(d0);
      if (d_max < fabs(d1)) { d_max = fabs(d1); }
      if (d_max < fabs(d2)) { d_max = fabs(d2); }
      if (d_max < fabs(d3)) { d_max = fabs(d3); }
      internal::GetScalingFactor(d_max, mult);
   }

   d0 /= mult;
   d1 /= mult;
   d2 /= mult;
   d3 /= mult;

   double t = 0.5*((d0+d2)*(d0-d2)+(d1-d3)*(d1+d3));
   double s = d0*d2 + d1*d3;
   s = sqrt(0.5*(d0*d0 + d1*d1 + d2*d2 + d3*d3) + sqrt(t*t + s*s));

   if (s == 0.0)
   {
      return 0.0;
   }
   t = fabs(d0*d3 - d1*d2) / s;
   if (t > s)
   {
      if (i == 0)
      {
         return t*mult;
      }
      return s*mult;
   }
   if (i == 0)
   {
      return s*mult;
   }
   return t*mult;
}

/// Return the i'th singular value of the matrix of size 3 with given @a data.
template<> MFEM_HOST_DEVICE inline
double CalcSingularvalue<3>(const double *data, const int i)
{
   double d0, d1, d2, d3, d4, d5, d6, d7, d8;
   d0 = data[0];  d3 = data[3];  d6 = data[6];
   d1 = data[1];  d4 = data[4];  d7 = data[7];
   d2 = data[2];  d5 = data[5];  d8 = data[8];
   double mult;
   {
      double d_max = fabs(d0);
      if (d_max < fabs(d1)) { d_max = fabs(d1); }
      if (d_max < fabs(d2)) { d_max = fabs(d2); }
      if (d_max < fabs(d3)) { d_max = fabs(d3); }
      if (d_max < fabs(d4)) { d_max = fabs(d4); }
      if (d_max < fabs(d5)) { d_max = fabs(d5); }
      if (d_max < fabs(d6)) { d_max = fabs(d6); }
      if (d_max < fabs(d7)) { d_max = fabs(d7); }
      if (d_max < fabs(d8)) { d_max = fabs(d8); }
      internal::GetScalingFactor(d_max, mult);
   }

   d0 /= mult;  d1 /= mult;  d2 /= mult;
   d3 /= mult;  d4 /= mult;  d5 /= mult;
   d6 /= mult;  d7 /= mult;  d8 /= mult;

   double b11 = d0*d0 + d1*d1 + d2*d2;
   double b12 = d0*d3 + d1*d4 + d2*d5;
   double b13 = d0*d6 + d1*d7 + d2*d8;
   double b22 = d3*d3 + d4*d4 + d5*d5;
   double b23 = d3*d6 + d4*d7 + d5*d8;
   double b33 = d6*d6 + d7*d7 + d8*d8;

   // double a, b, c;
   // a = -(b11 + b22 + b33);
   // b = b11*(b22 + b33) + b22*b33 - b12*b12 - b13*b13 - b23*b23;
   // c = b11*(b23*b23 - b22*b33) + b12*(b12*b33 - 2*b13*b23) + b13*b13*b22;

   // double Q = (a * a - 3 * b) / 9;
   // double Q = (b12*b12 + b13*b13 + b23*b23 +
   //             ((b11 - b22)*(b11 - b22) +
   //              (b11 - b33)*(b11 - b33) +
   //              (b22 - b33)*(b22 - b33))/6)/3;
   // Q = (3*(b12^2 + b13^2 + b23^2) +
   //      ((b11 - b22)^2 + (b11 - b33)^2 + (b22 - b33)^2)/2)/9
   //   or
   // Q = (1/6)*|B-tr(B)/3|_F^2
   // Q >= 0 and
   // Q = 0  <==> B = scalar * I
   // double R = (2 * a * a * a - 9 * a * b + 27 * c) / 54;
   double aa = (b11 + b22 + b33)/3;  // aa = tr(B)/3
   double c1, c2, c3;
   // c1 = b11 - aa; // ((b11 - b22) + (b11 - b33))/3
   // c2 = b22 - aa; // ((b22 - b11) + (b22 - b33))/3
   // c3 = b33 - aa; // ((b33 - b11) + (b33 - b22))/3
   {
      double b11_b22 = ((d0-d3)*(d0+d3)+(d1-d4)*(d1+d4)+(d2-d5)*(d2+d5));
      double b22_b33 = ((d3-d6)*(d3+d6)+(d4-d7)*(d4+d7)+(d5-d8)*(d5+d8));
      double b33_b11 = ((d6-d0)*(d6+d0)+(d7-d1)*(d7+d1)+(d8-d2)*(d8+d2));
      c1 = (b11_b22 - b33_b11)/3;
      c2 = (b22_b33 - b11_b22)/3;
      c3 = (b33_b11 - b22_b33)/3;
   }
   double Q, R;
   Q = (2*(b12*b12 + b13*b13 + b23*b23) + c1*c1 + c2*c2 + c3*c3)/6;
   R = (c1*(b23*b23 - c2*c3)+ b12*(b12*c3 - 2*b13*b23) +b13*b13*c2)/2;
   // R = (-1/2)*det(B-(tr(B)/3)*I)
   // Note: 54*(det(S))^2 <= |S|_F^6, when S^t=S and tr(S)=0, S is 3x3
   // Therefore: R^2 <= Q^3

   if (Q <= 0.) { ; }

   // else if (fabs(R) >= sqrtQ3)
   // {
   //    double det = (d[0] * (d[4] * d[8] - d[5] * d[7]) +
   //                  d[3] * (d[2] * d[7] - d[1] * d[8]) +
   //                  d[6] * (d[1] * d[5] - d[2] * d[4]));
   //
   //    if (R > 0.)
   //    {
   //       if (i == 2)
   //          // aa -= 2*sqrtQ;
   //          return fabs(det)/(aa + sqrtQ);
   //       else
   //          aa += sqrtQ;
   //    }
   //    else
   //    {
   //       if (i != 0)
   //          aa -= sqrtQ;
   //          // aa = fabs(det)/sqrt(aa + 2*sqrtQ);
   //       else
   //          aa += 2*sqrtQ;
   //    }
   // }

   else
   {
      double sqrtQ = sqrt(Q);
      double sqrtQ3 = Q*sqrtQ;
      // double sqrtQ3 = sqrtQ*sqrtQ*sqrtQ;
      // double sqrtQ3 = pow(Q, 1.5);
      double r;

      if (fabs(R) >= sqrtQ3)
      {
         if (R < 0.)
         {
            // R = -1.;
            r = 2*sqrtQ;
         }
         else
         {
            // R = 1.;
            r = -2*sqrtQ;
         }
      }
      else
      {
         R = R/sqrtQ3;

         // if (fabs(R) <= 0.95)
         if (fabs(R) <= 0.9)
         {
            if (i == 2)
            {
               aa -= 2*sqrtQ*cos(acos(R)/3);   // min
            }
            else if (i == 0)
            {
               aa -= 2*sqrtQ*cos((acos(R) + 2.0*M_PI)/3);   // max
            }
            else
            {
               aa -= 2*sqrtQ*cos((acos(R) - 2.0*M_PI)/3);   // mid
            }
            goto have_aa;
         }

         if (R < 0.)
         {
            r = -2*sqrtQ*cos((acos(R) + 2.0*M_PI)/3); // max
            if (i == 0)
            {
               aa += r;
               goto have_aa;
            }
         }
         else
         {
            r = -2*sqrtQ*cos(acos(R)/3); // min
            if (i == 2)
            {
               aa += r;
               goto have_aa;
            }
         }
      }

      // (tr(B)/3 + r) is the root which is separated from the other
      // two roots which are close to each other when |R| is close to 1

      c1 -= r;
      c2 -= r;
      c3 -= r;
      // aa += r;

      // Type of Householder reflections: z --> mu ek, where k is the index
      // of the entry in z with:
      // mode == 0: smallest absolute value --> angle closest to pi/2
      //            (eliminate large entries)
      // mode == 1: largest absolute value --> angle farthest from pi/2
      //            (eliminate small entries)
      const int mode = 1;

      // Find a unit vector z = (z1,z2,z3) in the "near"-kernel of
      //  |  c1  b12  b13 |
      //  | b12   c2  b23 | = B - aa*I
      //  | b13  b23   c3 |
      // This vector is also an eigenvector for B corresponding to aa
      // The vector z overwrites (c1,c2,c3).
      switch (internal::KernelVector3S(mode, b12, b13, b23, c1, c2, c3))
      {
         case 3:
            aa += r;
            goto have_aa;
         case 2:
         // ok, continue with the returned vector orthogonal to the kernel
         case 1:
            // ok, continue with the returned vector in the "near"-kernel
            ;
      }

      // Using the eigenvector c = (c1,c2,c3) to transform B into
      //                   | b11   0   0 |
      // B <-- Q P B P Q = |  0  b22 b23 |
      //                   |  0  b23 b33 |
      double v1, v2, v3, g;
      internal::Reduce3S(mode, b11, b22, b33, b12, b13, b23,
                         c1, c2, c3, v1, v2, v3, g);
      // Q = I - g v v^t
      // P - permutation matrix switching rows and columns 1 and k

      // find the eigenvalues of
      //  | b22 b23 |
      //  | b23 b33 |
      internal::Eigenvalues2S(b23, b22, b33);

      if (i == 2)
      {
         aa = fmin(fmin(b11, b22), b33);
      }
      else if (i == 1)
      {
         if (b11 <= b22)
         {
            aa = (b22 <= b33) ? b22 : fmax(b11, b33);
         }
         else
         {
            aa = (b11 <= b33) ? b11 : fmax(b33, b22);
         }
      }
      else
      {
         aa = fmax(fmax(b11, b22), b33);
      }
   }

have_aa:

   return sqrt(fabs(aa))*mult; // take abs before we sort?
}

} // namespace kernels

} // namespace mfem

#endif // MFEM_KERNELS_HPP
