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

#ifndef MFEM_DINVARIANTS_HPP
#define MFEM_DINVARIANTS_HPP

#include "../config/config.hpp"

#include "../general/cuda.hpp"
#include "../general/globals.hpp"

#include "dtensor.hpp"

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

   MFEM_HOST_DEVICE inline bool dont(int have_mask) const { return !(eval_state & have_mask); }

   MFEM_HOST_DEVICE inline void Eval_I1()
   {
      eval_state |= HAVE_I1;
      B[0] = J[0]*J[0] + J[3]*J[3] + J[6]*J[6];
      B[1] = J[1]*J[1] + J[4]*J[4] + J[7]*J[7];
      B[2] = J[2]*J[2] + J[5]*J[5] + J[8]*J[8];
      I1 = B[0] + B[1] + B[2];
   }

   MFEM_HOST_DEVICE inline void Eval_I1b() // det(J)^{-2/3}*I_1 = I_1/I_3^{1/3}
   {
      eval_state |= HAVE_I1b;
      I1b = Get_I1()*Get_I3b_p();
   }

   MFEM_HOST_DEVICE inline void Eval_B_offd()
   {
      eval_state |= HAVE_B_offd;
      // B = J J^t
      // B[3]=B(0,1), B[4]=B(0,2), B[5]=B(1,2)
      B[3] = J[0]*J[1] + J[3]*J[4] + J[6]*J[7]; // B(0,1)
      B[4] = J[0]*J[2] + J[3]*J[5] + J[6]*J[8]; // B(0,2)
      B[5] = J[1]*J[2] + J[4]*J[5] + J[7]*J[8]; // B(1,2)
   }
   MFEM_HOST_DEVICE inline void Eval_I2()
   {
      eval_state |= HAVE_I2;
      Get_I1();
      if (dont(HAVE_B_offd)) { Eval_B_offd(); }
      const double BF2 = B[0]*B[0] + B[1]*B[1] + B[2]*B[2] +
                         2*(B[3]*B[3] + B[4]*B[4] + B[5]*B[5]);
      I2 = (I1*I1 - BF2)/2;
   }
   MFEM_HOST_DEVICE inline void Eval_I2b() // I2b = I2*I3b^{-4/3}
   {
      eval_state |= HAVE_I2b;
      Get_I3b_p();
      I2b = Get_I2()*I3b_p*I3b_p;
   }
   MFEM_HOST_DEVICE inline void Eval_I3b() // det(J)
   {
      eval_state |= HAVE_I3b;
      I3b = J[0]*(J[4]*J[8] - J[7]*J[5]) - J[1]*(J[3]*J[8] - J[5]*J[6]) +
            J[2]*(J[3]*J[7] - J[4]*J[6]);
      sign_detJ = I3b >= 0.0 ? 1.0 : -1.0;
      I3b = sign_detJ*I3b;
   }
   MFEM_HOST_DEVICE inline double Get_I3b_p()  // I3b^{-2/3}
   {
      if (dont(HAVE_I3b_p))
      {
         eval_state |= HAVE_I3b_p;
         const double i3b = Get_I3b();
         I3b_p = sign_detJ * std::pow(i3b, -2./3.);
      }
      return I3b_p;
   }
   MFEM_HOST_DEVICE inline void Eval_dI1()
   {
      eval_state |= HAVE_dI1;
      for (int i = 0; i < 9; i++)
      {
         dI1[i] = 2*J[i];
      }
   }
   MFEM_HOST_DEVICE inline void Eval_dI1b()
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
   MFEM_HOST_DEVICE inline void Eval_dI2()
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
   MFEM_HOST_DEVICE inline void Eval_dI2b()
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
   MFEM_HOST_DEVICE inline void Eval_dI3()
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
   MFEM_HOST_DEVICE inline void Eval_dI3b()
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
   MFEM_HOST_DEVICE InvariantsEvaluator3D(const double *J): J(J), eval_state(0) { }

   MFEM_HOST_DEVICE inline double Get_I1()  { if (dont(HAVE_I1 )) { Eval_I1();  } return I1; }
   MFEM_HOST_DEVICE inline double Get_I1b() { if (dont(HAVE_I1b)) { Eval_I1b(); } return I1b; }
   MFEM_HOST_DEVICE inline double Get_I2()  { if (dont(HAVE_I2 )) { Eval_I2();  } return I2; }
   MFEM_HOST_DEVICE inline double Get_I2b() { if (dont(HAVE_I2b)) { Eval_I2b(); } return I2b; }
   MFEM_HOST_DEVICE inline double Get_I3()  { if (dont(HAVE_I3b)) { Eval_I3b(); } return I3b*I3b; }
   MFEM_HOST_DEVICE inline double Get_I3b() { if (dont(HAVE_I3b)) { Eval_I3b(); } return I3b; }

   MFEM_HOST_DEVICE inline const double *Get_dI1()
   {
      if (dont(HAVE_dI1 )) { Eval_dI1();  } return dI1;
   }
   MFEM_HOST_DEVICE inline const double *Get_dI1b()
   {
      if (dont(HAVE_dI1b)) { Eval_dI1b(); } return dI1b;
   }
   MFEM_HOST_DEVICE inline const double *Get_dI2()
   {
      if (dont(HAVE_dI2)) { Eval_dI2(); } return dI2;
   }
   MFEM_HOST_DEVICE inline const double *Get_dI2b()
   {
      if (dont(HAVE_dI2b)) { Eval_dI2b(); } return dI2b;
   }
   MFEM_HOST_DEVICE inline const double *Get_dI3()
   {
      if (dont(HAVE_dI3)) { Eval_dI3(); } return dI3;
   }
   MFEM_HOST_DEVICE inline const double *Get_dI3b()
   {
      if (dont(HAVE_dI3b)) { Eval_dI3b(); } return dI3b;
   }

   // *****************************************************************************
   // ddI1b = X1 + X2 + X3, where
   // X1_ijkl = (2/3*I1b/I3) [ 2/3 dI3b_ij dI3b_kl + dI3b_kj dI3b_il ]
   // X2_ijkl = (I3b^{-2/3}) ddI1_ijkl
   // X3_ijkl = -(4/3*I3b^{-5/3}) (J_ij dI3b_kl + dI3b_ij J_kl)
   MFEM_HOST_DEVICE inline const double *Get_ddI1b_ij(int i, int j)
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
   MFEM_HOST_DEVICE inline const double *Get_ddI2_ij(int i, int j)
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

      // x3_ijkl = -2 B_ik δ_jl
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
   MFEM_HOST_DEVICE inline const double *Get_ddI2b_ij(int i, int j)
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
      // X3_ijkl =  det(J)^{-4/3} ddI2_ijkl
      DeviceMatrix X3(X3_p,3,3);
      ConstDeviceMatrix ddI2(Get_ddI2_ij(i,j),3,3);
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

} // namespace kernels

} // namespace mfem

#endif // MFEM_DINVARIANTS_HPP
