// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_NONLININTEG
#define MFEM_NONLININTEG

#include "../config/config.hpp"
#include "fe.hpp"
#include "coefficient.hpp"

namespace mfem
{

/** The abstract base class NonlinearFormIntegrator is used to express the
    local action of a general nonlinear finite element operator. In addition
    it may provide the capability to assemble the local gradient operator
    and to compute the local energy. */
class NonlinearFormIntegrator
{
protected:
   const IntegrationRule *IntRule;

   NonlinearFormIntegrator(const IntegrationRule *ir = NULL)
      : IntRule(NULL) { }

public:
   /** @brief Prescribe a fixed IntegrationRule to use (when @a ir != NULL) or
       let the integrator choose (when @a ir == NULL). */
   void SetIntRule(const IntegrationRule *ir) { IntRule = ir; }

   /// Prescribe a fixed IntegrationRule to use.
   void SetIntegrationRule(const IntegrationRule &irule) { IntRule = &irule; }

   /// Perform the local action of the NonlinearFormIntegrator
   virtual void AssembleElementVector(const FiniteElement &el,
                                      ElementTransformation &Tr,
                                      const Vector &elfun, Vector &elvect) = 0;

   /// Assemble the local gradient matrix
   virtual void AssembleElementGrad(const FiniteElement &el,
                                    ElementTransformation &Tr,
                                    const Vector &elfun, DenseMatrix &elmat);

   /// Compute the local energy
   virtual double GetElementEnergy(const FiniteElement &el,
                                   ElementTransformation &Tr,
                                   const Vector &elfun);

   virtual ~NonlinearFormIntegrator() { }
};


class InvariantsEvaluator2D
{
protected:
   // Transformation Jacobian
   const double *J;

   // Invariants: I_1 = ||J||_F^2, \bar{I}_1 = I_1/det(J), \bar{I}_2 = det(J).
   double I1, I1b, I2b;

   // Derivatives of I1, I1b, I2, and I2b using column-major storage.
   double dI1[4], dI1b[4], dI2[4], dI2b[4];

   DenseMatrix D; // Always points to external data or is empty
   DenseMatrix DaJ, DJt;

   bool neg_detJ;

   enum EvalMasks
   {
      HAVE_I1   = 1,
      HAVE_I1b  = 2,
      HAVE_I2b  = 4,
      HAVE_dI1  = 8,
      HAVE_dI1b = 16,
      HAVE_dI2  = 32,
      HAVE_dI2b = 64,
      HAVE_DaJ  = 128, // D adj(J) = D dI2b^t
      HAVE_DJt  = 256  // D J^t
   };

   // Bitwise OR of EvalMasks
   int eval_state;

   bool dont(int have_mask) const { return !(eval_state & have_mask); }

   void Eval_I1()
   {
      eval_state |= HAVE_I1;
      I1 = J[0]*J[0] + J[1]*J[1] + J[2]*J[2] + J[3]*J[3];
   }
   void Eval_I1b()
   {
      eval_state |= HAVE_I1b;
      I1b = Get_I1()/Get_I2b();
   }
   void Eval_I2b()
   {
      eval_state |= HAVE_I2b;
      I2b = J[0]*J[3] - J[1]*J[2];
      neg_detJ = I2b < 0.0;
      I2b = std::abs(I2b);
   }
   void Eval_dI1()
   {
      eval_state |= HAVE_dI1;
      dI1[0] = 2*J[0]; dI1[1] = 2*J[1]; dI1[2] = 2*J[2]; dI1[3] = 2*J[3];
   }
   void Eval_dI1b()
   {
      eval_state |= HAVE_dI1b;
      // I1b = I1/I2b
      // dI1b = (1/I2b)*dI1 - (I1/I2b^2)*dI2b = (2/I2b)*[J - (I1b/2)*dI2b]
      const double c1 = 2.0/Get_I2b();
      const double c2 = Get_I1b()/2;
      Get_dI2b();
      dI1b[0] = c1*(J[0] - c2*dI2b[0]);
      dI1b[1] = c1*(J[1] - c2*dI2b[1]);
      dI1b[2] = c1*(J[2] - c2*dI2b[2]);
      dI1b[3] = c1*(J[3] - c2*dI2b[3]);
   }
   void Eval_dI2()
   {
      eval_state |= HAVE_dI2;
      // I2 = I2b^2
      // dI2 = 2*I2b*dI2b = 2*det(J)*adj(J)^T
      const double c1 = 2*Get_I2b();
      Get_dI2b();
      dI2[0] = c1*dI2b[0];
      dI2[1] = c1*dI2b[1];
      dI2[2] = c1*dI2b[2];
      dI2[3] = c1*dI2b[3];
   }
   void Eval_dI2b()
   {
      eval_state |= HAVE_dI2b;
      // I2b = det(J)
      // dI2b = adj(J)^T
      Get_I2b();
      if (!neg_detJ)
      {
         dI2b[0] =  J[3];
         dI2b[1] = -J[2];
         dI2b[2] = -J[1];
         dI2b[3] =  J[0];
      }
      else
      {
         dI2b[0] = -J[3];
         dI2b[1] =  J[2];
         dI2b[2] =  J[1];
         dI2b[3] = -J[0];
      }
   }
   void Eval_DaJ() // D adj(J) = D dI2b^t
   {
      MFEM_ASSERT(D.GetData(), "");
      eval_state |= HAVE_DaJ;
      DaJ.SetSize(D.Height(), D.Width());
      Get_dI2b();
      const int nd = D.Height();
      for (int i = 0; i < nd; i++)
      {
         // adj(J) = dI2b^t
         DaJ(i,0) = D(i,0)*dI2b[0] + D(i,1)*dI2b[2];
         DaJ(i,1) = D(i,0)*dI2b[1] + D(i,1)*dI2b[3];
      }
   }
   void Eval_DJt() // D J^t
   {
      MFEM_ASSERT(D.GetData(), "");
      eval_state |= HAVE_DJt;
      DJt.SetSize(D.Height(), D.Width());
      const int nd = D.Height();
      for (int i = 0; i < nd; i++)
      {
         DJt(i,0) = D(i,0)*J[0] + D(i,1)*J[2];
         DJt(i,1) = D(i,0)*J[1] + D(i,1)*J[3];
      }
   }

public:
   /// The Jacobian should use column-major storage.
   InvariantsEvaluator2D(const double *Jac = NULL)
      : J(Jac), eval_state(0) { }

   /// The Jacobian should use column-major storage.
   void SetJacobian(const double *Jac) { J = Jac; eval_state = 0; }

   void SetJacobian(const DenseMatrix &Jac)
   {
      MFEM_ASSERT(Jac.Height() == 2 && Jac.Width() == 2, "");
      SetJacobian(Jac.GetData());
   }

   void SetDerivativeMatrix(const DenseMatrix &Deriv)
   {
      MFEM_ASSERT(Deriv.Width() == 2, "");
      D.UseExternalData(Deriv.GetData(), Deriv.Height(), Deriv.Width());
      eval_state &= ~(HAVE_DaJ | HAVE_DJt);
   }

   double Get_I1()  { if (dont(HAVE_I1 )) { Eval_I1();  } return I1; }
   double Get_I1b() { if (dont(HAVE_I1b)) { Eval_I1b(); } return I1b; }
   double Get_I2()  { if (dont(HAVE_I2b)) { Eval_I2b(); } return I2b*I2b; }
   double Get_I2b() { if (dont(HAVE_I2b)) { Eval_I2b(); } return I2b; }

   const double *Get_dI1()
   {
      if (dont(HAVE_dI1 )) { Eval_dI1();  } return dI1;
   }
   const double *Get_dI1b()
   {
      if (dont(HAVE_dI1b)) { Eval_dI1b(); } return dI1b;
   }
   const double *Get_dI2()
   {
      if (dont(HAVE_dI2)) { Eval_dI2(); } return dI2;
   }
   const double *Get_dI2b()
   {
      if (dont(HAVE_dI2b)) { Eval_dI2b(); } return dI2b;
   }

   // Assemble operation for tensor X with components X_jslt:
   //    A(i+nd*j,k+nd*l) += (\sum_st  w D_is X_jslt D_kt)
   //    0 <= i,k < nd,  0 <= j,l,s,t < 2
   // where nd is the height of D, i.e. the number of DOFs in one component.

   void Assemble_ddI1(double w, DenseMatrix &A)
   {
      // ddI1_jslt = 2 I_jslt = 2 δ_jl δ_st
      //    A(i+nd*j,k+nd*l) += (\sum_st  2 w D_is δ_jl δ_st D_kt)
      // or
      //    A(i+nd*j,k+nd*l) += (2 w) (\sum_s  D_is D_ks) δ_jl
      //    A(i+nd*j,k+nd*l) += (2 w) (D D^t)_ik δ_jl

      const int nd = D.Height();
      const double a = 2*w;
      for (int i = 0; i < nd; i++)
      {
         const double aDi0 = a*D(i,0), aDi1 = a*D(i,1);
         // k == i
         const double aDDt_ii = aDi0*D(i,0) + aDi1*D(i,1);
         A(i+nd*0,i+nd*0) += aDDt_ii;
         A(i+nd*1,i+nd*1) += aDDt_ii;
         // 0 <= k < i
         for (int k = 0; k < i; k++)
         {
            const double aDDt_ik = aDi0*D(k,0) + aDi1*D(k,1);
            A(i+nd*0,k+nd*0) += aDDt_ik;
            A(k+nd*0,i+nd*0) += aDDt_ik;
            A(i+nd*1,k+nd*1) += aDDt_ik;
            A(k+nd*1,i+nd*1) += aDDt_ik;
         }
      }
   }
   void Assemble_ddI1b(double w, DenseMatrix &A)
   {
      // ddI1b = X1 + X2 + X3, where
      // X1_ijkl = (I1b/I2) [ (δ_ks δ_it + δ_kt δ_si) dI2b_tj dI2b_sl ]
      //         = (I1b/I2) [ dI2b_ij dI2b_kl + dI2b_kj dI2b_il ]
      // X2_ijkl = (2/I2b) δ_ik δ_jl = (1/I2b) ddI1_ijkl
      // X3_ijkl = -(2/I2) (δ_ks δ_it) (J_tj dI2b_sl + dI2b_tj J_sl)
      //         = -(2/I2) (J_ij dI2b_kl + dI2b_ij J_kl)
      //
      //    A(i+nd*j,k+nd*l) += (\sum_st  w D_is ddI1b_jslt D_kt)
      // or
      //    A(i+nd*j,k+nd*l) +=
      //       w (I1b/I2) [(D dI2b^t)_ij (D dI2b^t)_kl +
      //                   (D dI2b^t)_il (D dI2b^t)_kj]
      //     + w (2/I2b)  δ_jl (D D^t)_ik
      //     - w (2/I2)   [(D J^t)_ij (D dI2b^t)_kl + (D dI2b^t)_ij (D J^t)_kl]

      if (dont(HAVE_DaJ)) { Eval_DaJ(); }
      if (dont(HAVE_DJt)) { Eval_DJt(); }
      const int nd = D.Height();
      const double a = w*Get_I1b()/Get_I2();
      const double b = 2*w/Get_I2b();
      const double c = -2*w/Get_I2();
      for (int i = 0; i < nd; i++)
      {
         const double aDaJ_i0 = a*DaJ(i,0), aDaJ_i1 = a*DaJ(i,1);
         const double bD_i0 = b*D(i,0), bD_i1 = b*D(i,1);
         const double cDJt_i0 = c*DJt(i,0), cDJt_i1 = c*DJt(i,1);
         const double cDaJ_i0 = c*DaJ(i,0), cDaJ_i1 = c*DaJ(i,1);
         // k == i
         {
            // Symmetries: A2_ii_00 = A2_ii_11
            const double A2_ii = bD_i0*D(i,0) + bD_i1*D(i,1);

            A(i+nd*0,i+nd*0) += 2*(aDaJ_i0 + cDJt_i0)*DaJ(i,0) + A2_ii;

            // Symmetries: A_ii_01 = A_ii_10
            const double A_ii_01 =
               2*aDaJ_i0*DaJ(i,1) + cDJt_i0*DaJ(i,1) + cDaJ_i0*DJt(i,1);
            A(i+nd*0,i+nd*1) += A_ii_01;
            A(i+nd*1,i+nd*0) += A_ii_01;

            A(i+nd*1,i+nd*1) += 2*(aDaJ_i1 + cDJt_i1)*DaJ(i,1) + A2_ii;
         }
         // 0 <= k < i
         for (int k = 0; k < i; k++)
         {
            // Symmetries: A1_ik_01 = A1_ik_10 = A1_ki_01 = A1_ki_10
            const double A1_ik_01 = aDaJ_i0*DaJ(k,1) + aDaJ_i1*DaJ(k,0);

            // Symmetries: A2_ik_00 = A2_ik_11 = A2_ki_00 = A2_ki_11
            const double A2_ik = bD_i0*D(k,0) + bD_i1*D(k,1);

            const double A_ik_00 =
               (2*aDaJ_i0 + cDJt_i0)*DaJ(k,0) + A2_ik + cDaJ_i0*DJt(k,0);
            A(i+nd*0,k+nd*0) += A_ik_00;
            A(k+nd*0,i+nd*0) += A_ik_00;

            const double A_ik_01 =
               A1_ik_01 + cDJt_i0*DaJ(k,1) + cDaJ_i0*DJt(k,1);
            A(i+nd*0,k+nd*1) += A_ik_01;
            A(k+nd*1,i+nd*0) += A_ik_01;

            const double A_ik_10 =
               A1_ik_01 + cDJt_i1*DaJ(k,0) + cDaJ_i1*DJt(k,0);
            A(i+nd*1,k+nd*0) += A_ik_10;
            A(k+nd*0,i+nd*1) += A_ik_10;

            const double A_ik_11 =
               (2*aDaJ_i1 + cDJt_i1)*DaJ(k,1) + A2_ik + cDaJ_i1*DJt(k,1);
            A(i+nd*1,k+nd*1) += A_ik_11;
            A(k+nd*1,i+nd*1) += A_ik_11;
         }
      }
   }
   void Assemble_ddI2(double w, DenseMatrix &A)
   {
      // ddI2_ijkl = 2 (2 δ_ks δ_it - δ_kt δ_si) dI2b_tj dI2b_sl
      //           = 4 dI2b_ij dI2b_kl - 2 dI2b_kj dI2b_il
      //           = 2 dI2b_ij dI2b_kl + 2 (dI2b_ij dI2b_kl - dI2b_kj dI2b_il)
      //
      //    A(i+nd*j,k+nd*l) += (\sum_st  w D_is ddI2_jslt D_kt)
      // or
      //    A(i+nd*j,k+nd*l) +=
      //       (\sum_st  w D_is (4 dI2b_js dI2b_lt - 2 dI2b_ls dI2b_jt) D_kt)
      //    A(i+nd*j,k+nd*l) +=
      //       2 w [2 (D dI2b^t)_ij (D dI2b^t)_kl - (D dI2b^t)_il (D dI2b^t)_kj]
      //
      // Note: the expression
      //    (D dI2b^t)_ij (D dI2b^t)_kl - (D dI2b^t)_il (D dI2b^t)_kj
      // is the determinant of the 2x2 matrix formed by rows {i,k} and columns
      // {j,l} from the matrix (D dI2b^t).

      if (dont(HAVE_DaJ)) { Eval_DaJ(); }
      const int nd = D.Height();
      const double a = 2*w;
      Vector DaJ_as_vec(DaJ.GetData(), 2*nd);
      AddMult_a_VVt(a, DaJ_as_vec, A);
      const int j = 1, l = 0;
      for (int i = 0; i < nd; i++)
      {
         const double aDaJ_ij = a*DaJ(i,j), aDaJ_il = a*DaJ(i,l);
         for (int k = 0; k < i; k++)
         {
            const double A_ijkl = aDaJ_ij*DaJ(k,l) - aDaJ_il*DaJ(k,j);
            A(i+nd*j,k+nd*l) += A_ijkl;
            A(k+nd*l,i+nd*j) += A_ijkl;
            A(k+nd*j,i+nd*l) -= A_ijkl;
            A(i+nd*l,k+nd*j) -= A_ijkl;
         }
      }
   }
   void Assemble_ddI2b(double w, DenseMatrix &A)
   {
      // ddI2b_ijkl = (1/I2b) (δ_ks δ_it - δ_kt δ_si) dI2b_tj dI2b_sl
      //    [j -> u], [l -> v], [i -> j], [k -> l]
      // ddI2b_julv = (1/I2b) (δ_ls δ_jt - δ_lt δ_sj) dI2b_tu dI2b_sv
      //
      //    A(i+nd*j,k+nd*l) += (\sum_st  w D_is ddI2b_jslt D_kt)
      // or
      //    A(i+nd*j,k+nd*l) += (\sum_uv  w D_iu ddI2b_julv D_kv)
      //    A(i+nd*j,k+nd*l) +=
      //       (\sum_uvst (w/I2b)
      //          D_iu (δ_ls δ_jt - δ_lt δ_sj) dI2b_tu dI2b_sv D_kv)
      //    A(i+nd*j,k+nd*l) +=
      //       (\sum_st (w/I2b)
      //          (D dI2b^t)_it (δ_ls δ_jt - δ_lt δ_sj) (D dI2b^t)_ks)
      //    A(i+nd*j,k+nd*l) += (w/I2b)
      //       [ (D dI2b^t)_ij (D dI2b^t)_kl - (D dI2b^t)_il (D dI2b^t)_kj ]

      if (dont(HAVE_DaJ)) { Eval_DaJ(); }
      const int nd = D.Height();
      const int j = 1, l = 0;
      const double a = w/Get_I2b();
      for (int i = 0; i < nd; i++)
      {
         const double aDaJ_ij = a*DaJ(i,j), aDaJ_il = a*DaJ(i,l);
         for (int k = 0; k < i; k++)
         {
            const double A_ijkl = aDaJ_ij*DaJ(k,l) - aDaJ_il*DaJ(k,j);
            A(i+nd*j,k+nd*l) += A_ijkl;
            A(k+nd*l,i+nd*j) += A_ijkl;
            A(k+nd*j,i+nd*l) -= A_ijkl;
            A(i+nd*l,k+nd*j) -= A_ijkl;
         }
      }
   }
};


class InvariantsEvaluator3D
{
protected:
   // Transformation Jacobian
   const double *J;

   // Invariants:
   //    I_1 = ||J||_F^2, \bar{I}_1 = det(J)^{-2/3}*I_1,
   //    I_2 = (1/2)*(||J||_F^4-||J J^t||_F^2) = (1/2)*(I_1^2-||J J^t||_F^2),
   //    \bar{I}_2 = det(J)^{-4/3}*I_2,
   //    I_3 = det(J)^2, \bar{I}_3 = det(J).
   double I1, I1b, I2, I2b, I3b;

   // Derivatives of I1, I1b, I2, I2b, I3, and I3b using column-major storage.
   double dI1[9], dI1b[9], dI2[9], dI2b[9], dI3[9], dI3b[9];
   double B[6]; // B = J J^t (diagonal entries first, then off-diagonal)

   DenseMatrix D; // Always points to external data or is empty
   DenseMatrix DaJ, DJt;

   enum EvalMasks
   {
      HAVE_I1   = 1,
      HAVE_I1b  = 2,
      HAVE_I2   = 4,
      HAVE_I2b  = 8,
      HAVE_I3b  = 16,
      HAVE_dI1  = 32,
      HAVE_dI1b = 64,
      HAVE_dI2  = 128,
      HAVE_dI2b = 256,
      HAVE_dI3  = 512,
      HAVE_dI3b = 1024,
      HAVE_DaJ  = 2048, // D adj(J) = D dI3b^t
      HAVE_DJt  = 4096  // D J^t
   };

   // Bitwise OR of EvalMasks
   int eval_state;

   bool dont(int have_mask) const { return !(eval_state & have_mask); }

   void Eval_I1()
   {
      eval_state |= HAVE_I1;
      B[0] = J[0]*J[0] + J[3]*J[3] + J[6]*J[6];
      B[1] = J[1]*J[1] + J[4]*J[4] + J[7]*J[7];
      B[2] = J[2]*J[2] + J[5]*J[5] + J[8]*J[8];
      I1 = B[0] + B[1] + B[2];
   }
   void Eval_I1b() // det(J)^{-2/3}*I_1 = I_1/I_3^{1/3}
   {
      eval_state |= HAVE_I1b;
      // I1b = Get_I1()/std::cbrt(Get_I3()); // c++11
      I1b = Get_I1()*std::pow(Get_I3b(), -2./3.);
   }
   void Eval_I2()
   {
      eval_state |= HAVE_I2;
      Get_I1();
      B[3] = J[0]*J[1] + J[3]*J[4] + J[6]*J[7];
      B[4] = J[0]*J[2] + J[3]*J[5] + J[6]*J[8];
      B[5] = J[1]*J[2] + J[4]*J[5] + J[7]*J[8];
      const double BF2 = B[0]*B[0] + B[1]*B[1] + B[2]*B[2] +
                         2*(B[3]*B[3] + B[4]*B[4] + B[5]*B[5]);
      I2 = 0.5*(I1*I1 - BF2);
   }
   void Eval_I2b() // I2b = I2*I3b^{-4/3}
   {
      eval_state |= HAVE_I2b;
      I2b = Get_I2()*std::pow(Get_I3b(), -4./3.);
   }
   void Eval_I3b() // det(J)
   {
      eval_state |= HAVE_I3b;
      I3b = J[0]*(J[4]*J[8] - J[7]*J[5]) - J[1]*(J[3]*J[8] - J[5]*J[6]) +
            J[2]*(J[3]*J[7] - J[4]*J[6]);
   }
   void Eval_dI1()
   {
      eval_state |= HAVE_dI1;
      for (int i = 0; i < 9; i++)
      {
         dI1[i] = 2*J[i];
      }
   }
   void Eval_dI1b()
   {
      eval_state |= HAVE_dI1b;
      // I1b = I3b^{-2/3}*I1
      // dI1b = 2*I3b^{-2/3}*(J - (1/3)*I1/I3b*dI3b)
      // const double c1 = 2.0/std::cbrt(Get_I3()); // c++11
      const double c1 = 2.0*std::pow(Get_I3b(), -2./3.);
      const double c2 = Get_I1()/(3*I3b);
      Get_dI3b();
      for (int i = 0; i < 9; i++)
      {
         dI1b[i] = c1*(J[i] - c2*dI3b[i]);
      }
   }
   void Eval_dI2()
   {
      // FIXME
      mfem_error("TODO");
   }
   void Eval_dI2b()
   {
      // FIXME
      mfem_error("TODO");
   }
   void Eval_dI3()
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
   void Eval_dI3b()
   {
      eval_state |= HAVE_dI3b;
      // I3b = det(J)
      // dI3b = adj(J)^T
      dI2b[0] = J[4]*J[8] - J[5]*J[7];  // 0  3  6
      dI2b[1] = J[5]*J[6] - J[3]*J[8];  // 1  4  7
      dI2b[2] = J[3]*J[7] - J[4]*J[6];  // 2  5  8
      dI2b[3] = J[2]*J[7] - J[1]*J[8];
      dI2b[4] = J[0]*J[8] - J[2]*J[6];
      dI2b[5] = J[1]*J[6] - J[0]*J[7];
      dI2b[6] = J[1]*J[5] - J[2]*J[4];
      dI2b[7] = J[2]*J[3] - J[0]*J[5];
      dI2b[8] = J[0]*J[4] - J[1]*J[3];
   }
   void Eval_DaJ() // DaJ = D adj(J) = D dI3b^t
   {
      MFEM_ASSERT(D.GetData(), "");
      eval_state |= HAVE_DaJ;
      DaJ.SetSize(D.Height(), D.Width());
      Get_dI3b();
      const int nd = D.Height();
      for (int i = 0; i < nd; i++)
      {
         // adj(J) = dI3b^t
         DaJ(i,0) = D(i,0)*dI3b[0] + D(i,1)*dI3b[3] + D(i,2)*dI3b[6];
         DaJ(i,1) = D(i,0)*dI3b[1] + D(i,1)*dI3b[4] + D(i,2)*dI3b[7];
         DaJ(i,2) = D(i,0)*dI3b[2] + D(i,1)*dI3b[5] + D(i,2)*dI3b[8];
      }
   }
   void Eval_DJt() // DJt = D J^t
   {
      MFEM_ASSERT(D.GetData(), "");
      eval_state |= HAVE_DJt;
      DJt.SetSize(D.Height(), D.Width());
      const int nd = D.Height();
      for (int i = 0; i < nd; i++)
      {
         DJt(i,0) = D(i,0)*J[0] + D(i,1)*J[3] + D(i,2)*J[6];
         DJt(i,1) = D(i,0)*J[1] + D(i,1)*J[4] + D(i,2)*J[7];
         DJt(i,2) = D(i,0)*J[2] + D(i,1)*J[5] + D(i,2)*J[8];
      }
   }

public:
   /// The Jacobian should use column-major storage.
   InvariantsEvaluator3D(const double *Jac = NULL)
      : J(Jac), eval_state(0) { }

   /// The Jacobian should use column-major storage.
   void SetJacobian(const double *Jac) { J = Jac; eval_state = 0; }

   void SetJacobian(const DenseMatrix &Jac)
   {
      MFEM_ASSERT(Jac.Height() == 3 && Jac.Width() == 3, "");
      SetJacobian(Jac.GetData());
   }

   void SetDerivativeMatrix(const DenseMatrix &Deriv)
   {
      MFEM_ASSERT(Deriv.Width() == 3, "");
      D.UseExternalData(Deriv.GetData(), Deriv.Height(), Deriv.Width());
      eval_state &= ~(HAVE_DaJ | HAVE_DJt);
   }

   double Get_I1()  { if (dont(HAVE_I1 )) { Eval_I1();  } return I1; }
   double Get_I1b() { if (dont(HAVE_I1b)) { Eval_I1b(); } return I1b; }
   double Get_I2()  { if (dont(HAVE_I2 )) { Eval_I2();  } return I2; }
   double Get_I2b() { if (dont(HAVE_I2b)) { Eval_I2b(); } return I2b; }
   double Get_I3()  { if (dont(HAVE_I3b)) { Eval_I3b(); } return I3b*I3b; }
   double Get_I3b() { if (dont(HAVE_I3b)) { Eval_I3b(); } return I3b; }

   const double *Get_dI1()
   {
      if (dont(HAVE_dI1 )) { Eval_dI1();  } return dI1;
   }
   const double *Get_dI1b()
   {
      if (dont(HAVE_dI1b)) { Eval_dI1b(); } return dI1b;
   }
   const double *Get_dI2()
   {
      if (dont(HAVE_dI2)) { Eval_dI2(); } return dI2;
   }
   const double *Get_dI2b()
   {
      if (dont(HAVE_dI2b)) { Eval_dI2b(); } return dI2b;
   }
   const double *Get_dI3()
   {
      if (dont(HAVE_dI3)) { Eval_dI3(); } return dI3;
   }
   const double *Get_dI3b()
   {
      if (dont(HAVE_dI3b)) { Eval_dI3b(); } return dI3b;
   }

   // Assemble operation for tensor X with components X_jslt:
   //    A(i+nd*j,k+nd*l) += (\sum_st  w D_is X_jslt D_kt)
   //    0 <= i,k < nd,  0 <= j,l,s,t < 3
   // where nd is the height of D, i.e. the number of DOFs in one component.

   void Assemble_ddI1(double w, DenseMatrix &A)
   {
      // ddI1_jslt = 2 I_jslt = 2 δ_jl δ_st
      //    A(i+nd*j,k+nd*l) += (\sum_st  2 w D_is δ_jl δ_st D_kt)
      // or
      //    A(i+nd*j,k+nd*l) += (2 w) (\sum_s  D_is D_ks) δ_jl
      //    A(i+nd*j,k+nd*l) += (2 w) (D D^t)_ik δ_jl

      const int nd = D.Height();
      const double a = 2*w;
      for (int i = 0; i < nd; i++)
      {
         const double aDi0 = a*D(i,0), aDi1 = a*D(i,1), aDi2 = a*D(i,2);
         // k == i
         const double aDDt_ii = aDi0*D(i,0) + aDi1*D(i,1) + aDi2*D(i,2);
         A(i+nd*0,i+nd*0) += aDDt_ii;
         A(i+nd*1,i+nd*1) += aDDt_ii;
         A(i+nd*2,i+nd*2) += aDDt_ii;
         // 0 <= k < i
         for (int k = 0; k < i; k++)
         {
            const double aDDt_ik = aDi0*D(k,0) + aDi1*D(k,1) + aDi2*D(k,2);
            A(i+nd*0,k+nd*0) += aDDt_ik;
            A(k+nd*0,i+nd*0) += aDDt_ik;
            A(i+nd*1,k+nd*1) += aDDt_ik;
            A(k+nd*1,i+nd*1) += aDDt_ik;
            A(i+nd*2,k+nd*2) += aDDt_ik;
            A(k+nd*2,i+nd*2) += aDDt_ik;
         }
      }
   }
   void Assemble_ddI1b(double w, DenseMatrix &A)
   {
      // Similar to InvariantsEvaluator2D::Assemble_ddI1b():
      //
      // ddI1b = X1 + X2 + X3, where
      // X1_ijkl = (2/3*I1b/I3) [ (2/3 δ_ks δ_it + δ_kt δ_si) dI3b_tj dI3b_sl ]
      //         = (2/3*I1b/I3) [ 2/3 dI3b_ij dI3b_kl + dI3b_kj dI3b_il ]
      // X2_ijkl = (2*I3b^{-2/3}) δ_ik δ_jl = (I3b^{-2/3}) ddI1_ijkl
      // X3_ijkl = -(4/3*I3b^{-5/3}) (δ_ks δ_it) (J_tj dI3b_sl + dI3b_tj J_sl)
      //         = -(4/3*I3b^{-5/3}) (J_ij dI3b_kl + dI3b_ij J_kl)
      //
      //    A(i+nd*j,k+nd*l) += (\sum_st  w D_is ddI1b_jslt D_kt)
      // or
      //    A(i+nd*j,k+nd*l) +=
      //       w (2/3*I1b/I3) [ 2/3 DaJ_ij DaJ_kl + DaJ_il DaJ_kj ]
      //     + w (2*I3b^{-2/3}) (D D^t)_ik δ_jl
      //     - w (4/3*I3b^{-5/3}) [ DJt_ij DaJ_kl + DaJ_ij DJt_kl ]

      if (dont(HAVE_DaJ)) { Eval_DaJ(); }
      if (dont(HAVE_DJt)) { Eval_DJt(); }
      const int nd = D.Height();
      const double r23 = 2./3.;
      const double r53 = 5./3.;
      const double a = r23*w*Get_I1b()/Get_I3();
      const double b = 2*w*std::pow(I3b, -r23);
      const double c = -r23*b/I3b;
      for (int i = 0; i < nd; i++)
      {
         // A1a_ik_jl = 2/3 a DaJ_ij DaJ_kl, A1b_ik_jl = a DaJ_il DaJ_kj
         // Symmetries: A1a_ik_jl = A1a_ki_lj = 2/3 A1b_ik_lj = 2/3 A1b_ki_jl
         //    A1_ik_jl = A1_ki_lj =     A1b_ik_jl + 2/3 A1b_ik_lj
         //    A1_ik_lj = A1_ki_jl = 2/3 A1b_ik_jl +     A1b_ik_lj
         // k == i:
         //    A1_ii_jl = A1_ii_lj = (5/3) a DaJ_ij DaJ_il
         // l == j:
         //    A1_ik_jj = A1_ki_jj = (5/3) a DaJ_ij DaJ_kj
         // k == i && l == j:
         //    A1_ii_jj = (5/3) a DaJ_ij^2

         // A2_ik_jl = b (D D^t)_ik δ_jl
         // Symmetries:

         // A3_ik_jl = c [ DJt_ij DaJ_kl + DaJ_ij DJt_kl ]
         // Symmetries:
         //    A3_ik_jl = A3_ki_lj = c [ DJt_ij DaJ_kl + DaJ_ij DJt_kl ]
         //    A3_ik_lj = A3_ki_jl = c [ DJt_il DaJ_kj + DaJ_il DJt_kj ]
         // k == i:
         //    A3_ii_jl = A3_ii_lj = c [ DJt_ij DaJ_il + DaJ_ij DJt_il ]
         // l == j:
         //    A3_ik_jj = A3_ki_jj =  c [ DJt_ij DaJ_kj + DaJ_ij DJt_kj ]
         // k == i && l == j:
         //    A3_ii_jj = 2 c DJt_ij DaJ_ij

         const double aDaJ_i[3] = { a*DaJ(i,0), a*DaJ(i,1), a*DaJ(i,2) };
         const double bD_i[3] = { b*D(i,0), b*D(i,1), b*D(i,2) };
         const double cDJt_i[3] = { c*DJt(i,0), c*DJt(i,1), c*DJt(i,2) };
         const double cDaJ_i[3] = { c*DaJ(i,0), c*DaJ(i,1), c*DaJ(i,2) };
         // k == i
         {
            // Symmetries: A2_ii_00 = A2_ii_11 = A2_ii_22
            const double A2_ii = bD_i[0]*D(i,0)+bD_i[1]*D(i,1)+bD_i[2]*D(i,2);
            A(i+nd*0,i+nd*0) += (r53*aDaJ_i[0] + 2*cDJt_i[0])*DaJ(i,0) + A2_ii;
            A(i+nd*1,i+nd*1) += (r53*aDaJ_i[1] + 2*cDJt_i[1])*DaJ(i,1) + A2_ii;
            A(i+nd*2,i+nd*2) += (r53*aDaJ_i[2] + 2*cDJt_i[2])*DaJ(i,2) + A2_ii;

            // Symmetries: A_ii_jl = A_ii_lj
            for (int j = 1; j < 3; j++)
            {
               for (int l = 0; l < j; l++)
               {
                  const double A_ii_jl =
                     (r53*aDaJ_i[j] + cDJt_i[j])*DaJ(i,l) + cDaJ_i[j]*DJt(i,l);
                  A(i+nd*j,i+nd*l) += A_ii_jl;
                  A(i+nd*l,i+nd*j) += A_ii_jl;
               }
            }
         }
         // 0 <= k < i
         for (int k = 0; k < i; k++)
         {
            // Symmetries: A2_ik_jj = A2_ki_ll
            const double A2_ik = bD_i[0]*D(k,0)+bD_i[1]*D(k,1)+bD_i[2]*D(k,2);

            // l == j
            for (int j = 0; j < 3; j++)
            {
               const double A_ik_jj = (r53*aDaJ_i[j] + cDJt_i[j])*DaJ(k,j) +
                                      cDaJ_i[j]*DJt(k,j) + A2_ik;
               A(i+nd*j,k+nd*j) += A_ik_jj;
               A(k+nd*j,i+nd*j) += A_ik_jj;
            }

            // 0 <= l < j
            for (int j = 1; j < 3; j++)
            {
               for (int l = 0; l < j; l++)
               {
                  // A1b_ik_jl = a DaJ_il DaJ_kj
                  const double A1b_ik_jl = aDaJ_i[l]*DaJ(k,j);
                  const double A1b_ik_lj = aDaJ_i[j]*DaJ(k,l);
                  // A1_ik_jl = A1_ki_lj =     A1b_ik_jl + 2/3 A1b_ik_lj
                  // A1_ik_lj = A1_ki_jl = 2/3 A1b_ik_jl +     A1b_ik_lj
                  // A3_ik_jl = c [ DJt_ij DaJ_kl + DaJ_ij DJt_kl ]
                  const double A_ik_jl = A1b_ik_jl + r23*A1b_ik_lj +
                                         cDJt_i[j]*DaJ(k,l)+cDaJ_i[j]*DJt(k,l);
                  A(i+nd*j,k+nd*l) += A_ik_jl;
                  A(k+nd*l,i+nd*j) += A_ik_jl;
                  const double A_ik_lj = r23*A1b_ik_jl + A1b_ik_lj +
                                         cDJt_i[l]*DaJ(k,j)+cDaJ_i[l]*DJt(k,j);
                  A(i+nd*l,k+nd*j) += A_ik_lj;
                  A(k+nd*j,i+nd*l) += A_ik_lj;
               }
            }
         }
      }
   }
   void Assemble_ddI2(double w, DenseMatrix &A)
   {
      // FIXME
      mfem_error("TODO");
   }
   void Assemble_ddI2b(double w, DenseMatrix &A)
   {
      // FIXME
      mfem_error("TODO");
   }
   void Assemble_ddI3(double w, DenseMatrix &A)
   {
      // Similar to InvariantsEvaluator2D::Assemble_ddI2():
      //
      //    A(i+nd*j,k+nd*l) += 2 w [ 2 DaJ_ij DaJ_kl - DaJ_il DaJ_kj ]
      //
      // Note: the expression ( DaJ_ij DaJ_kl - DaJ_il DaJ_kj ) is the
      // determinant of the 2x2 matrix formed by rows {i,k} and columns {j,l}
      // from the matrix DaJ = D dI3b^t.

      if (dont(HAVE_DaJ)) { Eval_DaJ(); }
      const int nd = D.Height();
      const double a = 2*w;
      Vector DaJ_as_vec(DaJ.GetData(), 3*nd);
      AddMult_a_VVt(a, DaJ_as_vec, A);
      for (int j = 1; j < 3; j++)
      {
         for (int l = 0; l < j; l++)
         {
            for (int i = 0; i < nd; i++)
            {
               const double aDaJ_ij = a*DaJ(i,j), aDaJ_il = a*DaJ(i,l);
               for (int k = 0; k < i; k++)
               {
                  const double A_ijkl = aDaJ_ij*DaJ(k,l) - aDaJ_il*DaJ(k,j);
                  A(i+nd*j,k+nd*l) += A_ijkl;
                  A(k+nd*l,i+nd*j) += A_ijkl;
                  A(k+nd*j,i+nd*l) -= A_ijkl;
                  A(i+nd*l,k+nd*j) -= A_ijkl;
               }
            }
         }
      }
   }
   void Assemble_ddI3b(double w, DenseMatrix &A)
   {
      // Similar to InvariantsEvaluator2D::Assemble_ddI2b():
      //
      //    A(i+nd*j,k+nd*l) += (w/I3b) [ DaJ_ij DaJ_kl - DaJ_il DaJ_kj ]
      //
      // | DaJ_ij  DaJ_il | = determinant of rows {i,k}, columns {j,l} from DaJ
      // | DaJ_kj  DaJ_kl |

      if (dont(HAVE_DaJ)) { Eval_DaJ(); }
      const int nd = D.Height();
      const double a = w/Get_I3b();
      for (int j = 1; j < 3; j++)
      {
         for (int l = 0; l < j; l++)
         {
            for (int i = 0; i < nd; i++)
            {
               const double aDaJ_ij = a*DaJ(i,j), aDaJ_il = a*DaJ(i,l);
               for (int k = 0; k < i; k++)
               {
                  const double A_ijkl = aDaJ_ij*DaJ(k,l) - aDaJ_il*DaJ(k,j);
                  A(i+nd*j,k+nd*l) += A_ijkl;
                  A(k+nd*l,i+nd*j) += A_ijkl;
                  A(k+nd*j,i+nd*l) -= A_ijkl;
                  A(i+nd*l,k+nd*j) -= A_ijkl;
               }
            }
         }
      }
   }
};


/// Abstract class for hyperelastic models
class HyperelasticModel
{
protected:
   ElementTransformation *Ttr;
   const DenseMatrix *Jtr;

   /// First invariant of the given 2x2 matrix @a M.
   static double Dim2Invariant1(const DenseMatrix &M);
   /// Second invariant of the given 2x2 matrix @a M.
   static double Dim2Invariant2(const DenseMatrix &M);

   /// 1st derivative of the first invariant for the given 2x2 matrix @a M.
   static void Dim2Invariant1_dM(const DenseMatrix &M, DenseMatrix &dM);
   /// 1st derivative of the second invariant for the given 2x2 matrix @a M.
   static void Dim2Invariant2_dM(const DenseMatrix &M, DenseMatrix &dM);

   /// 2nd derivative of the first invariant for the given 2x2 matrix @a M.
   static void Dim2Invariant1_dMdM(const DenseMatrix &M, int i, int j,
                                   DenseMatrix &dMdM);
   /// 2nd derivative of the second invariant for the given 2x2 matrix @a M.
   static void Dim2Invariant2_dMdM(const DenseMatrix &M, int i, int j,
                                   DenseMatrix &dMdM);

   /// First invariant of the given 3x3 matrix @a M.
   static double Dim3Invariant1(const DenseMatrix &M);
   /// Second invariant of the given 3x3 matrix @a M.
   static double Dim3Invariant2(const DenseMatrix &M);
   /// Third invariant of the given 3x3 matrix @a M.
   static double Dim3Invariant3(const DenseMatrix &M);

   /// 1st derivative of the first invariant for the given 3x3 matrix @a M.
   static void Dim3Invariant1_dM(const DenseMatrix &M, DenseMatrix &dM);
   /// 1st derivative of the second invariant for the given 3x3 matrix @a M.
   static void Dim3Invariant2_dM(const DenseMatrix &M, DenseMatrix &dM);
   /// 1st derivative of the third invariant for the given 3x3 matrix @a M.
   static void Dim3Invariant3_dM(const DenseMatrix &M, DenseMatrix &dM);

   /// 2nd derivative of the first invariant for the given 3x3 matrix @a M.
   static void Dim3Invariant1_dMdM(const DenseMatrix &M, int i, int j,
                                   DenseMatrix &dMdM);
   /// 2nd derivative of the second invariant for the given 3x3 matrix @a M.
   static void Dim3Invariant2_dMdM(const DenseMatrix &M, int i, int j,
                                   DenseMatrix &dMdM);
   /// 2nd derivative of the third invariant for the given 3x3 matrix @a M.
   static void Dim3Invariant3_dMdM(const DenseMatrix &M, int i, int j,
                                   DenseMatrix &dMdM);

public:
   HyperelasticModel() : Ttr(NULL), Jtr(NULL) { }
   virtual ~HyperelasticModel() { }

   /// A ref->target transformation that can be used to evaluate coefficients.
   /** @note It's assumed that _Ttr.SetIntPoint() is already called for
       the point of interest. */
   void SetTransformation(ElementTransformation &_Ttr) { Ttr = &_Ttr; }

   /** @brief Specify the ref->target transformation Jacobian matrix for the
       point of interest.

       Using @a Jtr is an alternative to using @a T, when one cannot define
       the target Jacobians by a single ElementTransformation for the whole
       zone, e.g., in the TMOP paradigm. */
   void SetTargetJacobian(const DenseMatrix &_Jtr) { Jtr = &_Jtr; }

   /** @brief Evaluate the strain energy density function, W = W(Jpt).
       @param[in] Jpt  Represents the target->physical transformation
                       Jacobian matrix. */
   virtual double EvalW(const DenseMatrix &Jpt) const = 0;

   /** @brief Evaluate the 1st Piola-Kirchhoff stress tensor, P = P(Jpt).
       @param[in] Jpt  Represents the target->physical transformation
                       Jacobian matrix.
       @param[out]  P  The evaluated 1st Piola-Kirchhoff stress tensor. */
   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const = 0;

   /** @brief Evaluate the derivative of the 1st Piola-Kirchhoff stress tensor
       and assemble its contribution to the local gradient matrix 'A'.
       @param[in] Jpt     Represents the target->physical transformation
                          Jacobian matrix.
       @param[in] DS      Gradient of the basis matrix (dof x dim).
       @param[in] weight  Quadrature weight coefficient for the point.
       @param[in,out]  A  Local gradient matrix where the contribution from this
                          point will be added.

       Computes weight * d(dW_dxi)_d(xj) at the current point, for all i and j,
       where x1 ... xn are the FE dofs. This function is usually defined using
       the matrix invariants and their derivatives.
   */
   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const = 0;
};


/** Inverse-harmonic hyperelastic model with a strain energy density function
    given by the formula: W(J) = (1/2) det(J) Tr((J J^t)^{-1}) where J is the
    deformation gradient. */
class InverseHarmonicModel : public HyperelasticModel
{
protected:
   mutable DenseMatrix Z, S; // dim x dim
   mutable DenseMatrix G, C; // dof x dim

public:
   virtual double EvalW(const DenseMatrix &J) const;

   virtual void EvalP(const DenseMatrix &J, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &J, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;
};


/** Neo-Hookean hyperelastic model with a strain energy density function given
    by the formula: \f$(\mu/2)(\bar{I}_1 - dim) + (K/2)(det(J)/g - 1)^2\f$ where
    J is the deformation gradient and \f$\bar{I}_1 = (det(J))^{-2/dim} Tr(J
    J^t)\f$. The parameters \f$\mu\f$ and K are the shear and bulk moduli,
    respectively, and g is a reference volumetric scaling. */
class NeoHookeanModel : public HyperelasticModel
{
protected:
   mutable double mu, K, g;
   Coefficient *c_mu, *c_K, *c_g;
   bool have_coeffs;

   mutable DenseMatrix Z;    // dim x dim
   mutable DenseMatrix G, C; // dof x dim

   inline void EvalCoeffs() const;

public:
   NeoHookeanModel(double _mu, double _K, double _g = 1.0)
      : mu(_mu), K(_K), g(_g), have_coeffs(false) { c_mu = c_K = c_g = NULL; }

   NeoHookeanModel(Coefficient &_mu, Coefficient &_K, Coefficient *_g = NULL)
      : mu(0.0), K(0.0), g(1.0), c_mu(&_mu), c_K(&_K), c_g(_g),
        have_coeffs(true) { }

   virtual double EvalW(const DenseMatrix &J) const;

   virtual void EvalP(const DenseMatrix &J, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &J, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;
};

class TMOPHyperelasticModel001 : public HyperelasticModel
{
public:
   // W = |J|^2.
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;
};

class TMOPHyperelasticModel002 : public HyperelasticModel
{
protected:
   mutable InvariantsEvaluator2D ie;

public:
   // W = 0.5|J|^2 / det(J) - 1.
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;
};

class TMOPHyperelasticModel007 : public HyperelasticModel
{
public:
   // W = |J - J^-t|^2.
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;
};

class TMOPHyperelasticModel009 : public HyperelasticModel
{
public:
   // W = det(J) * |J - J^-t|^2.
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;
};

class TMOPHyperelasticModel022 : public HyperelasticModel
{
private:
   double &tau0;

public:
   TMOPHyperelasticModel022(double &t0): tau0(t0) {}

   // W = 0.5(|J|^2 - 2det(J)) / (det(J) - tau0).
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;
};

class TMOPHyperelasticModel050 : public HyperelasticModel
{
public:
   // W = 0.5|J^t J|^2 / det(J)^2 - 1.
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;
};

class TMOPHyperelasticModel052 : public HyperelasticModel
{
private:
   double &tau0;

public:
   TMOPHyperelasticModel052(double &t0): tau0(t0) {}

   // W = 0.5(det(J) - 1)^2 / (det(J) - tau0).
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;
};

class TMOPHyperelasticModel055 : public HyperelasticModel
{
public:
   // W = (det(J) - 1)^2.
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;

};

class TMOPHyperelasticModel056 : public HyperelasticModel
{
public:
   // W = 0.5( sqrt(det(J)) - 1 / sqrt(det(J)) )^2.
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;

};

class TMOPHyperelasticModel058 : public HyperelasticModel
{
public:
   // W = |J^t J|^2 / det(J)^2 - 2|J|^2 / det(J) + 2.
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;

};

class TMOPHyperelasticModel077 : public HyperelasticModel
{
public:
   // W = 0.5(det(J) - 1 / det(J))^2.
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;

};

class TMOPHyperelasticModel211 : public HyperelasticModel
{
private:
   const double eps;

public:
   TMOPHyperelasticModel211() : eps(1e-4) { }

   // W = (det(J) - 1)^2 - det(J) + sqrt(det(J)^2 + eps).
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;
};

class TMOPHyperelasticModel301 : public HyperelasticModel
{
public:
   // W = |J| |J^-1| / 3 - 1.
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;
};

class TMOPHyperelasticModel302 : public HyperelasticModel
{
public:
   // W = |J|^2 |J^-1|^2 / 9 - 1.
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;
};

class TMOPHyperelasticModel303 : public HyperelasticModel
{
protected:
   mutable InvariantsEvaluator3D ie;

public:
   // W = |J|^2 / 3 * det(J)^(2/3) - 1.
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;
};

class TMOPHyperelasticModel315 : public HyperelasticModel
{
public:
   // W = (det(J) - 1)^2.
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;
};

class TMOPHyperelasticModel316 : public HyperelasticModel
{
public:
   // W = 0.5( sqrt(det(J)) - 1 / sqrt(det(J)) )^2.
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;
};

class TMOPHyperelasticModel321 : public HyperelasticModel
{
public:
   // W = |J - J^-t|^2.
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;
};

class TMOPHyperelasticModel352 : public HyperelasticModel
{
private:
   double &tau0;

public:
   TMOPHyperelasticModel352(double &t0): tau0(t0) {}

   // W = 0.5(det(J) - 1)^2 / (det(J) - tau0).
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;
};


/** Used to compute ref->target transformation Jacobian for different target
    options; used in class HyperelasticNLFIntegrator. */
class TargetJacobian
{
private:
   // Current nodes, initial nodes, target nodes that are
   // used in ComputeElementTargets(int), depending on target_type.
   const GridFunction *nodes, *nodes0, *tnodes;
   double avg_volume0;
   const bool serial_use;

   static void ConstructIdealJ(int geom, DenseMatrix &J);

#ifdef MFEM_USE_MPI
   MPI_Comm comm;
#endif

public:
   enum target {CURRENT, IDEAL, IDEAL_EQ_SIZE, IDEAL_INIT_SIZE, TARGET_MESH};
   const target target_type;

   // Additional scaling applied for IDEAL_EQ_SIZE. When the target is active
   // only in some part of the domain, this can be used to set relative sizes.
   double size_scale;

   TargetJacobian(target ttype)
      : nodes(NULL), nodes0(NULL), tnodes(NULL), serial_use(true),
        target_type(ttype), size_scale(1.0) { }
#ifdef MFEM_USE_MPI
   TargetJacobian(target ttype, MPI_Comm mpicomm)
      : nodes(NULL), nodes0(NULL), tnodes(NULL), serial_use(false),
        comm(mpicomm), target_type(ttype), size_scale(1.0) { }
#endif
   ~TargetJacobian() { }

   void SetNodes(const GridFunction &n)         { nodes  = &n;  }
   // Sets initial nodes and computes the average volume of the initial mesh.
   void SetInitialNodes(const GridFunction &n0);
   void SetTargetNodes(const GridFunction &tn)  { tnodes = &tn; }

   /** @brief Given an element and quadrature rule, computes ref->target
       transformation Jacobians for each quadrature point in the element. */
   void ComputeElementTargets(int e_id, const FiniteElement &fe,
                              const IntegrationRule &ir,
                              DenseTensor &Jtr) const;
};

/** Hyperelastic integrator for any given HyperelasticModel.
    Represents @f$ \int W(Jpt) dx @f$ over a target zone,
    where W is the @a model's strain energy density function, and
    Jpt is the Jacobian of the target->physical coordinates transformation. */
class HyperelasticNLFIntegrator : public NonlinearFormIntegrator
{
private:
   HyperelasticModel *model;
   const TargetJacobian *targetJ;

   // Data used for "limiting" the HyperelasticNLFIntegrator.
   bool limited;
   double eps;
   const GridFunction *nodes0;

   // Can be used to create "composite" integrators for the TMOP purposes.
   Coefficient *coeff;

   //   Jrt: the inverse of the ref->target transformation Jacobian.
   //   Jpr: the ref->physical transformation Jacobian.
   //   Jpt: the target->physical transformation Jacobians.
   //     P: represents dW_d(Jtp) (dim x dim).
   //   DSh: gradients of reference shape functions (dof x dim).
   //    DS: represents d(Jtp)_dx (dof x dim).
   // PMatI: current coordinates of the nodes (dof x dim).
   // PMat0: represents dW_dx (dof x dim).
   DenseMatrix DSh, DS, Jrt, Jpr, Jpt, P, PMatI, PMatO;

public:
   /** @param[in] m  HyperelasticModel that defines F(T).
       @param[in] tJ See HyperelasticModel::SetTargetJacobian(). */
   HyperelasticNLFIntegrator(HyperelasticModel *m, TargetJacobian *tJ = NULL)
      : model(m), targetJ(tJ),
        limited(false), eps(0.0), nodes0(NULL), coeff(NULL) { }

   const TargetJacobian *GetTargetJacobian() { return targetJ; } const

   /// Adds an extra term to the integral.
   /** The integral of interest becomes
       @f$ \int \epsilon F(T) + 0.5 (x - x_0)^2 dx@f$,
       where the second term measures the change with respect to the
       original physical positions.
       @param[in] eps_  Scaling of the @a model's contribution.
       @param[in] n0    Original mesh coordinates. */
   void SetLimited(double eps_, const GridFunction &n0)
   {
      limited = true;
      eps = eps_;
      nodes0 = &n0;
   }

   /// Sets a scaling Coefficient for the integral.
   void SetCoefficient(Coefficient &c) { coeff = &c; }

   /** @brief Computes the integral of W(Jacobian(Trt)) over a target zone
       @param[in] el     Type of FiniteElement.
       @param[in] Ttr    Represents ref->target coordinates transformation.
       @param[in] elfun  Physical coordinates of the zone. */
   virtual double GetElementEnergy(const FiniteElement &el,
                                   ElementTransformation &Ttr,
                                   const Vector &elfun);

   virtual void AssembleElementVector(const FiniteElement &el,
                                      ElementTransformation &Ttr,
                                      const Vector &elfun, Vector &elvect);

   virtual void AssembleElementGrad(const FiniteElement &el,
                                    ElementTransformation &Ttr,
                                    const Vector &elfun, DenseMatrix &elmat);

   virtual ~HyperelasticNLFIntegrator();
};

/// Interpolates the @a model's values at the nodes of @a gf.
/** Assumes that @a gf's FiniteElementSpace is initialized. */
void InterpolateHyperElasticModel(HyperelasticModel &model,
                                  const TargetJacobian &tj,
                                  const Mesh &mesh, GridFunction &gf);
}

#endif
