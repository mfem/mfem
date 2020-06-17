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

#include "tmop.hpp"
#include "linearform.hpp"
#include "pgridfunc.hpp"
#include "tmop_tools.hpp"

namespace mfem
{

// Target-matrix optimization paradigm (TMOP) mesh quality metrics.

double TMOP_Metric_001::EvalW(const DenseMatrix &Jpt) const
{
   ie.SetJacobian(Jpt.GetData());
   return ie.Get_I1();
}

void TMOP_Metric_001::EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const
{
   ie.SetJacobian(Jpt.GetData());
   P = ie.Get_dI1();
}

void TMOP_Metric_001::AssembleH(const DenseMatrix &Jpt,
                                const DenseMatrix &DS,
                                const double weight,
                                DenseMatrix &A) const
{
   ie.SetJacobian(Jpt.GetData());
   ie.SetDerivativeMatrix(DS.Height(), DS.GetData());
   ie.Assemble_ddI1(weight, A.GetData());
}

double TMOP_Metric_skew2D::EvalW(const DenseMatrix &Jpt) const
{
   MFEM_VERIFY(Jtr != NULL,
               "Requires a target Jacobian, use SetTargetJacobian().");

   DenseMatrix Jpr(2, 2);
   Mult(Jpt, *Jtr, Jpr);

   Vector col1, col2;
   Jpr.GetColumn(0, col1);
   Jpr.GetColumn(1, col2);
   double norm_prod = col1.Norml2() * col2.Norml2();
   const double cos_Jpr = (col1 * col2) / norm_prod,
                sin_Jpr = fabs(Jpr.Det()) / norm_prod;

   Jtr->GetColumn(0, col1);
   Jtr->GetColumn(1, col2);
   norm_prod = col1.Norml2() * col2.Norml2();
   const double cos_Jtr = (col1 * col2) / norm_prod,
                sin_Jtr = fabs(Jtr->Det()) / norm_prod;

   return 0.5 * (1.0 - cos_Jpr * cos_Jtr - sin_Jpr * sin_Jtr);
}

double TMOP_Metric_skew3D::EvalW(const DenseMatrix &Jpt) const
{
   MFEM_VERIFY(Jtr != NULL,
               "Requires a target Jacobian, use SetTargetJacobian().");

   DenseMatrix Jpr(3, 3);
   Mult(Jpt, *Jtr, Jpr);

   Vector col1, col2, col3;
   Jpr.GetColumn(0, col1);
   Jpr.GetColumn(1, col2);
   Jpr.GetColumn(2, col3);
   double norm_c1 = col1.Norml2(),
          norm_c2 = col2.Norml2(),
          norm_c3 = col3.Norml2();
   double cos_Jpr_12 = (col1 * col2) / (norm_c1 * norm_c2),
          cos_Jpr_13 = (col1 * col3) / (norm_c1 * norm_c3),
          cos_Jpr_23 = (col2 * col3) / (norm_c2 * norm_c3);
   double sin_Jpr_12 = std::sqrt(1.0 - cos_Jpr_12 * cos_Jpr_12),
          sin_Jpr_13 = std::sqrt(1.0 - cos_Jpr_13 * cos_Jpr_13),
          sin_Jpr_23 = std::sqrt(1.0 - cos_Jpr_23 * cos_Jpr_23);

   Jtr->GetColumn(0, col1);
   Jtr->GetColumn(1, col2);
   Jtr->GetColumn(2, col3);
   norm_c1 = col1.Norml2();
   norm_c2 = col2.Norml2(),
   norm_c3 = col3.Norml2();
   double cos_Jtr_12 = (col1 * col2) / (norm_c1 * norm_c2),
          cos_Jtr_13 = (col1 * col3) / (norm_c1 * norm_c3),
          cos_Jtr_23 = (col2 * col3) / (norm_c2 * norm_c3);
   double sin_Jtr_12 = std::sqrt(1.0 - cos_Jtr_12 * cos_Jtr_12),
          sin_Jtr_13 = std::sqrt(1.0 - cos_Jtr_13 * cos_Jtr_13),
          sin_Jtr_23 = std::sqrt(1.0 - cos_Jtr_23 * cos_Jtr_23);

   return (3.0 - cos_Jpr_12 * cos_Jtr_12 - sin_Jpr_12 * sin_Jtr_12
           - cos_Jpr_13 * cos_Jtr_13 - sin_Jpr_13 * sin_Jtr_13
           - cos_Jpr_23 * cos_Jtr_23 - sin_Jpr_23 * sin_Jtr_23) / 6.0;
}

double TMOP_Metric_aspratio2D::EvalW(const DenseMatrix &Jpt) const
{
   MFEM_VERIFY(Jtr != NULL,
               "Requires a target Jacobian, use SetTargetJacobian().");

   DenseMatrix Jpr(2, 2);
   Mult(Jpt, *Jtr, Jpr);

   Vector col1, col2;
   Jpr.GetColumn(0, col1);
   Jpr.GetColumn(1, col2);
   const double ratio_Jpr = col2.Norml2() / col1.Norml2();

   Jtr->GetColumn(0, col1);
   Jtr->GetColumn(1, col2);
   const double ratio_Jtr = col2.Norml2() / col1.Norml2();

   return 0.5 * (ratio_Jpr / ratio_Jtr + ratio_Jtr / ratio_Jpr) - 1.0;
}

double TMOP_Metric_aspratio3D::EvalW(const DenseMatrix &Jpt) const
{
   MFEM_VERIFY(Jtr != NULL,
               "Requires a target Jacobian, use SetTargetJacobian().");

   DenseMatrix Jpr(3, 3);
   Mult(Jpt, *Jtr, Jpr);

   Vector col1, col2, col3;
   Jpr.GetColumn(0, col1);
   Jpr.GetColumn(1, col2);
   Jpr.GetColumn(2, col3);
   double norm_c1 = col1.Norml2(),
          norm_c2 = col2.Norml2(),
          norm_c3 = col3.Norml2();
   double ratio_Jpr_1 = norm_c1 / std::sqrt(norm_c2 * norm_c3),
          ratio_Jpr_2 = norm_c2 / std::sqrt(norm_c1 * norm_c3),
          ratio_Jpr_3 = norm_c3 / std::sqrt(norm_c1 * norm_c2);

   Jtr->GetColumn(0, col1);
   Jtr->GetColumn(1, col2);
   Jtr->GetColumn(2, col3);
   norm_c1 = col1.Norml2();
   norm_c2 = col2.Norml2();
   norm_c3 = col3.Norml2();
   double ratio_Jtr_1 = norm_c1 / std::sqrt(norm_c2 * norm_c3),
          ratio_Jtr_2 = norm_c2 / std::sqrt(norm_c1 * norm_c3),
          ratio_Jtr_3 = norm_c3 / std::sqrt(norm_c1 * norm_c2);

   return ( 0.5 * (ratio_Jpr_1 / ratio_Jtr_1 + ratio_Jtr_1 / ratio_Jpr_1) +
            0.5 * (ratio_Jpr_2 / ratio_Jtr_2 + ratio_Jtr_2 / ratio_Jpr_2) +
            0.5 * (ratio_Jpr_3 / ratio_Jtr_3 + ratio_Jtr_3 / ratio_Jpr_3) - 3.0
          ) / 3.0;
}

// mu_14 = |T-I|^2
double TMOP_Metric_SSA2D::EvalW(const DenseMatrix &Jpt) const
{
   MFEM_VERIFY(Jtr != NULL,
               "Requires a target Jacobian, use SetTargetJacobian().");

   DenseMatrix Id(2,2);

   Id(0,0) = 1; Id(0,1) = 0;
   Id(1,0) = 0; Id(1,1) = 1;

   DenseMatrix Mat(2,2);
   Mat = Jpt;
   Mat.Add(-1,Id);
   return Mat.FNorm2();
}

// mu_85 = |T-T'|^2, where T'= |T|*I/sqrt(2)
double TMOP_Metric_SS2D::EvalW(const DenseMatrix &Jpt) const
{
   MFEM_VERIFY(Jtr != NULL,
               "Requires a target Jacobian, use SetTargetJacobian().");

   DenseMatrix Id(2,2);
   DenseMatrix Mat(2,2);
   Mat = Jpt;

   Id(0,0) = 1; Id(0,1) = 0;
   Id(1,0) = 0; Id(1,1) = 1;
   Id *= Mat.FNorm()/pow(2,0.5);

   Mat.Add(-1.,Id);
   return Mat.FNorm2();
}

double TMOP_Metric_002::EvalW(const DenseMatrix &Jpt) const
{
   ie.SetJacobian(Jpt.GetData());
   return 0.5 * ie.Get_I1b() - 1.0;
}

void TMOP_Metric_002::EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const
{
   ie.SetJacobian(Jpt.GetData());
   P.Set(0.5, ie.Get_dI1b());
}

void TMOP_Metric_002::AssembleH(const DenseMatrix &Jpt,
                                const DenseMatrix &DS,
                                const double weight,
                                DenseMatrix &A) const
{
   ie.SetJacobian(Jpt.GetData());
   ie.SetDerivativeMatrix(DS.Height(), DS.GetData());
   ie.Assemble_ddI1b(0.5*weight, A.GetData());
}

double TMOP_Metric_007::EvalW(const DenseMatrix &Jpt) const
{
   // mu_7 = |J-J^{-t}|^2 = |J|^2 + |J^{-1}|^2 - 4
   ie.SetJacobian(Jpt.GetData());
   return ie.Get_I1()*(1. + 1./ie.Get_I2()) - 4.0;
}

void TMOP_Metric_007::EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const
{
   // P = d(I1*(1 + 1/I2)) = (1 + 1/I2) dI1 - I1/I2^2 dI2
   ie.SetJacobian(Jpt.GetData());
   const double I2 = ie.Get_I2();
   Add(1. + 1./I2, ie.Get_dI1(), -ie.Get_I1()/(I2*I2), ie.Get_dI2(), P);
}

void TMOP_Metric_007::AssembleH(const DenseMatrix &Jpt,
                                const DenseMatrix &DS,
                                const double weight,
                                DenseMatrix &A) const
{
   //  P = d(I1*(1 + 1/I2))
   //    = (1 + 1/I2) dI1 - I1/I2^2 dI2
   //
   // dP = (-1/I2^2) (dI1 x dI2) + (1 + 1/I2) ddI1 -
   //      (dI2 x d(I1/I2^2)) - I1/I2^2 ddI2
   //    = (-1/I2^2) (dI1 x dI2) + (1 + 1/I2) ddI1 +
   //      (-1/I2^2) (dI2 x [dI1 - 2 I1/I2 dI2]) - I1/I2^2 ddI2
   //    = (-1/I2^2) (dI1 x dI2 + dI2 x dI1) + (1 + 1/I2) ddI1 +
   //      (2 I1/I2^3) (dI2 x dI2) - I1/I2^2 ddI2
   ie.SetJacobian(Jpt.GetData());
   ie.SetDerivativeMatrix(DS.Height(), DS.GetData());
   const double c1 = 1./ie.Get_I2();
   const double c2 = weight*c1*c1;
   const double c3 = ie.Get_I1()*c2;
   ie.Assemble_ddI1(weight*(1. + c1), A.GetData());
   ie.Assemble_ddI2(-c3, A.GetData());
   ie.Assemble_TProd(-c2, ie.Get_dI1(), ie.Get_dI2(), A.GetData());
   ie.Assemble_TProd(2*c1*c3, ie.Get_dI2(), A.GetData());
}

double TMOP_Metric_009::EvalW(const DenseMatrix &Jpt) const
{
   // mu_9 = det(J)*|J-J^{-t}|^2 = I1b * (I2b^2 + 1) - 4 * I2b
   //      = (I1 - 4)*I2b + I1b
   ie.SetJacobian(Jpt.GetData());
   return (ie.Get_I1() - 4.0)*ie.Get_I2b() + ie.Get_I1b();
}

void TMOP_Metric_009::EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const
{
   // mu_9 = (I1 - 4)*I2b + I1b
   // P = (I1 - 4)*dI2b + I2b*dI1 + dI1b
   ie.SetJacobian(Jpt.GetData());
   Add(ie.Get_I1() - 4.0, ie.Get_dI2b(), ie.Get_I2b(), ie.Get_dI1(), P);
   P += ie.Get_dI1b();
}

void TMOP_Metric_009::AssembleH(const DenseMatrix &Jpt,
                                const DenseMatrix &DS,
                                const double weight,
                                DenseMatrix &A) const
{
   // P = (I1 - 4)*dI2b + I2b*dI1 + dI1b
   // dP = dI2b x dI1 + (I1-4)*ddI2b + dI1 x dI2b + I2b*ddI1 + ddI1b
   //    = (dI1 x dI2b + dI2b x dI1) + (I1-4)*ddI2b + I2b*ddI1 + ddI1b
   ie.SetJacobian(Jpt.GetData());
   ie.SetDerivativeMatrix(DS.Height(), DS.GetData());
   ie.Assemble_TProd(weight, ie.Get_dI1(), ie.Get_dI2b(), A.GetData());
   ie.Assemble_ddI2b(weight*(ie.Get_I1()-4.0), A.GetData());
   ie.Assemble_ddI1(weight*ie.Get_I2b(), A.GetData());
   ie.Assemble_ddI1b(weight, A.GetData());
}

double TMOP_Metric_022::EvalW(const DenseMatrix &Jpt) const
{
   // mu_22 = (0.5*|J|^2 - det(J)) / (det(J) - tau0)
   //       = (0.5*I1 - I2b) / (I2b - tau0)
   ie.SetJacobian(Jpt.GetData());
   const double I2b = ie.Get_I2b();
   return (0.5*ie.Get_I1() - I2b) / (I2b - tau0);
}

void TMOP_Metric_022::EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const
{
   // mu_22 = (0.5*I1 - I2b) / (I2b - tau0)
   // P = 1/(I2b - tau0)*(0.5*dI1 - dI2b) - (0.5*I1 - I2b)/(I2b - tau0)^2*dI2b
   //   = 0.5/(I2b - tau0)*dI1 + (tau0 - 0.5*I1)/(I2b - tau0)^2*dI2b
   ie.SetJacobian(Jpt.GetData());
   const double c1 = 1.0/(ie.Get_I2b() - tau0);
   Add(c1/2, ie.Get_dI1(), (tau0 - ie.Get_I1()/2)*c1*c1, ie.Get_dI2b(), P);
}

void TMOP_Metric_022::AssembleH(const DenseMatrix &Jpt,
                                const DenseMatrix &DS,
                                const double weight,
                                DenseMatrix &A) const
{
   // P  = 0.5/(I2b - tau0)*dI1 + (tau0 - 0.5*I1)/(I2b - tau0)^2*dI2b
   // dP = -0.5/(I2b - tau0)^2*(dI1 x dI2b) + 0.5/(I2b - tau0)*ddI1
   //      + (dI2b x dz) + z*ddI2b
   //
   // z  = (tau0 - 0.5*I1)/(I2b - tau0)^2
   // dz = -0.5/(I2b - tau0)^2*dI1 - 2*(tau0 - 0.5*I1)/(I2b - tau0)^3*dI2b
   //
   // dP = -0.5/(I2b - tau0)^2*(dI1 x dI2b + dI2b x dI1)
   //      -2*z/(I2b - tau0)*(dI2b x dI2b)
   //      +0.5/(I2b - tau0)*ddI1 + z*ddI2b
   ie.SetJacobian(Jpt.GetData());
   ie.SetDerivativeMatrix(DS.Height(), DS.GetData());
   const double c1 = 1.0/(ie.Get_I2b() - tau0);
   const double c2 = weight*c1/2;
   const double c3 = c1*c2;
   const double c4 = (2*tau0 - ie.Get_I1())*c3; // weight*z
   ie.Assemble_TProd(-c3, ie.Get_dI1(), ie.Get_dI2b(), A.GetData());
   ie.Assemble_TProd(-2*c1*c4, ie.Get_dI2b(), A.GetData());
   ie.Assemble_ddI1(c2, A.GetData());
   ie.Assemble_ddI2b(c4, A.GetData());
}

double TMOP_Metric_050::EvalW(const DenseMatrix &Jpt) const
{
   // mu_50 = 0.5*|J^t J|^2/det(J)^2 - 1
   //       = 0.5*(l1^4 + l2^4)/(l1*l2)^2 - 1
   //       = 0.5*((l1/l2)^2 + (l2/l1)^2) - 1 = 0.5*(l1/l2 - l2/l1)^2
   //       = 0.5*(l1/l2 + l2/l1)^2 - 2 = 0.5*I1b^2 - 2
   ie.SetJacobian(Jpt.GetData());
   const double I1b = ie.Get_I1b();
   return 0.5*I1b*I1b - 2.0;
}

void TMOP_Metric_050::EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const
{
   // mu_50 = 0.5*I1b^2 - 2
   // P = I1b*dI1b
   ie.SetJacobian(Jpt.GetData());
   P.Set(ie.Get_I1b(), ie.Get_dI1b());
}

void TMOP_Metric_050::AssembleH(const DenseMatrix &Jpt,
                                const DenseMatrix &DS,
                                const double weight,
                                DenseMatrix &A) const
{
   // P  = I1b*dI1b
   // dP = dI1b x dI1b + I1b*ddI1b
   ie.SetJacobian(Jpt.GetData());
   ie.SetDerivativeMatrix(DS.Height(), DS.GetData());
   ie.Assemble_TProd(weight, ie.Get_dI1b(), A.GetData());
   ie.Assemble_ddI1b(weight*ie.Get_I1b(), A.GetData());
}

double TMOP_Metric_055::EvalW(const DenseMatrix &Jpt) const
{
   // mu_55 = (det(J) - 1)^2 = (I2b - 1)^2
   ie.SetJacobian(Jpt.GetData());
   const double c1 = ie.Get_I2b() - 1.0;
   return c1*c1;
}

void TMOP_Metric_055::EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const
{
   // mu_55 = (I2b - 1)^2
   // P = 2*(I2b - 1)*dI2b
   ie.SetJacobian(Jpt.GetData());
   P.Set(2*(ie.Get_I2b() - 1.0), ie.Get_dI2b());
}

void TMOP_Metric_055::AssembleH(const DenseMatrix &Jpt,
                                const DenseMatrix &DS,
                                const double weight,
                                DenseMatrix &A) const
{
   // P  = 2*(I2b - 1)*dI2b
   // dP = 2*(dI2b x dI2b) + 2*(I2b - 1)*ddI2b
   ie.SetJacobian(Jpt.GetData());
   ie.SetDerivativeMatrix(DS.Height(), DS.GetData());
   ie.Assemble_TProd(2*weight, ie.Get_dI2b(), A.GetData());
   ie.Assemble_ddI2b(2*weight*(ie.Get_I2b() - 1.0), A.GetData());
}

double TMOP_Metric_056::EvalW(const DenseMatrix &Jpt) const
{
   // mu_56 = 0.5*(I2b + 1/I2b) - 1
   ie.SetJacobian(Jpt.GetData());
   const double I2b = ie.Get_I2b();
   return 0.5*(I2b + 1.0/I2b) - 1.0;
}

void TMOP_Metric_056::EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const
{
   // mu_56 = 0.5*(I2b + 1/I2b) - 1
   // P = 0.5*(1 - 1/I2b^2)*dI2b
   ie.SetJacobian(Jpt.GetData());
   P.Set(0.5 - 0.5/ie.Get_I2(), ie.Get_dI2b());
}

void TMOP_Metric_056::AssembleH(const DenseMatrix &Jpt,
                                const DenseMatrix &DS,
                                const double weight,
                                DenseMatrix &A) const
{
   // P  = 0.5*(1 - 1/I2b^2)*dI2b
   // dP = (1/I2b^3)*(dI2b x dI2b) + (0.5 - 0.5/I2)*ddI2b
   ie.SetJacobian(Jpt.GetData());
   ie.SetDerivativeMatrix(DS.Height(), DS.GetData());
   ie.Assemble_TProd(weight/(ie.Get_I2()*ie.Get_I2b()),
                     ie.Get_dI2b(), A.GetData());
   ie.Assemble_ddI2b(weight*(0.5 - 0.5/ie.Get_I2()), A.GetData());
}

double TMOP_Metric_058::EvalW(const DenseMatrix &Jpt) const
{
   // mu_58 = I1b*(I1b - 2)
   ie.SetJacobian(Jpt.GetData());
   const double I1b = ie.Get_I1b();
   return I1b*(I1b - 1.0);
}

void TMOP_Metric_058::EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const
{
   // mu_58 = I1b*(I1b - 2)
   // P = (2*I1b - 2)*dI1b
   ie.SetJacobian(Jpt.GetData());
   P.Set(2*ie.Get_I1b() - 2.0, ie.Get_dI1b());
}

void TMOP_Metric_058::AssembleH(const DenseMatrix &Jpt,
                                const DenseMatrix &DS,
                                const double weight,
                                DenseMatrix &A) const
{
   // P  = (2*I1b - 2)*dI1b
   // dP =  2*(dI1b x dI1b) + (2*I1b - 2)*ddI1b
   ie.SetJacobian(Jpt.GetData());
   ie.SetDerivativeMatrix(DS.Height(), DS.GetData());
   ie.Assemble_TProd(2*weight, ie.Get_dI1b(), A.GetData());
   ie.Assemble_ddI1b(weight*(2*ie.Get_I1b() - 2.0), A.GetData());
}

double TMOP_Metric_077::EvalW(const DenseMatrix &Jpt) const
{
   ie.SetJacobian(Jpt.GetData());
   const double I2 = ie.Get_I2b();
   return  0.5*(I2*I2 + 1./(I2*I2) - 2.);
}

void TMOP_Metric_077::EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const
{
   // Using I2b^2 = I2.
   // dmu77_dJ = 1/2 (1 - 1/I2^2) dI2_dJ.
   ie.SetJacobian(Jpt.GetData());
   const double I2 = ie.Get_I2();
   P.Set(0.5 * (1.0 - 1.0 / (I2 * I2)), ie.Get_dI2());
}

void TMOP_Metric_077::AssembleH(const DenseMatrix &Jpt,
                                const DenseMatrix &DS,
                                const double weight,
                                DenseMatrix &A) const
{
   ie.SetJacobian(Jpt.GetData());
   ie.SetDerivativeMatrix(DS.Height(), DS.GetData());
   const double I2 = ie.Get_I2(), I2inv_sq = 1.0 / (I2 * I2);
   ie.Assemble_ddI2(weight*0.5*(1.0 - I2inv_sq), A.GetData());
   ie.Assemble_TProd(weight * I2inv_sq / I2, ie.Get_dI2(), A.GetData());
}

double TMOP_Metric_211::EvalW(const DenseMatrix &Jpt) const
{
   // mu_211 = (det(J) - 1)^2 - det(J) + (det(J)^2 + eps)^{1/2}
   //        = (I2b - 1)^2 - I2b + sqrt(I2b^2 + eps)
   ie.SetJacobian(Jpt.GetData());
   const double I2b = ie.Get_I2b();
   return (I2b - 1.0)*(I2b - 1.0) - I2b + std::sqrt(I2b*I2b + eps);
}

void TMOP_Metric_211::EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const
{
   MFEM_ABORT("Metric not implemented yet. Use metric mu_55 instead.");
}

void TMOP_Metric_211::AssembleH(const DenseMatrix &Jpt,
                                const DenseMatrix &DS,
                                const double weight,
                                DenseMatrix &A) const
{
   MFEM_ABORT("Metric not implemented yet. Use metric mu_55 instead.");
}

double TMOP_Metric_252::EvalW(const DenseMatrix &Jpt) const
{
   // mu_252 = 0.5*(det(J) - 1)^2 / (det(J) - tau0).
   ie.SetJacobian(Jpt.GetData());
   const double I2b = ie.Get_I2b();
   return 0.5*(I2b - 1.0)*(I2b - 1.0)/(I2b - tau0);
}

void TMOP_Metric_252::EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const
{
   // mu_252 = 0.5*(det(J) - 1)^2 / (det(J) - tau0)
   // P = (c - 0.5*c*c ) * dI2b
   //
   // c = (I2b - 1)/(I2b - tau0), see TMOP_Metric_352 for details
   ie.SetJacobian(Jpt.GetData());
   const double I2b = ie.Get_I2b();
   const double c = (I2b - 1.0)/(I2b - tau0);
   P.Set(c - 0.5*c*c, ie.Get_dI2b());
}

void TMOP_Metric_252::AssembleH(const DenseMatrix &Jpt,
                                const DenseMatrix &DS,
                                const double weight,
                                DenseMatrix &A) const
{
   // c = (I2b - 1)/(I2b - tau0), see TMOP_Metric_352 for details
   //
   // P  = (c - 0.5*c*c ) * dI2b
   // dP = (1 - c)^2/(I2b - tau0)*(dI2b x dI2b) + (c - 0.5*c*c)*ddI2b
   ie.SetJacobian(Jpt.GetData());
   ie.SetDerivativeMatrix(DS.Height(), DS.GetData());
   const double I2b = ie.Get_I2b();
   const double c0 = 1.0/(I2b - tau0);
   const double c = c0*(I2b - 1.0);
   ie.Assemble_TProd(weight*c0*(1.0 - c)*(1.0 - c), ie.Get_dI2b(), A.GetData());
   ie.Assemble_ddI2b(weight*(c - 0.5*c*c), A.GetData());
}

double TMOP_Metric_301::EvalW(const DenseMatrix &Jpt) const
{
   ie.SetJacobian(Jpt.GetData());
   return std::sqrt(ie.Get_I1b()*ie.Get_I2b())/3. - 1.;
}

void TMOP_Metric_301::EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const
{
   //  W = (1/3)*sqrt(I1b*I2b) - 1
   // dW = (1/6)/sqrt(I1b*I2b)*[I2b*dI1b + I1b*dI2b]
   ie.SetJacobian(Jpt.GetData());
   const double a = 1./(6.*std::sqrt(ie.Get_I1b()*ie.Get_I2b()));
   Add(a*ie.Get_I2b(), ie.Get_dI1b(), a*ie.Get_I1b(), ie.Get_dI2b(), P);
}

void TMOP_Metric_301::AssembleH(const DenseMatrix &Jpt,
                                const DenseMatrix &DS,
                                const double weight,
                                DenseMatrix &A) const
{
   //  dW = (1/6)/sqrt(I1b*I2b)*[I2b*dI1b + I1b*dI2b]
   //  dW = (1/6)*[z2*dI1b + z1*dI2b], z1 = sqrt(I1b/I2b), z2 = sqrt(I2b/I1b)
   // ddW = (1/6)*[dI1b x dz2 + z2*ddI1b + dI2b x dz1 + z1*ddI2b]
   //
   // dz1 = (1/2)*sqrt(I2b/I1b) [ (1/I2b)*dI1b + (I1b/(I2b*I2b))*dI2b ]
   //     = (1/2)/sqrt(I1b*I2b) [ dI1b + (I1b/I2b)*dI2b ]
   // dz2 = (1/2)/sqrt(I1b*I2b) [ (I2b/I1b)*dI1b + dI2b ]
   //
   // dI1b x dz2 + dI2b x dz1 =
   //    (1/2)/sqrt(I1b*I2b) dI1b x [ (I2b/I1b)*dI1b + dI2b ] +
   //    (1/2)/sqrt(I1b*I2b) dI2b x [ dI1b + (I1b/I2b)*dI2b ] =
   //    (1/2)/sqrt(I1b*I2b) [sqrt(I2b/I1b)*dI1b + sqrt(I1b/I2b)*dI2b] x
   //                        [sqrt(I2b/I1b)*dI1b + sqrt(I1b/I2b)*dI2b] =
   //    (1/2)/sqrt(I1b*I2b) [ 6*dW x 6*dW ] =
   //    (1/2)*(I1b*I2b)^{-3/2} (I2b*dI1b + I1b*dI2b) x (I2b*dI1b + I1b*dI2b)
   //
   // z1 = I1b/sqrt(I1b*I2b), z2 = I2b/sqrt(I1b*I2b)

   ie.SetJacobian(Jpt.GetData());
   ie.SetDerivativeMatrix(DS.Height(), DS.GetData());
   double d_I1b_I2b_data[9];
   DenseMatrix d_I1b_I2b(d_I1b_I2b_data, 3, 3);
   Add(ie.Get_I2b(), ie.Get_dI1b(), ie.Get_I1b(), ie.Get_dI2b(), d_I1b_I2b);
   const double I1b_I2b = ie.Get_I1b()*ie.Get_I2b();
   const double a = weight/(6*std::sqrt(I1b_I2b));
   ie.Assemble_ddI1b(a*ie.Get_I2b(), A.GetData());
   ie.Assemble_ddI2b(a*ie.Get_I1b(), A.GetData());
   ie.Assemble_TProd(a/(2*I1b_I2b), d_I1b_I2b_data, A.GetData());
}

double TMOP_Metric_302::EvalW(const DenseMatrix &Jpt) const
{
   // mu_2 = |J|^2 |J^{-1}|^2 / 9 - 1
   //      = (l1^2 + l2^2 + l3^3)*(l1^{-2} + l2^{-2} + l3^{-2}) / 9 - 1
   //      = I1*(l2^2*l3^2 + l1^2*l3^2 + l1^2*l2^2)/l1^2/l2^2/l3^2/9 - 1
   //      = I1*I2/det(J)^2/9 - 1 = I1b*I2b/9-1
   ie.SetJacobian(Jpt.GetData());
   return ie.Get_I1b()*ie.Get_I2b()/9. - 1.;
}

void TMOP_Metric_302::EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const
{
   // mu_2 = I1b*I2b/9-1
   // P = (I1b/9)*dI2b + (I2b/9)*dI1b
   ie.SetJacobian(Jpt.GetData());
   Add(ie.Get_I1b()/9, ie.Get_dI2b(), ie.Get_I2b()/9, ie.Get_dI1b(), P);
}

void TMOP_Metric_302::AssembleH(const DenseMatrix &Jpt,
                                const DenseMatrix &DS,
                                const double weight,
                                DenseMatrix &A) const
{
   // P  = (I1b/9)*dI2b + (I2b/9)*dI1b
   // dP = (dI2b x dI1b)/9 + (I1b/9)*ddI2b + (dI1b x dI2b)/9 + (I2b/9)*ddI1b
   //    = (dI2b x dI1b + dI1b x dI2b)/9 + (I1b/9)*ddI2b + (I2b/9)*ddI1b
   ie.SetJacobian(Jpt.GetData());
   ie.SetDerivativeMatrix(DS.Height(), DS.GetData());
   const double c1 = weight/9;
   ie.Assemble_TProd(c1, ie.Get_dI1b(), ie.Get_dI2b(), A.GetData());
   ie.Assemble_ddI2b(c1*ie.Get_I1b(), A.GetData());
   ie.Assemble_ddI1b(c1*ie.Get_I2b(), A.GetData());
}

double TMOP_Metric_303::EvalW(const DenseMatrix &Jpt) const
{
   ie.SetJacobian(Jpt.GetData());
   return ie.Get_I1b()/3.0 - 1.0;
}

void TMOP_Metric_303::EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const
{
   ie.SetJacobian(Jpt.GetData());
   P.Set(1./3., ie.Get_dI1b());
}

void TMOP_Metric_303::AssembleH(const DenseMatrix &Jpt,
                                const DenseMatrix &DS,
                                const double weight,
                                DenseMatrix &A) const
{
   ie.SetJacobian(Jpt.GetData());
   ie.SetDerivativeMatrix(DS.Height(), DS.GetData());
   ie.Assemble_ddI1b(weight/3., A.GetData());
}

double TMOP_Metric_315::EvalW(const DenseMatrix &Jpt) const
{
   // mu_315 = mu_15_3D = (det(J) - 1)^2
   ie.SetJacobian(Jpt.GetData());
   const double c1 = ie.Get_I3b() - 1.0;
   return c1*c1;
}

void TMOP_Metric_315::EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const
{
   // mu_315 = (I3b - 1)^2
   // P = 2*(I3b - 1)*dI3b
   ie.SetJacobian(Jpt.GetData());
   P.Set(2*(ie.Get_I3b() - 1.0), ie.Get_dI3b());
}

void TMOP_Metric_315::AssembleH(const DenseMatrix &Jpt,
                                const DenseMatrix &DS,
                                const double weight,
                                DenseMatrix &A) const
{
   // P  = 2*(I3b - 1)*dI3b
   // dP = 2*(dI3b x dI3b) + 2*(I3b - 1)*ddI3b
   ie.SetJacobian(Jpt.GetData());
   ie.SetDerivativeMatrix(DS.Height(), DS.GetData());
   ie.Assemble_TProd(2*weight, ie.Get_dI3b(), A.GetData());
   ie.Assemble_ddI3b(2*weight*(ie.Get_I3b() - 1.0), A.GetData());
}

double TMOP_Metric_316::EvalW(const DenseMatrix &Jpt) const
{
   // mu_316 = mu_16_3D = 0.5*(I3b + 1/I3b) - 1
   ie.SetJacobian(Jpt.GetData());
   const double I3b = ie.Get_I3b();
   return 0.5*(I3b + 1.0/I3b) - 1.0;
}

void TMOP_Metric_316::EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const
{
   // mu_316 = mu_16_3D = 0.5*(I3b + 1/I3b) - 1
   // P = 0.5*(1 - 1/I3b^2)*dI3b = (0.5 - 0.5/I3)*dI3b
   ie.SetJacobian(Jpt.GetData());
   P.Set(0.5 - 0.5/ie.Get_I3(), ie.Get_dI3b());
}

void TMOP_Metric_316::AssembleH(const DenseMatrix &Jpt,
                                const DenseMatrix &DS,
                                const double weight,
                                DenseMatrix &A) const
{
   // P  = 0.5*(1 - 1/I3b^2)*dI3b = (0.5 - 0.5/I3)*dI3b
   // dP = (1/I3b^3)*(dI3b x dI3b) + (0.5 - 0.5/I3)*ddI3b
   ie.SetJacobian(Jpt.GetData());
   ie.SetDerivativeMatrix(DS.Height(), DS.GetData());
   ie.Assemble_TProd(weight/(ie.Get_I3()*ie.Get_I3b()),
                     ie.Get_dI3b(), A.GetData());
   ie.Assemble_ddI3b(weight*(0.5 - 0.5/ie.Get_I3()), A.GetData());
}

double TMOP_Metric_321::EvalW(const DenseMatrix &Jpt) const
{
   // mu_321 = mu_21_3D = |J - J^{-t}|^2
   //        = |J|^2 + |J^{-1}|^2 - 6
   //        = |J|^2 + (l1^{-2} + l2^{-2} + l3^{-2}) - 6
   //        = |J|^2 + (l2^2*l3^2 + l1^2*l3^2 + l1^2*l2^2)/det(J)^2 - 6
   //        = I1 + I2/I3b^2 - 6 = I1 + I2/I3 - 6
   ie.SetJacobian(Jpt.GetData());
   return ie.Get_I1() + ie.Get_I2()/ie.Get_I3() - 6.0;
}

void TMOP_Metric_321::EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const
{
   // mu_321 = I1 + I2/I3b^2 - 6 = I1 + I2/I3 - 6
   // P = dI1 + (1/I3)*dI2 - (2*I2/I3b^3)*dI3b
   ie.SetJacobian(Jpt.GetData());
   const double I3 = ie.Get_I3();
   Add(1.0/I3, ie.Get_dI2(),
       -2*ie.Get_I2()/(I3*ie.Get_I3b()), ie.Get_dI3b(), P);
   P += ie.Get_dI1();
}

void TMOP_Metric_321::AssembleH(const DenseMatrix &Jpt,
                                const DenseMatrix &DS,
                                const double weight,
                                DenseMatrix &A) const
{
   // P  = dI1 + (1/I3)*dI2 - (2*I2/I3b^3)*dI3b
   // dP = ddI1 + (-2/I3b^3)*(dI2 x dI3b) + (1/I3)*ddI2 + (dI3b x dz) + z*ddI3b
   //
   // z  = -2*I2/I3b^3
   // dz = (-2/I3b^3)*dI2 + (2*I2)*(3/I3b^4)*dI3b
   //
   // dP = ddI1 + (-2/I3b^3)*(dI2 x dI3b + dI3b x dI2) + (1/I3)*ddI2
   //      + (6*I2/I3b^4)*(dI3b x dI3b) + (-2*I2/I3b^3)*ddI3b
   ie.SetJacobian(Jpt.GetData());
   ie.SetDerivativeMatrix(DS.Height(), DS.GetData());
   const double c0 = 1.0/ie.Get_I3b();
   const double c1 = weight*c0*c0;
   const double c2 = -2*c0*c1;
   const double c3 = c2*ie.Get_I2();
   ie.Assemble_ddI1(weight, A.GetData());
   ie.Assemble_ddI2(c1, A.GetData());
   ie.Assemble_ddI3b(c3, A.GetData());
   ie.Assemble_TProd(c2, ie.Get_dI2(), ie.Get_dI3b(), A.GetData());
   ie.Assemble_TProd(-3*c0*c3, ie.Get_dI3b(), A.GetData());
}

double TMOP_Metric_352::EvalW(const DenseMatrix &Jpt) const
{
   // mu_352 = 0.5*(det(J) - 1)^2 / (det(J) - tau0)
   ie.SetJacobian(Jpt.GetData());
   const double I3b = ie.Get_I3b();
   return 0.5*(I3b - 1.0)*(I3b - 1.0)/(I3b - tau0);
}

void TMOP_Metric_352::EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const
{
   // mu_352 = 0.5*(det(J) - 1)^2 / (det(J) - tau0)
   // P = (I3b - 1)/(I3b - tau0)*dI3b + 0.5*(I3b - 1)^2*(-1/(I3b - tau0)^2)*dI3b
   //   = [ (I3b - 1)/(I3b - tau0) - 0.5*(I3b - 1)^2/(I3b - tau0)^2 ] * dI3b
   //   = (c - 0.5*c*c) * dI3b
   ie.SetJacobian(Jpt.GetData());
   const double I3b = ie.Get_I3b();
   const double c = (I3b - 1.0)/(I3b - tau0);
   P.Set(c - 0.5*c*c, ie.Get_dI3b());
}

void TMOP_Metric_352::AssembleH(const DenseMatrix &Jpt,
                                const DenseMatrix &DS,
                                const double weight,
                                DenseMatrix &A) const
{
   // c = (I3b - 1)/(I3b - tau0)
   //
   // P  = (c - 0.5*c*c) * dI3b
   // dP = (1 - c)*(dI3b x dc) + (c - 0.5*c*c)*ddI3b
   //
   // dc = 1/(I3b - tau0)*dI3b - (I3b - 1)/(I3b - tau)^2*dI3b =
   //    = (1 - c)/(I3b - tau0)*dI3b
   //
   // dP = (1 - c)^2/(I3b - tau0)*(dI3b x dI3b) + (c - 0.5*c*c)*ddI3b
   ie.SetJacobian(Jpt.GetData());
   ie.SetDerivativeMatrix(DS.Height(), DS.GetData());
   const double I3b = ie.Get_I3b();
   const double c0 = 1.0/(I3b - tau0);
   const double c = c0*(I3b - 1.0);
   ie.Assemble_TProd(weight*c0*(1.0 - c)*(1.0 - c), ie.Get_dI3b(), A.GetData());
   ie.Assemble_ddI3b(weight*(c - 0.5*c*c), A.GetData());
}


void TargetConstructor::ComputeAvgVolume() const
{
   MFEM_VERIFY(nodes, "Nodes are not given!");
   MFEM_ASSERT(avg_volume == 0.0, "The average volume is already computed!");

   Mesh *mesh = nodes->FESpace()->GetMesh();
   const int NE = mesh->GetNE();
   IsoparametricTransformation Tr;
   double volume = 0.0;

   for (int i = 0; i < NE; i++)
   {
      mesh->GetElementTransformation(i, *nodes, &Tr);
      const IntegrationRule &ir =
         IntRules.Get(mesh->GetElementBaseGeometry(i), Tr.OrderJ());
      for (int j = 0; j < ir.GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir.IntPoint(j);
         Tr.SetIntPoint(&ip);
         volume += ip.weight * Tr.Weight();
      }
   }

   NCMesh *ncmesh = mesh->ncmesh;
   if (Parallel() == false)
   {
      avg_volume = (ncmesh == NULL) ?
                   volume / NE : volume / ncmesh->GetNumRootElements();

   }
#ifdef MFEM_USE_MPI
   else
   {
      double area_NE[4];
      area_NE[0] = volume; area_NE[1] = NE;
      MPI_Allreduce(area_NE, area_NE + 2, 2, MPI_DOUBLE, MPI_SUM, comm);
      avg_volume = (ncmesh == NULL) ?
                   area_NE[2] / area_NE[3] : area_NE[2] / ncmesh->GetNumRootElements();
   }
#endif
}

bool TargetConstructor::ContainsVolumeInfo() const
{
   switch (target_type)
   {
      case IDEAL_SHAPE_UNIT_SIZE: return false;
      case IDEAL_SHAPE_EQUAL_SIZE:
      case IDEAL_SHAPE_GIVEN_SIZE:
      case GIVEN_SHAPE_AND_SIZE:
      case GIVEN_FULL: return true;
      default: MFEM_ABORT("TargetType not added to ContainsVolumeInfo.");
   }
   return false;
}

void TargetConstructor::ComputeElementTargets(int e_id, const FiniteElement &fe,
                                              const IntegrationRule &ir,
                                              const Vector &elfun,
                                              DenseTensor &Jtr) const
{
   MFEM_ASSERT(target_type == IDEAL_SHAPE_UNIT_SIZE || nodes != NULL, "");

   const FiniteElement *nfe = (target_type != IDEAL_SHAPE_UNIT_SIZE) ?
                              nodes->FESpace()->GetFE(e_id) : NULL;
   const DenseMatrix &Wideal =
      Geometries.GetGeomToPerfGeomJac(fe.GetGeomType());
   MFEM_ASSERT(Wideal.Height() == Jtr.SizeI(), "");
   MFEM_ASSERT(Wideal.Width() == Jtr.SizeJ(), "");

   switch (target_type)
   {
      case IDEAL_SHAPE_UNIT_SIZE:
      {
         for (int i = 0; i < ir.GetNPoints(); i++) { Jtr(i) = Wideal; }
         break;
      }
      case IDEAL_SHAPE_EQUAL_SIZE:
      {
         if (avg_volume == 0.0) { ComputeAvgVolume(); }
         DenseMatrix W(Wideal.Height());

         NCMesh *ncmesh = nodes->FESpace()->GetMesh()->ncmesh;
         double el_volume = avg_volume;
         if (ncmesh)
         {
            el_volume = avg_volume / ncmesh->GetElementSizeReduction(e_id);
         }

         W.Set(std::pow(volume_scale * el_volume / Wideal.Det(),
                        1./W.Height()), Wideal);
         for (int i = 0; i < ir.GetNPoints(); i++) { Jtr(i) = W; }
         break;
      }
      case IDEAL_SHAPE_GIVEN_SIZE:
      case GIVEN_SHAPE_AND_SIZE:
      {
         const int dim = nfe->GetDim(), dof = nfe->GetDof();
         MFEM_ASSERT(dim == nodes->FESpace()->GetVDim(), "");
         DenseMatrix dshape(dof, dim), pos(dof, dim);
         Array<int> xdofs(dof * dim);
         Vector posV(pos.Data(), dof * dim);
         double detW;

         // always initialize detW to suppress a warning:
         detW = (target_type == IDEAL_SHAPE_GIVEN_SIZE) ? Wideal.Det() : 0.0;
         nodes->FESpace()->GetElementVDofs(e_id, xdofs);
         nodes->GetSubVector(xdofs, posV);
         for (int i = 0; i < ir.GetNPoints(); i++)
         {
            nfe->CalcDShape(ir.IntPoint(i), dshape);
            MultAtB(pos, dshape, Jtr(i));
            if (target_type == IDEAL_SHAPE_GIVEN_SIZE)
            {
               const double det = Jtr(i).Det();
               MFEM_VERIFY(det > 0.0, "The given mesh is inverted!");
               Jtr(i).Set(std::pow(det / detW, 1./dim), Wideal);
            }
         }
         break;
      }
      default:
         MFEM_ABORT("invalid target type!");
   }
}

void AnalyticAdaptTC::SetAnalyticTargetSpec(Coefficient *sspec,
                                            VectorCoefficient *vspec,
                                            MatrixCoefficient *mspec)
{
   scalar_tspec = sspec;
   vector_tspec = vspec;
   matrix_tspec = mspec;
}

void AnalyticAdaptTC::ComputeElementTargets(int e_id, const FiniteElement &fe,
                                            const IntegrationRule &ir,
                                            const Vector &elfun,
                                            DenseTensor &Jtr) const
{
   DenseMatrix point_mat;
   point_mat.UseExternalData(elfun.GetData(), fe.GetDof(), fe.GetDim());

   switch (target_type)
   {
      case GIVEN_FULL:
      {
         MFEM_VERIFY(matrix_tspec != NULL,
                     "Target type GIVEN_FULL requires a MatrixCoefficient.");

         IsoparametricTransformation Tpr;
         Tpr.SetFE(&fe);
         Tpr.ElementNo = e_id;
         Tpr.ElementType = ElementTransformation::ELEMENT;
         Tpr.GetPointMat().Transpose(point_mat);

         for (int i = 0; i < ir.GetNPoints(); i++)
         {
            const IntegrationPoint &ip = ir.IntPoint(i);
            Tpr.SetIntPoint(&ip);
            matrix_tspec->Eval(Jtr(i), Tpr, ip);
         }
         break;
      }
      default:
         MFEM_ABORT("Incompatible target type for analytic adaptation!");
   }
}

#ifdef MFEM_USE_MPI
void DiscreteAdaptTC::FinalizeParDiscreteTargetSpec(const ParGridFunction
                                                    &tspec_)
{
   MFEM_VERIFY(adapt_eval, "SetAdaptivityEvaluator() has not been called!")
   MFEM_VERIFY(ncomp > 0, "No target specifications have been set!");

   ParFiniteElementSpace *ptspec_fes = tspec_.ParFESpace();

   adapt_eval->SetParMetaInfo(*ptspec_fes->GetParMesh(),
                              *ptspec_fes->FEColl(), ncomp);
   adapt_eval->SetInitialField(*tspec_fes->GetMesh()->GetNodes(), tspec);

   tspec_sav = tspec;

   delete tspec_fesv;
   tspec_fesv = new FiniteElementSpace(tspec_fes->GetMesh(),
                                       tspec_fes->FEColl(), ncomp);
}

void DiscreteAdaptTC::SetTspecAtIndex(int idx, const ParGridFunction &tspec_)
{
   const int vdim     = tspec_.FESpace()->GetVDim(),
             dof_cnt  = tspec_.Size()/vdim;
   for (int i = 0; i < dof_cnt*vdim; i++)
   {
      tspec(i+idx*dof_cnt) = tspec_(i);
   }

   FinalizeParDiscreteTargetSpec(tspec_);
}

void DiscreteAdaptTC::SetParDiscreteTargetSize(const ParGridFunction &tspec_)
{
   if (sizeidx > -1) { SetTspecAtIndex(sizeidx, tspec_); return; }
   sizeidx = ncomp;
   SetDiscreteTargetBase(tspec_);
   FinalizeParDiscreteTargetSpec(tspec_);
}

void DiscreteAdaptTC::SetParDiscreteTargetSkew(const ParGridFunction &tspec_)
{
   if (skewidx > -1) { SetTspecAtIndex(skewidx, tspec_); return; }
   skewidx = ncomp;
   SetDiscreteTargetBase(tspec_);
   FinalizeParDiscreteTargetSpec(tspec_);
}

void DiscreteAdaptTC::SetParDiscreteTargetAspectRatio(const ParGridFunction
                                                      &tspec_)
{
   if (aspectratioidx > -1) { SetTspecAtIndex(aspectratioidx, tspec_); return; }
   aspectratioidx = ncomp;
   SetDiscreteTargetBase(tspec_);
   FinalizeParDiscreteTargetSpec(tspec_);
}

void DiscreteAdaptTC::SetParDiscreteTargetOrientation(const ParGridFunction
                                                      &tspec_)
{
   if (orientationidx > -1) { SetTspecAtIndex(orientationidx, tspec_); return; }
   orientationidx = ncomp;
   SetDiscreteTargetBase(tspec_);
   FinalizeParDiscreteTargetSpec(tspec_);
}

void DiscreteAdaptTC::SetParDiscreteTargetSpec(const ParGridFunction &tspec_)
{
   SetParDiscreteTargetSize(tspec_);
   FinalizeParDiscreteTargetSpec(tspec_);
}
#endif

void DiscreteAdaptTC::SetDiscreteTargetBase(const GridFunction &tspec_)
{
   const int vdim     = tspec_.FESpace()->GetVDim(),
             dof_cnt  = tspec_.Size()/vdim;

   ncomp += vdim;

   delete tspec_fes;
   tspec_fes = new FiniteElementSpace(tspec_.FESpace()->GetMesh(),
                                      tspec_.FESpace()->FEColl(), 1);

   // need to append data to tspec
   // make a copy of tspec->tspec_temp, increase its size, and
   // copy data from tspec_temp -> tspec, then add new entries
   Vector tspec_temp = tspec;
   tspec.SetSize(ncomp*dof_cnt);

   for (int i = 0; i < tspec_temp.Size(); i++)
   {
      tspec(i) = tspec_temp(i);
   }

   for (int i = 0; i < dof_cnt*vdim; i++)
   {
      tspec(i+(ncomp-vdim)*dof_cnt) = tspec_(i);
   }
}

void DiscreteAdaptTC::SetTspecAtIndex(int idx, const GridFunction &tspec_)
{
   const int vdim     = tspec_.FESpace()->GetVDim(),
             dof_cnt  = tspec_.Size()/vdim;
   for (int i = 0; i < dof_cnt*vdim; i++)
   {
      tspec(i+idx*dof_cnt) = tspec_(i);
   }

   FinalizeSerialDiscreteTargetSpec();
}

void DiscreteAdaptTC::SetSerialDiscreteTargetSize(const GridFunction &tspec_)
{

   if (sizeidx > -1) { SetTspecAtIndex(sizeidx, tspec_); return; }
   sizeidx = ncomp;
   SetDiscreteTargetBase(tspec_);
   FinalizeSerialDiscreteTargetSpec();
}

void DiscreteAdaptTC::SetSerialDiscreteTargetSkew(const GridFunction &tspec_)
{
   if (skewidx > -1) { SetTspecAtIndex(skewidx, tspec_); return; }
   skewidx = ncomp;
   SetDiscreteTargetBase(tspec_);
   FinalizeSerialDiscreteTargetSpec();
}

void DiscreteAdaptTC::SetSerialDiscreteTargetAspectRatio(
   const GridFunction &tspec_)
{
   if (aspectratioidx > -1) { SetTspecAtIndex(aspectratioidx, tspec_); return; }
   aspectratioidx = ncomp;
   SetDiscreteTargetBase(tspec_);
   FinalizeSerialDiscreteTargetSpec();
}

void DiscreteAdaptTC::SetSerialDiscreteTargetOrientation(
   const GridFunction &tspec_)
{
   if (orientationidx > -1) { SetTspecAtIndex(orientationidx, tspec_); return; }
   orientationidx = ncomp;
   SetDiscreteTargetBase(tspec_);
   FinalizeSerialDiscreteTargetSpec();
}

void DiscreteAdaptTC::FinalizeSerialDiscreteTargetSpec()
{
   MFEM_VERIFY(adapt_eval, "SetAdaptivityEvaluator() has not been called!")
   MFEM_VERIFY(ncomp > 0, "No target specifications have been set!");

   adapt_eval->SetSerialMetaInfo(*tspec_fes->GetMesh(),
                                 *tspec_fes->FEColl(), ncomp);
   adapt_eval->SetInitialField(*tspec_fes->GetMesh()->GetNodes(), tspec);

   tspec_sav = tspec;

   delete tspec_fesv;
   tspec_fesv = new FiniteElementSpace(tspec_fes->GetMesh(),
                                       tspec_fes->FEColl(), ncomp);
}

void DiscreteAdaptTC::SetSerialDiscreteTargetSpec(const GridFunction &tspec_)
{
   SetSerialDiscreteTargetSize(tspec_);
   FinalizeSerialDiscreteTargetSpec();
}


void DiscreteAdaptTC::UpdateTargetSpecification(const Vector &new_x,
                                                bool use_flag)
{
   if (use_flag && good_tspec) { return; }

   MFEM_VERIFY(tspec.Size() > 0, "Target specification is not set!");
   adapt_eval->ComputeAtNewPosition(new_x, tspec);
   tspec_sav = tspec;

   good_tspec = use_flag;
}

void DiscreteAdaptTC::UpdateTargetSpecification(Vector &new_x,
                                                Vector &IntData)
{
   adapt_eval->ComputeAtNewPosition(new_x, IntData);
}

void DiscreteAdaptTC::UpdateTargetSpecificationAtNode(const FiniteElement &el,
                                                      ElementTransformation &T,
                                                      int dofidx, int dir,
                                                      const Vector &IntData)
{
   MFEM_VERIFY(tspec.Size() > 0, "Target specification is not set!");

   Array<int> dofs;
   tspec_fes->GetElementDofs(T.ElementNo, dofs);
   const int cnt = tspec.Size()/ncomp; //dofs per scalar-field

   for (int i = 0; i < ncomp; i++)
   {
      tspec(dofs[dofidx]+i*cnt) = IntData(dofs[dofidx] + i*cnt + dir*cnt*ncomp);
   }
}

void DiscreteAdaptTC::RestoreTargetSpecificationAtNode(ElementTransformation &T,
                                                       int dofidx)
{
   MFEM_VERIFY(tspec.Size() > 0, "Target specification is not set!");

   Array<int> dofs;
   tspec_fes->GetElementDofs(T.ElementNo, dofs);
   const int cnt = tspec.Size()/ncomp;
   for (int i = 0; i < ncomp; i++)
   {
      tspec(dofs[dofidx] + i*cnt) = tspec_sav(dofs[dofidx] + i*cnt);
   }
}

void DiscreteAdaptTC::ComputeElementTargets(int e_id, const FiniteElement &fe,
                                            const IntegrationRule &ir,
                                            const Vector &elfun,
                                            DenseTensor &Jtr) const
{
   MFEM_VERIFY(tspec_fesv, "No target specifications have been set.");

   switch (target_type)
   {
      case IDEAL_SHAPE_GIVEN_SIZE:
      case GIVEN_SHAPE_AND_SIZE:
      {
         const DenseMatrix &Wideal =
            Geometries.GetGeomToPerfGeomJac(fe.GetGeomType());
         const int dim = Wideal.Height(),
                   ndofs = tspec_fes->GetFE(0)->GetDof(),
                   ntspec_dofs = ndofs*ncomp;

         Vector shape(ndofs), tspec_vals(ntspec_dofs), par_vals,
                par_vals_c1, par_vals_c2, par_vals_c3;

         Array<int> dofs;
         DenseMatrix D_rho(dim), Q_phi(dim), R_theta(dim);
         tspec_fesv->GetElementVDofs(e_id, dofs);
         tspec.GetSubVector(dofs, tspec_vals);

         for (int i = 0; i < ir.GetNPoints(); i++)
         {
            const IntegrationPoint &ip = ir.IntPoint(i);
            tspec_fes->GetFE(e_id)->CalcShape(ip, shape);
            Jtr(i) = Wideal; //Initialize to identity

            if (sizeidx != -1) //Set size
            {
               par_vals.SetDataAndSize(tspec_vals.GetData()+sizeidx*ndofs, ndofs);
               const double min_size = par_vals.Min();
               MFEM_VERIFY(min_size > 0.0,
                           "Non-positive size propagated in the target definition.");
               const double size = std::max(shape * par_vals, min_size);
               Jtr(i).Set(std::pow(size, 1.0/dim), Jtr(i));
            } //Done size

            if (target_type == IDEAL_SHAPE_GIVEN_SIZE) { continue; }

            if (aspectratioidx != -1) //Set aspect ratio
            {
               if (dim == 2)
               {
                  par_vals.SetDataAndSize(tspec_vals.GetData()+
                                          aspectratioidx*ndofs, ndofs);

                  const double aspectratio = shape * par_vals;
                  D_rho = 0.;
                  D_rho(0,0) = 1./pow(aspectratio,0.5);
                  D_rho(1,1) = pow(aspectratio,0.5);
               }
               else
               {
                  par_vals.SetDataAndSize(tspec_vals.GetData()+
                                          aspectratioidx*ndofs, ndofs*3);
                  par_vals_c1.SetDataAndSize(par_vals.GetData(), ndofs);
                  par_vals_c2.SetDataAndSize(par_vals.GetData()+ndofs, ndofs);
                  par_vals_c3.SetDataAndSize(par_vals.GetData()+2*ndofs, ndofs);

                  const double rho1 = shape * par_vals_c1;
                  const double rho2 = shape * par_vals_c2;
                  const double rho3 = shape * par_vals_c3;
                  D_rho = 0.;
                  D_rho(0,0) = pow(rho1,2./3.);
                  D_rho(1,1) = pow(rho2,2./3.);
                  D_rho(2,2) = pow(rho3,2./3.);
               }

               DenseMatrix Temp = Jtr(i);
               Mult(D_rho, Temp, Jtr(i));
            } //Done aspect ratio

            if (skewidx != -1) //Set skew
            {
               if (dim == 2)
               {
                  par_vals.SetDataAndSize(tspec_vals.GetData()+
                                          skewidx*ndofs, ndofs);

                  const double skew = shape * par_vals;

                  Q_phi = 0.;
                  Q_phi(0,0) = 1.;
                  Q_phi(0,1) = cos(skew);
                  Q_phi(1,1) = sin(skew);
               }
               else
               {
                  par_vals.SetDataAndSize(tspec_vals.GetData()+
                                          skewidx*ndofs, ndofs*3);
                  par_vals_c1.SetDataAndSize(par_vals.GetData(), ndofs);
                  par_vals_c2.SetDataAndSize(par_vals.GetData()+ndofs, ndofs);
                  par_vals_c3.SetDataAndSize(par_vals.GetData()+2*ndofs, ndofs);

                  const double phi12  = shape * par_vals_c1;
                  const double phi13  = shape * par_vals_c2;
                  const double chi = shape * par_vals_c3;

                  Q_phi = 0.;
                  Q_phi(0,0) = 1.;
                  Q_phi(0,1) = cos(phi12);
                  Q_phi(0,2) = cos(phi13);

                  Q_phi(1,1) = sin(phi12);
                  Q_phi(1,2) = sin(phi13)*cos(chi);

                  Q_phi(2,2) = sin(phi13)*sin(chi);
               }

               DenseMatrix Temp = Jtr(i);
               Mult(Q_phi, Temp, Jtr(i));
            } // done skew

            if (orientationidx != -1) //Set orientation
            {
               if (dim == 2)
               {
                  par_vals.SetDataAndSize(tspec_vals.GetData()+
                                          orientationidx*ndofs, ndofs);

                  const double theta = shape * par_vals;
                  R_theta(0,0) =  cos(theta);
                  R_theta(0,1) = -sin(theta);
                  R_theta(1,0) =  sin(theta);
                  R_theta(1,1) =  cos(theta);
               }
               else
               {
                  par_vals.SetDataAndSize(tspec_vals.GetData()+
                                          orientationidx*ndofs, ndofs*3);
                  par_vals_c1.SetDataAndSize(par_vals.GetData(), ndofs);
                  par_vals_c2.SetDataAndSize(par_vals.GetData()+ndofs, ndofs);
                  par_vals_c3.SetDataAndSize(par_vals.GetData()+2*ndofs, ndofs);

                  const double theta = shape * par_vals_c1;
                  const double psi   = shape * par_vals_c2;
                  const double beta  = shape * par_vals_c3;

                  DenseMatrix R_tp(dim), R_beta(dim), R_theta(dim);
                  double ct = cos(theta), st = sin(theta),
                         cp = cos(psi),   sp = sin(psi);
                  R_tp(0,0) = ct*sp;
                  R_tp(1,0) = st*sp;
                  R_tp(2,0) = cp;

                  R_tp(0,1) = -(ct*st*sp*sp)/(1+cp);
                  R_tp(1,1) = cp+(pow(ct,2.)*pow(sp,2.))/(1+cp);
                  R_tp(2,1) = -st*sp;

                  R_tp(0,2) = -cp-(pow(st,2.)*pow(sp,2.))/(1+cp);
                  R_tp(1,2) = -R_tp(0,1);
                  R_tp(2,2) =  ct*sp;

                  R_beta = 0.;
                  R_beta(0,0) = 1.;
                  R_beta(1,1) =  cos(beta);
                  R_beta(1,2) = -sin(beta);
                  R_beta(2,1) =  sin(beta);
                  R_beta(2,2) =  cos(beta);

                  Mult(R_tp, R_beta, R_theta);
               }
               DenseMatrix Temp = Jtr(i);
               Mult(R_theta, Temp, Jtr(i));
            } // done orientation
         }
         break;
      }
      default:
         MFEM_ABORT("Incompatible target type for discrete adaptation!");
   }
}

void DiscreteAdaptTC::UpdateGradientTargetSpecification(const Vector &x,
                                                        const double dx,
                                                        bool use_flag)
{
   if (use_flag && good_tspec_grad) { return; }

   const int dim = tspec_fes->GetFE(0)->GetDim(),
             cnt = x.Size()/dim;

   tspec_pert1h.SetSize(x.Size()*ncomp);

   Vector TSpecTemp;
   Vector xtemp = x;
   for (int j = 0; j < dim; j++)
   {
      for (int i = 0; i < cnt; i++) { xtemp(j*cnt+i) += dx; }

      TSpecTemp.NewDataAndSize(tspec_pert1h.GetData() + j*cnt*ncomp, cnt*ncomp);
      UpdateTargetSpecification(xtemp, TSpecTemp);

      for (int i = 0; i < cnt; i++) { xtemp(j*cnt+i) -= dx; }
   }

   good_tspec_grad = use_flag;
}

void DiscreteAdaptTC::UpdateHessianTargetSpecification(const Vector &x,
                                                       double dx, bool use_flag)
{

   if (use_flag && good_tspec_hess) { return; }

   const int dim    = tspec_fes->GetFE(0)->GetDim(),
             cnt    = x.Size()/dim,
             totmix = 1+2*(dim-2);

   tspec_pert2h.SetSize(cnt*dim*ncomp);
   tspec_pertmix.SetSize(cnt*totmix*ncomp);

   Vector TSpecTemp;
   Vector xtemp = x;

   // T(x+2h)
   for (int j = 0; j < dim; j++)
   {
      for (int i = 0; i < cnt; i++) { xtemp(j*cnt+i) += 2*dx; }

      TSpecTemp.NewDataAndSize(tspec_pert2h.GetData() + j*cnt*ncomp, cnt*ncomp);
      UpdateTargetSpecification(xtemp, TSpecTemp);

      for (int i = 0; i < cnt; i++) { xtemp(j*cnt+i) -= 2*dx; }
   }

   // T(x+h,y+h)
   int j = 0;
   for (int k1 = 0; k1 < dim; k1++)
   {
      for (int k2 = 0; (k1 != k2) && (k2 < dim); k2++)
      {
         for (int i = 0; i < cnt; i++)
         {
            xtemp(k1*cnt+i) += dx;
            xtemp(k2*cnt+i) += dx;
         }

         TSpecTemp.NewDataAndSize(tspec_pertmix.GetData() + j*cnt*ncomp, cnt*ncomp);
         UpdateTargetSpecification(xtemp, TSpecTemp);

         for (int i = 0; i < cnt; i++)
         {
            xtemp(k1*cnt+i) -= dx;
            xtemp(k2*cnt+i) -= dx;
         }
         j++;
      }
   }

   good_tspec_hess = use_flag;
}

void AdaptivityEvaluator::SetSerialMetaInfo(const Mesh &m,
                                            const FiniteElementCollection &fec,
                                            int num_comp)
{
   delete fes;
   delete mesh;
   mesh = new Mesh(m, true);
   fes = new FiniteElementSpace(mesh, &fec, num_comp);
   dim = fes->GetFE(0)->GetDim();
   ncomp = num_comp;
}

#ifdef MFEM_USE_MPI
void AdaptivityEvaluator::SetParMetaInfo(const ParMesh &m,
                                         const FiniteElementCollection &fec,
                                         int num_comp)
{
   delete pfes;
   delete pmesh;
   pmesh = new ParMesh(m, true);
   pfes  = new ParFiniteElementSpace(pmesh, &fec, num_comp);
   dim = pfes->GetFE(0)->GetDim();
   ncomp = num_comp;
}
#endif

AdaptivityEvaluator::~AdaptivityEvaluator()
{
   delete fes;
   delete mesh;
#ifdef MFEM_USE_MPI
   delete pfes;
   delete pmesh;
#endif
}

TMOP_Integrator::~TMOP_Integrator()
{
   delete lim_func;
   delete zeta;
   for (int i = 0; i < ElemDer.Size(); i++)
   {
      delete ElemDer[i];
      delete ElemPertEnergy[i];
   }
}

void TMOP_Integrator::EnableLimiting(const GridFunction &n0,
                                     const GridFunction &dist, Coefficient &w0,
                                     TMOP_LimiterFunction *lfunc)
{
   EnableLimiting(n0, w0, lfunc);
   lim_dist = &dist;
}
void TMOP_Integrator::EnableLimiting(const GridFunction &n0, Coefficient &w0,
                                     TMOP_LimiterFunction *lfunc)
{
   nodes0 = &n0;
   coeff0 = &w0;
   lim_dist = NULL;

   delete lim_func;
   if (lfunc)
   {
      lim_func = lfunc;
   }
   else
   {
      lim_func = new TMOP_QuadraticLimiter;
   }
}

void TMOP_Integrator::EnableAdaptiveLimiting(const GridFunction &z0,
                                             Coefficient &coeff,
                                             AdaptivityEvaluator &ae)
{
   zeta_0 = &z0;
   delete zeta;
   zeta   = new GridFunction(z0);
   coeff_zeta = &coeff;
   adapt_eval = &ae;

   adapt_eval->SetSerialMetaInfo(*zeta->FESpace()->GetMesh(),
                                 *zeta->FESpace()->FEColl(), 1);
   adapt_eval->SetInitialField
   (*zeta->FESpace()->GetMesh()->GetNodes(), *zeta);
}

#ifdef MFEM_USE_MPI
void TMOP_Integrator::EnableAdaptiveLimiting(const ParGridFunction &z0,
                                             Coefficient &coeff,
                                             AdaptivityEvaluator &ae)
{
   zeta_0 = &z0;
   delete zeta;
   zeta   = new GridFunction(z0);
   coeff_zeta = &coeff;
   adapt_eval = &ae;

   adapt_eval->SetParMetaInfo(*z0.ParFESpace()->GetParMesh(),
                              *z0.ParFESpace()->FEColl(), 1);
   adapt_eval->SetInitialField
   (*zeta->FESpace()->GetMesh()->GetNodes(), *zeta);
}
#endif

double TMOP_Integrator::GetElementEnergy(const FiniteElement &el,
                                         ElementTransformation &T,
                                         const Vector &elfun)
{
   const int dof = el.GetDof(), dim = el.GetDim();
   double energy;

   // No adaptive limiting terms if this is a FD computation.
   const bool adaptive_limiting = (zeta && fd_call_flag == false);

   DSh.SetSize(dof, dim);
   Jrt.SetSize(dim);
   Jpr.SetSize(dim);
   Jpt.SetSize(dim);
   PMatI.UseExternalData(elfun.GetData(), dof, dim);

   const IntegrationRule *ir = EnergyIntegrationRule(el);

   energy = 0.0;
   DenseTensor Jtr(dim, dim, ir->GetNPoints());
   targetC->ComputeElementTargets(T.ElementNo, el, *ir, elfun, Jtr);

   // Limited case.
   Vector shape, p, p0, d_vals;
   DenseMatrix pos0;
   if (coeff0)
   {
      shape.SetSize(dof);
      p.SetSize(dim);
      p0.SetSize(dim);
      pos0.SetSize(dof, dim);
      Vector pos0V(pos0.Data(), dof * dim);
      Array<int> pos_dofs;
      nodes0->FESpace()->GetElementVDofs(T.ElementNo, pos_dofs);
      nodes0->GetSubVector(pos_dofs, pos0V);
      if (lim_dist)
      {
         lim_dist->GetValues(T.ElementNo, *ir, d_vals);
      }
      else
      {
         d_vals.SetSize(ir->GetNPoints()); d_vals = 1.0;
      }
   }

   // Define ref->physical transformation, when a Coefficient is specified.
   IsoparametricTransformation *Tpr = NULL;
   if (coeff1 || coeff0 || adaptive_limiting)
   {
      Tpr = new IsoparametricTransformation;
      Tpr->SetFE(&el);
      Tpr->ElementNo = T.ElementNo;
      Tpr->ElementType = ElementTransformation::ELEMENT;
      Tpr->Attribute = T.Attribute;
      Tpr->GetPointMat().Transpose(PMatI); // PointMat = PMatI^T
   }
   // TODO: computing the coefficients 'coeff1' and 'coeff0' in physical
   //       coordinates means that, generally, the gradient and Hessian of the
   //       TMOP_Integrator will depend on the derivatives of the coefficients.
   //
   //       In some cases the coefficients are independent of any movement of
   //       the physical coordinates (i.e. changes in 'elfun'), e.g. when the
   //       coefficient is a ConstantCoefficient or a GridFunctionCoefficient.

   Vector zeta_q, zeta0_q;
   if (adaptive_limiting)
   {
      zeta->GetValues(T.ElementNo, *ir, zeta_q);
      zeta_0->GetValues(T.ElementNo, *ir, zeta0_q);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      const DenseMatrix &Jtr_i = Jtr(i);
      metric->SetTargetJacobian(Jtr_i);
      CalcInverse(Jtr_i, Jrt);
      const double weight = ip.weight * Jtr_i.Det();

      el.CalcDShape(ip, DSh);
      MultAtB(PMatI, DSh, Jpr);
      Mult(Jpr, Jrt, Jpt);

      double val = metric_normal * metric->EvalW(Jpt);
      if (coeff1) { val *= coeff1->Eval(*Tpr, ip); }

      if (coeff0)
      {
         el.CalcShape(ip, shape);
         PMatI.MultTranspose(shape, p);
         pos0.MultTranspose(shape, p0);
         val += lim_normal *
                lim_func->Eval(p, p0, d_vals(i)) * coeff0->Eval(*Tpr, ip);
      }

      if (adaptive_limiting)
      {
         const double diff = zeta_q(i) - zeta0_q(i);
         val += coeff_zeta->Eval(*Tpr, ip) * lim_normal * diff * diff;
      }

      energy += weight * val;
   }
   delete Tpr;

   return energy;
}
void TMOP_Integrator::AssembleElementVector(const FiniteElement &el,
                                            ElementTransformation &T,
                                            const Vector &elfun, Vector &elvect)
{
   if (!fdflag)
   {
      AssembleElementVectorExact(el, T, elfun, elvect);
   }
   else
   {
      AssembleElementVectorFD(el, T, elfun, elvect);
   }
}

void TMOP_Integrator::AssembleElementGrad(const FiniteElement &el,
                                          ElementTransformation &T,
                                          const Vector &elfun,
                                          DenseMatrix &elmat)
{
   if (!fdflag)
   {
      AssembleElementGradExact(el, T, elfun, elmat);
   }
   else
   {
      AssembleElementGradFD(el, T, elfun, elmat);
   }
}

void TMOP_Integrator::AssembleElementVectorExact(const FiniteElement &el,
                                                 ElementTransformation &T,
                                                 const Vector &elfun,
                                                 Vector &elvect)
{
   const int dof = el.GetDof(), dim = el.GetDim();

   DSh.SetSize(dof, dim);
   DS.SetSize(dof, dim);
   Jrt.SetSize(dim);
   Jpt.SetSize(dim);
   P.SetSize(dim);
   PMatI.UseExternalData(elfun.GetData(), dof, dim);
   elvect.SetSize(dof*dim);
   PMatO.UseExternalData(elvect.GetData(), dof, dim);

   const IntegrationRule *ir = ActionIntegrationRule(el);
   const int nqp = ir->GetNPoints();

   elvect = 0.0;
   Vector weights(nqp);
   DenseTensor Jtr(dim, dim, nqp);
   targetC->ComputeElementTargets(T.ElementNo, el, *ir, elfun, Jtr);

   // Limited case.
   DenseMatrix pos0;
   Vector shape, p, p0, d_vals, grad;
   if (coeff0)
   {
      shape.SetSize(dof);
      p.SetSize(dim);
      p0.SetSize(dim);
      pos0.SetSize(dof, dim);
      Vector pos0V(pos0.Data(), dof * dim);
      Array<int> pos_dofs;
      nodes0->FESpace()->GetElementVDofs(T.ElementNo, pos_dofs);
      nodes0->GetSubVector(pos_dofs, pos0V);
      if (lim_dist)
      {
         lim_dist->GetValues(T.ElementNo, *ir, d_vals);
      }
      else
      {
         d_vals.SetSize(nqp); d_vals = 1.0;
      }
   }

   // Define ref->physical transformation, when a Coefficient is specified.
   IsoparametricTransformation *Tpr = NULL;
   if (coeff1 || coeff0 || zeta)
   {
      Tpr = new IsoparametricTransformation;
      Tpr->SetFE(&el);
      Tpr->ElementNo = T.ElementNo;
      Tpr->ElementType = ElementTransformation::ELEMENT;
      Tpr->Attribute = T.Attribute;
      Tpr->GetPointMat().Transpose(PMatI); // PointMat = PMatI^T
   }

   for (int q = 0; q < nqp; q++)
   {
      const IntegrationPoint &ip = ir->IntPoint(q);
      const DenseMatrix &Jtr_q = Jtr(q);
      metric->SetTargetJacobian(Jtr_q);
      CalcInverse(Jtr_q, Jrt);
      weights(q) = ip.weight * Jtr_q.Det();
      double weight_m = weights(q) * metric_normal;

      el.CalcDShape(ip, DSh);
      Mult(DSh, Jrt, DS);
      MultAtB(PMatI, DS, Jpt);

      metric->EvalP(Jpt, P);

      if (coeff1) { weight_m *= coeff1->Eval(*Tpr, ip); }

      P *= weight_m;
      AddMultABt(DS, P, PMatO);

      // TODO: derivatives of adaptivity-based targets.

      if (coeff0)
      {
         el.CalcShape(ip, shape);
         PMatI.MultTranspose(shape, p);
         pos0.MultTranspose(shape, p0);
         lim_func->Eval_d1(p, p0, d_vals(q), grad);
         grad *= weights(q) * lim_normal * coeff0->Eval(*Tpr, ip);
         AddMultVWt(shape, grad, PMatO);
      }
   }

   if (zeta) { AssembleElemVecAdaptLim(el, weights, *Tpr, *ir, PMatO); }

   delete Tpr;
}

void TMOP_Integrator::AssembleElementGradExact(const FiniteElement &el,
                                               ElementTransformation &T,
                                               const Vector &elfun,
                                               DenseMatrix &elmat)
{
   const int dof = el.GetDof(), dim = el.GetDim();

   DSh.SetSize(dof, dim);
   DS.SetSize(dof, dim);
   Jrt.SetSize(dim);
   Jpt.SetSize(dim);
   PMatI.UseExternalData(elfun.GetData(), dof, dim);
   elmat.SetSize(dof*dim);

   const IntegrationRule *ir = GradientIntegrationRule(el);
   const int nqp = ir->GetNPoints();

   elmat = 0.0;
   Vector weights(nqp);
   DenseTensor Jtr(dim, dim, nqp);
   targetC->ComputeElementTargets(T.ElementNo, el, *ir, elfun, Jtr);

   // Limited case.
   DenseMatrix pos0, grad_grad;
   Vector shape, p, p0, d_vals;
   if (coeff0)
   {
      shape.SetSize(dof);
      p.SetSize(dim);
      p0.SetSize(dim);
      pos0.SetSize(dof, dim);
      Vector pos0V(pos0.Data(), dof * dim);
      Array<int> pos_dofs;
      nodes0->FESpace()->GetElementVDofs(T.ElementNo, pos_dofs);
      nodes0->GetSubVector(pos_dofs, pos0V);
      if (lim_dist)
      {
         lim_dist->GetValues(T.ElementNo, *ir, d_vals);
      }
      else
      {
         d_vals.SetSize(nqp); d_vals = 1.0;
      }
   }

   // Define ref->physical transformation, when a Coefficient is specified.
   IsoparametricTransformation *Tpr = NULL;
   if (coeff1 || coeff0 || zeta)
   {
      Tpr = new IsoparametricTransformation;
      Tpr->SetFE(&el);
      Tpr->ElementNo = T.ElementNo;
      Tpr->ElementType = ElementTransformation::ELEMENT;
      Tpr->Attribute = T.Attribute;
      Tpr->GetPointMat().Transpose(PMatI);
   }

   for (int q = 0; q < nqp; q++)
   {
      const IntegrationPoint &ip = ir->IntPoint(q);
      const DenseMatrix &Jtr_q = Jtr(q);
      metric->SetTargetJacobian(Jtr_q);
      CalcInverse(Jtr_q, Jrt);
      weights(q) = ip.weight * Jtr_q.Det();
      double weight_m = weights(q) * metric_normal;

      el.CalcDShape(ip, DSh);
      Mult(DSh, Jrt, DS);
      MultAtB(PMatI, DS, Jpt);

      if (coeff1) { weight_m *= coeff1->Eval(*Tpr, ip); }

      metric->AssembleH(Jpt, DS, weight_m, elmat);

      // TODO: derivatives of adaptivity-based targets.

      // TODO optimize by symmetry.
      if (coeff0)
      {
         el.CalcShape(ip, shape);
         PMatI.MultTranspose(shape, p);
         pos0.MultTranspose(shape, p0);
         weight_m = weights(q) * lim_normal * coeff0->Eval(*Tpr, ip);
         lim_func->Eval_d2(p, p0, d_vals(q), grad_grad);
         for (int i = 0; i < dof; i++)
         {
            const double w_shape_i = weight_m * shape(i);
            for (int j = 0; j < dof; j++)
            {
               const double w = w_shape_i * shape(j);
               for (int d1 = 0; d1 < dim; d1++)
               {
                  for (int d2 = 0; d2 < dim; d2++)
                  {
                     elmat(d1*dof + i, d2*dof + j) += w * grad_grad(d1, d2);
                  }
               }
            }
         }
      }
   }

   if (zeta) { AssembleElemGradAdaptLim(el, weights, *Tpr, *ir, elmat); }

   delete Tpr;
}

void TMOP_Integrator::AssembleElemVecAdaptLim(const FiniteElement &el,
                                              const Vector &weights,
                                              IsoparametricTransformation &Tpr,
                                              const IntegrationRule &ir,
                                              DenseMatrix &mat)
{
   if (zeta == NULL) { return; }

   const int dof = el.GetDof(), dim = el.GetDim();
   Vector shape(dof), zeta_e, zeta_q, zeta0_q;

   Array<int> dofs;
   zeta->FESpace()->GetElementDofs(Tpr.ElementNo, dofs);
   zeta->GetSubVector(dofs, zeta_e);
   zeta->GetValues(Tpr.ElementNo, ir, zeta_q);
   zeta_0->GetValues(Tpr.ElementNo, ir, zeta0_q);

   // Project the gradient of zeta in the same space.
   // The FE coefficients of the gradient go in zeta_grad_e.
   DenseMatrix zeta_grad_e(dof, dim);
   DenseMatrix grad_phys; // This will be (dof x dim, dof).
   el.ProjectGrad(el, Tpr, grad_phys);
   Vector grad_ptr(zeta_grad_e.GetData(), dof*dim);
   grad_phys.Mult(zeta_e, grad_ptr);

   Vector zeta_grad_q(dim);

   const int nqp = weights.Size();
   for (int q = 0; q < nqp; q++)
   {
      const IntegrationPoint &ip = ir.IntPoint(q);
      el.CalcShape(ip, shape);
      zeta_grad_e.MultTranspose(shape, zeta_grad_q);
      zeta_grad_q *= 2.0 * (zeta_q(q) - zeta0_q(q));
      zeta_grad_q *= weights(q) * lim_normal * coeff_zeta->Eval(Tpr, ip);
      AddMultVWt(shape, zeta_grad_q, mat);
   }
}

void TMOP_Integrator::AssembleElemGradAdaptLim(const FiniteElement &el,
                                               const Vector &weights,
                                               IsoparametricTransformation &Tpr,
                                               const IntegrationRule &ir,
                                               DenseMatrix &mat)
{
   if (zeta == NULL) { return; }

   const int dof = el.GetDof(), dim = el.GetDim();
   Vector shape(dof), zeta_e, zeta_q, zeta0_q;

   Array<int> dofs;
   zeta->FESpace()->GetElementDofs(Tpr.ElementNo, dofs);
   zeta->GetSubVector(dofs, zeta_e);
   zeta->GetValues(Tpr.ElementNo, ir, zeta_q);
   zeta_0->GetValues(Tpr.ElementNo, ir, zeta0_q);

   // Project the gradient of zeta in the same space.
   // The FE coefficients of the gradient go in zeta_grad_e.
   DenseMatrix zeta_grad_e(dof, dim);
   DenseMatrix grad_phys; // This will be (dof x dim, dof).
   el.ProjectGrad(el, Tpr, grad_phys);
   Vector grad_ptr(zeta_grad_e.GetData(), dof*dim);
   grad_phys.Mult(zeta_e, grad_ptr);

   // Project the gradient of each gradient of zeta in the same space.
   // The FE coefficients of the second derivatives go in zeta_grad_grad_e.
   DenseMatrix zeta_grad_grad_e(dof*dim, dim);
   Mult(grad_phys, zeta_grad_e, zeta_grad_grad_e);
   // Reshape to be more convenient later (no change in the data).
   zeta_grad_grad_e.SetSize(dof, dim*dim);

   Vector zeta_grad_q(dim);
   DenseMatrix zeta_grad_grad_q(dim, dim);

   const int nqp = weights.Size();
   for (int q = 0; q < nqp; q++)
   {
      const IntegrationPoint &ip = ir.IntPoint(q);
      el.CalcShape(ip, shape);

      zeta_grad_e.MultTranspose(shape, zeta_grad_q);
      Vector gg_ptr(zeta_grad_grad_q.GetData(), dim*dim);
      zeta_grad_grad_e.MultTranspose(shape, gg_ptr);

      const double w = weights(q) * lim_normal * coeff_zeta->Eval(Tpr, ip);
      for (int i = 0; i < dof * dim; i++)
      {
         const int idof = i % dof, idim = i / dof;
         for (int j = 0; j <= i; j++)
         {
            const int jdof = j % dof, jdim = j / dof;
            const double entry =
               w * ( 2.0 * zeta_grad_q(idim) * shape(idof) *
                     /* */ zeta_grad_q(jdim) * shape(jdof) +
                     2.0 * (zeta_q(q) - zeta0_q(q)) *
                     zeta_grad_grad_q(idim, jdim) * shape(idof) * shape(jdof));
            mat(i, j) += entry;
            if (i != j) { mat(j, i) += entry; }
         }
      }
   }
}

double TMOP_Integrator::GetFDDerivative(const FiniteElement &el,
                                        ElementTransformation &T,
                                        Vector &elfun, const int dofidx,
                                        const int dir, const double e_fx,
                                        bool update_stored)
{
   int dof = el.GetDof();
   int idx = dir*dof+dofidx;
   elfun[idx]    += dx;
   double e_fxph  = GetElementEnergy(el, T, elfun);
   elfun[idx]    -= dx;
   double dfdx    = (e_fxph-e_fx)/dx;

   if (update_stored)
   {
      (*(ElemPertEnergy[T.ElementNo]))(idx) = e_fxph;
      (*(ElemDer[T.ElementNo]))(idx) = dfdx;
   }

   return dfdx;
}

void TMOP_Integrator::AssembleElementVectorFD(const FiniteElement &el,
                                              ElementTransformation &T,
                                              const Vector &elfun,
                                              Vector &elvect)
{
   const int dof = el.GetDof(), dim = el.GetDim(), elnum = T.ElementNo;
   if (elnum>=ElemDer.Size())
   {
      ElemDer.Append(new Vector);
      ElemPertEnergy.Append(new Vector);
      ElemDer[elnum]->SetSize(dof*dim);
      ElemPertEnergy[elnum]->SetSize(dof*dim);
   }

   elvect.SetSize(dof*dim);
   Vector elfunmod(elfun);

   // In GetElementEnergy(), skip terms that have exact derivative calculations.
   fd_call_flag = true;

   // Energy for unperturbed configuration.
   const double e_fx = GetElementEnergy(el, T, elfun);

   for (int j = 0; j < dim; j++)
   {
      for (int i = 0; i < dof; i++)
      {
         if (discr_tc)
         {
            discr_tc->UpdateTargetSpecificationAtNode(
               el, T, i, j, discr_tc->GetTspecPert1H());
         }
         elvect(j*dof+i) = GetFDDerivative(el, T, elfunmod, i, j, e_fx, true);
         if (discr_tc) { discr_tc->RestoreTargetSpecificationAtNode(T, i); }
      }
   }
   fd_call_flag = false;

   // Contributions from adaptive limiting (exact derivatives).
   if (zeta)
   {
      const IntegrationRule *ir = ActionIntegrationRule(el);
      const int nqp = ir->GetNPoints();
      DenseTensor Jtr(dim, dim, nqp);
      targetC->ComputeElementTargets(T.ElementNo, el, *ir, elfun, Jtr);

      IsoparametricTransformation Tpr;
      Tpr.SetFE(&el);
      Tpr.ElementNo = T.ElementNo;
      Tpr.Attribute = T.Attribute;
      PMatI.UseExternalData(elfun.GetData(), dof, dim);
      Tpr.GetPointMat().Transpose(PMatI); // PointMat = PMatI^T

      Vector weights(nqp);
      for (int q = 0; q < nqp; q++)
      {
         weights(q) = ir->IntPoint(q).weight * Jtr(q).Det();
      }

      PMatO.UseExternalData(elvect.GetData(), dof, dim);
      AssembleElemVecAdaptLim(el, weights, Tpr, *ir, PMatO);
   }
}

void TMOP_Integrator::AssembleElementGradFD(const FiniteElement &el,
                                            ElementTransformation &T,
                                            const Vector &elfun,
                                            DenseMatrix &elmat)
{
   const int dof = el.GetDof(), dim = el.GetDim();

   elmat.SetSize(dof*dim);
   Vector elfunmod(elfun);

   const Vector &ElemDerLoc = *(ElemDer[T.ElementNo]);
   const Vector &ElemPertLoc = *(ElemPertEnergy[T.ElementNo]);

   // In GetElementEnergy(), skip terms that have exact derivative calculations.
   fd_call_flag = true;
   for (int i = 0; i < dof; i++)
   {
      for (int j = 0; j < i+1; j++)
      {
         for (int k1 = 0; k1 < dim; k1++)
         {
            for (int k2 = 0; k2 < dim; k2++)
            {
               elfunmod(k2*dof+j) += dx;

               if (discr_tc)
               {
                  discr_tc->UpdateTargetSpecificationAtNode(
                     el, T, j, k2, discr_tc->GetTspecPert1H());
                  if (j != i)
                  {
                     discr_tc->UpdateTargetSpecificationAtNode(
                        el, T, i, k1, discr_tc->GetTspecPert1H());
                  }
                  else // j==i
                  {
                     if (k1 != k2)
                     {
                        int idx = k1+k2-1;
                        discr_tc->UpdateTargetSpecificationAtNode(
                           el, T, i, idx, discr_tc->GetTspecPertMixH());
                     }
                     else // j==i && k1==k2
                     {
                        discr_tc->UpdateTargetSpecificationAtNode(
                           el, T, i, k1, discr_tc->GetTspecPert2H());
                     }
                  }
               }

               double e_fx   = ElemPertLoc(k2*dof+j);
               double e_fpxph = GetFDDerivative(el, T, elfunmod, i, k1, e_fx,
                                                false);
               elfunmod(k2*dof+j) -= dx;
               double e_fpx = ElemDerLoc(k1*dof+i);

               elmat(k1*dof+i, k2*dof+j) = (e_fpxph - e_fpx) / dx;
               elmat(k2*dof+j, k1*dof+i) = (e_fpxph - e_fpx) / dx;

               if (discr_tc)
               {
                  discr_tc->RestoreTargetSpecificationAtNode(T, i);
                  discr_tc->RestoreTargetSpecificationAtNode(T, j);
               }
            }
         }
      }
   }
   fd_call_flag = false;

   // Contributions from adaptive limiting.
   if (zeta)
   {
      const IntegrationRule *ir = GradientIntegrationRule(el);
      const int nqp = ir->GetNPoints();
      DenseTensor Jtr(dim, dim, nqp);
      targetC->ComputeElementTargets(T.ElementNo, el, *ir, elfun, Jtr);

      IsoparametricTransformation Tpr;
      Tpr.SetFE(&el);
      Tpr.ElementNo = T.ElementNo;
      Tpr.Attribute = T.Attribute;
      PMatI.UseExternalData(elfun.GetData(), dof, dim);
      Tpr.GetPointMat().Transpose(PMatI); // PointMat = PMatI^T

      Vector weights(nqp);
      for (int q = 0; q < nqp; q++)
      {
         weights(q) = ir->IntPoint(q).weight * Jtr(q).Det();
      }

      AssembleElemGradAdaptLim(el, weights, Tpr, *ir, elmat);
   }
}

void TMOP_Integrator::EnableNormalization(const GridFunction &x)
{
   ComputeNormalizationEnergies(x, metric_normal, lim_normal);
   metric_normal = 1.0 / metric_normal;
   lim_normal = 1.0 / lim_normal;
}

#ifdef MFEM_USE_MPI
void TMOP_Integrator::ParEnableNormalization(const ParGridFunction &x)
{
   double loc[2];
   ComputeNormalizationEnergies(x, loc[0], loc[1]);
   double rdc[2];
   MPI_Allreduce(loc, rdc, 2, MPI_DOUBLE, MPI_SUM, x.ParFESpace()->GetComm());
   metric_normal = 1.0 / rdc[0];
   lim_normal    = 1.0 / rdc[1];
}
#endif

void TMOP_Integrator::ComputeNormalizationEnergies(const GridFunction &x,
                                                   double &metric_energy,
                                                   double &lim_energy)
{
   Array<int> vdofs;
   Vector x_vals;
   const FiniteElementSpace* const fes = x.FESpace();
   const FiniteElement *fe = fes->GetFE(0);

   const int dof = fes->GetFE(0)->GetDof(), dim = fes->GetFE(0)->GetDim();

   DSh.SetSize(dof, dim);
   Jrt.SetSize(dim);
   Jpr.SetSize(dim);
   Jpt.SetSize(dim);

   const IntegrationRule *ir = EnergyIntegrationRule(*fe);
   DenseTensor Jtr(dim, dim, ir->GetNPoints());

   metric_energy = 0.0;
   lim_energy = 0.0;
   for (int i = 0; i < fes->GetNE(); i++)
   {
      fe = fes->GetFE(i);
      fes->GetElementVDofs(i, vdofs);
      x.GetSubVector(vdofs, x_vals);
      PMatI.UseExternalData(x_vals.GetData(), dof, dim);

      targetC->ComputeElementTargets(i, *fe, *ir, x_vals, Jtr);

      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         metric->SetTargetJacobian(Jtr(i));
         CalcInverse(Jtr(i), Jrt);
         const double weight = ip.weight * Jtr(i).Det();

         fe->CalcDShape(ip, DSh);
         MultAtB(PMatI, DSh, Jpr);
         Mult(Jpr, Jrt, Jpt);

         metric_energy += weight * metric->EvalW(Jpt);
         lim_energy += weight;
      }
   }
   if (targetC->ContainsVolumeInfo() == false)
   {
      // Special case when the targets don't contain volumetric information.
      lim_energy = fes->GetNE();
   }
}

void TMOP_Integrator::ComputeMinJac(const Vector &x,
                                    const FiniteElementSpace &fes)
{
   const FiniteElement *fe = fes.GetFE(0);
   const IntegrationRule *ir = EnergyIntegrationRule(*fe);
   const int NE = fes.GetMesh()->GetNE(), dim = fe->GetDim(),
             dof = fe->GetDof(), nsp = ir->GetNPoints();

   Array<int> xdofs(dof * dim);
   DenseMatrix Jpr(dim), dshape(dof, dim), pos(dof, dim);
   Vector posV(pos.Data(), dof * dim);

   dx = std::numeric_limits<float>::max();

   double detv_sum;
   double detv_avg_min = std::numeric_limits<float>::max();
   for (int i = 0; i < NE; i++)
   {
      fes.GetElementVDofs(i, xdofs);
      x.GetSubVector(xdofs, posV);
      detv_sum = 0.;
      for (int j = 0; j < nsp; j++)
      {
         fes.GetFE(i)->CalcDShape(ir->IntPoint(j), dshape);
         MultAtB(pos, dshape, Jpr);
         detv_sum += std::fabs(Jpr.Det());
      }
      double detv_avg = pow(detv_sum/nsp, 1./dim);
      detv_avg_min = std::min(detv_avg, detv_avg_min);
   }
   dx = detv_avg_min / dxscale;
}

void TMOP_Integrator::UpdateAfterMeshChange(const Vector &new_x)
{
   // Update zeta if adaptive limiting is enabled.
   if (zeta) { adapt_eval->ComputeAtNewPosition(new_x, *zeta); }
}

void TMOP_Integrator::ComputeFDh(const Vector &x, const FiniteElementSpace &fes)
{
   if (!fdflag) { return; }
   ComputeMinJac(x, fes);
}

#ifdef MFEM_USE_MPI
void TMOP_Integrator::ComputeFDh(const Vector &x,
                                 const ParFiniteElementSpace &pfes)
{
   if (!fdflag) { return; }
   ComputeMinJac(x, pfes);
   double min_jac_all;
   MPI_Allreduce(&dx, &min_jac_all, 1, MPI_DOUBLE, MPI_MIN, pfes.GetComm());
   dx = min_jac_all;
}
#endif

void TMOP_Integrator::EnableFiniteDifferences(const GridFunction &x)
{
   fdflag = true;
   const FiniteElementSpace *fes = x.FESpace();
   ComputeFDh(x,*fes);
   if (discr_tc)
   {
      discr_tc->UpdateTargetSpecification(x);
      discr_tc->UpdateGradientTargetSpecification(x, dx);
      discr_tc->UpdateHessianTargetSpecification(x, dx);
   }
}

#ifdef MFEM_USE_MPI
void TMOP_Integrator::EnableFiniteDifferences(const ParGridFunction &x)
{
   fdflag = true;
   const ParFiniteElementSpace *pfes = x.ParFESpace();
   ComputeFDh(x,*pfes);
   if (discr_tc)
   {
      discr_tc->UpdateTargetSpecification(x);
      discr_tc->UpdateGradientTargetSpecification(x, dx);
      discr_tc->UpdateHessianTargetSpecification(x, dx);
   }
}
#endif

void TMOPComboIntegrator::EnableLimiting(const GridFunction &n0,
                                         const GridFunction &dist,
                                         Coefficient &w0,
                                         TMOP_LimiterFunction *lfunc)
{
   MFEM_VERIFY(tmopi.Size() > 0, "No TMOP_Integrators were added.");

   tmopi[0]->EnableLimiting(n0, dist, w0, lfunc);
   for (int i = 1; i < tmopi.Size(); i++) { tmopi[i]->DisableLimiting(); }
}

void TMOPComboIntegrator::EnableLimiting(const GridFunction &n0,
                                         Coefficient &w0,
                                         TMOP_LimiterFunction *lfunc)
{
   MFEM_VERIFY(tmopi.Size() > 0, "No TMOP_Integrators were added.");

   tmopi[0]->EnableLimiting(n0, w0, lfunc);
   for (int i = 1; i < tmopi.Size(); i++) { tmopi[i]->DisableLimiting(); }
}

void TMOPComboIntegrator::SetLimitingNodes(const GridFunction &n0)
{
   MFEM_VERIFY(tmopi.Size() > 0, "No TMOP_Integrators were added.");

   tmopi[0]->SetLimitingNodes(n0);
   for (int i = 1; i < tmopi.Size(); i++) { tmopi[i]->DisableLimiting(); }
}

double TMOPComboIntegrator::GetElementEnergy(const FiniteElement &el,
                                             ElementTransformation &T,
                                             const Vector &elfun)
{
   double energy= 0.0;
   for (int i = 0; i < tmopi.Size(); i++)
   {
      energy += tmopi[i]->GetElementEnergy(el, T, elfun);
   }
   return energy;
}

void TMOPComboIntegrator::AssembleElementVector(const FiniteElement &el,
                                                ElementTransformation &T,
                                                const Vector &elfun,
                                                Vector &elvect)
{
   MFEM_VERIFY(tmopi.Size() > 0, "No TMOP_Integrators were added.");

   tmopi[0]->AssembleElementVector(el, T, elfun, elvect);
   for (int i = 1; i < tmopi.Size(); i++)
   {
      Vector elvect_i;
      tmopi[i]->AssembleElementVector(el, T, elfun, elvect_i);
      elvect += elvect_i;
   }
}

void TMOPComboIntegrator::AssembleElementGrad(const FiniteElement &el,
                                              ElementTransformation &T,
                                              const Vector &elfun,
                                              DenseMatrix &elmat)
{
   MFEM_VERIFY(tmopi.Size() > 0, "No TMOP_Integrators were added.");

   tmopi[0]->AssembleElementGrad(el, T, elfun, elmat);
   for (int i = 1; i < tmopi.Size(); i++)
   {
      DenseMatrix elmat_i;
      tmopi[i]->AssembleElementGrad(el, T, elfun, elmat_i);
      elmat += elmat_i;
   }
}

void TMOPComboIntegrator::EnableNormalization(const GridFunction &x)
{
   const int cnt = tmopi.Size();
   double total_integral = 0.0;
   for (int i = 0; i < cnt; i++)
   {
      tmopi[i]->EnableNormalization(x);
      total_integral += 1.0 / tmopi[i]->metric_normal;
   }
   for (int i = 0; i < cnt; i++)
   {
      tmopi[i]->metric_normal = 1.0 / total_integral;
   }
}

#ifdef MFEM_USE_MPI
void TMOPComboIntegrator::ParEnableNormalization(const ParGridFunction &x)
{
   const int cnt = tmopi.Size();
   double total_integral = 0.0;
   for (int i = 0; i < cnt; i++)
   {
      tmopi[i]->ParEnableNormalization(x);
      total_integral += 1.0 / tmopi[i]->metric_normal;
   }
   for (int i = 0; i < cnt; i++)
   {
      tmopi[i]->metric_normal = 1.0 / total_integral;
   }
}
#endif


void InterpolateTMOP_QualityMetric(TMOP_QualityMetric &metric,
                                   const TargetConstructor &tc,
                                   const Mesh &mesh, GridFunction &metric_gf)
{
   const int NE = mesh.GetNE();
   const GridFunction &nodes = *mesh.GetNodes();
   const int dim = mesh.Dimension();
   DenseMatrix Winv(dim), T(dim), A(dim), dshape, pos;
   Array<int> pos_dofs, gf_dofs;
   DenseTensor W;
   Vector posV;

   for (int i = 0; i < NE; i++)
   {
      const FiniteElement &fe_pos = *nodes.FESpace()->GetFE(i);
      const IntegrationRule &ir = metric_gf.FESpace()->GetFE(i)->GetNodes();
      const int nsp = ir.GetNPoints(), dof = fe_pos.GetDof();

      dshape.SetSize(dof, dim);
      pos.SetSize(dof, dim);
      posV.SetDataAndSize(pos.Data(), dof * dim);

      metric_gf.FESpace()->GetElementDofs(i, gf_dofs);
      nodes.FESpace()->GetElementVDofs(i, pos_dofs);
      nodes.GetSubVector(pos_dofs, posV);

      W.SetSize(dim, dim, nsp);
      tc.ComputeElementTargets(i, fe_pos, ir, posV, W);

      for (int j = 0; j < nsp; j++)
      {
         const DenseMatrix &Wj = W(j);
         metric.SetTargetJacobian(Wj);
         CalcInverse(Wj, Winv);

         const IntegrationPoint &ip = ir.IntPoint(j);
         fe_pos.CalcDShape(ip, dshape);
         MultAtB(pos, dshape, A);
         Mult(A, Winv, T);

         metric_gf(gf_dofs[j]) = metric.EvalW(T);
      }
   }
}
} // namespace mfem
