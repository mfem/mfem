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

#include "tmop.hpp"
#include "linearform.hpp"

// #define MFEM_USE_OLD_TMOP_INVARIANTS

namespace mfem
{

// I1 = |M|^2 / det(M).
double TMOP_QualityMetric::Dim2Invariant1(const DenseMatrix &M)
{
   MFEM_ASSERT(M.Height() == 2 && M.Width() == 2, "Incorrect dimensions!");

   return M.FNorm2() / M.Det();
}

// I2 = det(M).
double TMOP_QualityMetric::Dim2Invariant2(const DenseMatrix &M)
{
   MFEM_ASSERT(M.Height() == 2 && M.Width() == 2, "Incorrect dimensions!");

   return M.Det();
}

// dI1_dM = [ 2 det(M) M - |M|^2 adj(M)^T ] / det(M)^2.
void TMOP_QualityMetric::Dim2Invariant1_dM(const DenseMatrix &M,
                                           DenseMatrix &dM)
{
   MFEM_ASSERT(M.Height() == 2 && M.Width() == 2, "Incorrect dimensions!");

   const double fnorm2 = M.FNorm2(), det = M.Det();

   Dim2Invariant2_dM(M, dM);
   dM *= - fnorm2/(det*det);
   dM.Add(2.0/det, M);
}

// dI2_dM = d(det(M))_dM = adj(M)^T.
void TMOP_QualityMetric::Dim2Invariant2_dM(const DenseMatrix &M,
                                           DenseMatrix &dM)
{
   MFEM_ASSERT(M.Height() == 2 && M.Width() == 2, "Incorrect dimensions!");

   dM(0, 0) =  M(1, 1); dM(0, 1) = -M(1, 0);
   dM(1, 0) = -M(0, 1); dM(1, 1) =  M(0, 0);
}

// (dI1_dM)_d(Mij) = d[(2 det(M) M - |M|^2 adj(M)^T) / det(M)^2]_d[Mij].
void TMOP_QualityMetric::Dim2Invariant1_dMdM(const DenseMatrix &M, int i, int j,
                                             DenseMatrix &dMdM)
{
   MFEM_ASSERT(M.Height() == 2 && M.Width() == 2, "Incorrect dimensions!");

   // Compute d(det(M))_d(Mij), d(|M|^2)_d(Mij).
   DenseMatrix dI(2);
   Dim2Invariant2_dM(M, dI);
   const double ddet   = dI(i,j);
   const double dfnorm2 = 2.0 * M(i,j);

   const double det    = M.Det();
   const double det2   = det * det;
   const double fnorm2 = M.FNorm2();

   DenseMatrix dM(2); dM = 0.0; dM(i, j) = 1.0;
   DenseMatrix ddI(2);
   Dim2Invariant2_dMdM(M, i, j, ddI);
   for (int r = 0; r < 2; r++)
   {
      for (int c = 0; c < 2; c++)
      {
         dMdM(r,c) =
            (det2 *
             (2.0 * ddet * M(r,c) + 2.0 * det * dM(r,c)
              - dfnorm2 * dI(r,c) - fnorm2 * ddI(r,c))
             - 2.0 * det * ddet *
             (2.0 * det * M(r,c) - fnorm2 * dI(r,c)) ) / (det2 * det2);
      }
   }
}

// (dI2_dM)_d(Mij) = ...
void TMOP_QualityMetric::Dim2Invariant2_dMdM(const DenseMatrix &M, int i, int j,
                                             DenseMatrix &dMdM)
{
   MFEM_ASSERT(M.Height() == 2 && M.Width() == 2, "Incorrect dimensions!");

   dMdM = 0.0;
   dMdM(1-i,1-j) = (i == j) ? 1.0 : -1.0;
}

// I1 = |M|^2/ det(M)^(2/3).
double TMOP_QualityMetric::Dim3Invariant1(const DenseMatrix &M)
{
   MFEM_ASSERT(M.Height() == 3 && M.Width() == 3, "Incorrect dimensions!");

   const double fnorm = M.FNorm(), det = M.Det();
   return fnorm * fnorm / pow(det, 2.0/3.0);
}


// I2 = |adj(M)|^2 / det(M)^(4/3).
double TMOP_QualityMetric::Dim3Invariant2(const DenseMatrix &M)
{
   MFEM_ASSERT(M.Height() == 3 && M.Width() == 3, "Incorrect dimensions!");

   DenseMatrix Madj(3);
   CalcAdjugate(M, Madj);

   const double fnorm = Madj.FNorm(), det = M.Det();
   return fnorm * fnorm / pow(det, 4.0/3.0);
}


// I3 = det(M).
double TMOP_QualityMetric::Dim3Invariant3(const DenseMatrix &M)
{
   MFEM_ASSERT(M.Height() == 3 && M.Width() == 3, "Incorrect dimensions!");

   return M.Det();
}

// dI1_dM = [ 2 det(M) M - 2/3 |M|^2 det(M)^(-1/3) adj(M)^T ] / det(M)^4/3.
void TMOP_QualityMetric::Dim3Invariant1_dM(const DenseMatrix &M,
                                           DenseMatrix &dM)
{
   MFEM_ASSERT(M.Height() == 3 && M.Width() == 3, "Incorrect dimensions!");

   DenseMatrix Madj(3);
   CalcAdjugate(M, Madj);
   const double fnorm = M.FNorm(), det = M.Det();

   Dim3Invariant3_dM(M, dM);
   dM *= -(2./3.)* fnorm * fnorm * pow(det, -1./3.);
   dM.Add(2.0 * pow(det, 2./3.), M);
   dM *= 1.0 / pow(det, 4./3.);
}

// dI2_dM = [ -4/3 |adj(M)|^2  det(M)^(1/3) adj(M)^T ] / det(M)^(8/3).
void TMOP_QualityMetric::Dim3Invariant2_dM(const DenseMatrix &M,
                                           DenseMatrix &dM)
{
   MFEM_ASSERT(M.Height() == 3 && M.Width() == 3, "Incorrect dimensions!");

   DenseMatrix Madj(3);
   // dM will have Madj^t because it is the third invariant's derivative.
   CalcAdjugate(M, Madj);
   const double fnorm = Madj.FNorm(), det = M.Det();

   Dim3Invariant3_dM(M, dM);
   dM *= -(4./3.)* fnorm * fnorm * pow(det, 1./3.);
   dM *= 1.0 / (pow(det, 8./3.));
}

// dI3_dM = d(det(M))_dM = adj(M)^T.
void TMOP_QualityMetric::Dim3Invariant3_dM(const DenseMatrix &M,
                                           DenseMatrix &dM)
{
   MFEM_ASSERT(M.Height() == 3 && M.Width() == 3, "Incorrect dimensions!");

   CalcAdjugateTranspose(M, dM);
}

void TMOP_QualityMetric::Dim3Invariant1_dMdM(const DenseMatrix &M, int i, int j,
                                             DenseMatrix &dMdM)
{
   MFEM_ASSERT(M.Height() == 3 && M.Width() == 3, "Incorrect dimensions!");

   DenseMatrix dI(3);
   Dim3Invariant3_dM(M, dI);
   const double fnorm  = M.FNorm(), det = M.Det();

   DenseMatrix dM(3); dM = 0.0; dM(i, j) = 1.0;
   for (int r = 0; r < 3; r++)
   {
      for (int c = 0; c < 3; c++)
      {
         dMdM(r,c) = (2.0 * det * det * dM(r,c)
                      + dI(i,j) * (10./9.) * fnorm * fnorm * dI(r,c)
                      - (4./3.) * dI(i,j) * det * M(r,c)
                      - (4./3.) * det * M(i,j) * dI(r,c))
                     / pow(det, 8./3.);
      }
   }
}

void TMOP_QualityMetric::Dim3Invariant2_dMdM(const DenseMatrix &M, int i, int j,
                                             DenseMatrix &dMdM)
{
   MFEM_ASSERT(M.Height() == 3 && M.Width() == 3, "Incorrect dimensions!");

   DenseMatrix dI(3);
   Dim3Invariant3_dM(M, dI);
   DenseMatrix Madj(3);
   CalcAdjugate(M, Madj);
   const double det   = M.Det();
   const double fnorm = Madj.FNorm();

   DenseMatrix dM(3); dM = 0.0; dM(i, j) = 1.0;
   for (int r = 0; r < 3; r++)
   {
      for (int c = 0; c < 3; c++)
      {
         dMdM(r,c) = (28./9.) * fnorm * fnorm *
                     det * det * dI(i,j) * dI(r,c) / pow(det, 16./3.);
      }
   }
}

// (dI3_dM)_d(Mij) = 0.
void TMOP_QualityMetric::Dim3Invariant3_dMdM(const DenseMatrix &M, int i, int j,
                                             DenseMatrix &dMdM)
{
   MFEM_ASSERT(M.Height() == 3 && M.Width() == 3, "Incorrect dimensions!");

   dMdM(i, j) = 0.0;
}


double TMOP_Metric_001::EvalW(const DenseMatrix &Jpt) const
{
   return Dim2Invariant1(Jpt) * Dim2Invariant2(Jpt);
}

void TMOP_Metric_001::EvalP(const DenseMatrix &Jpt,
                            DenseMatrix &P) const
{
   Dim2Invariant1_dM(Jpt, P);
   P *= Dim2Invariant2(Jpt);

   DenseMatrix PP(P.Size());
   Dim2Invariant2_dM(Jpt, PP);
   PP *= Dim2Invariant1(Jpt);

   P += PP;
}

void TMOP_Metric_001::AssembleH(const DenseMatrix &Jpt,
                                const DenseMatrix &DS,
                                const double weight,
                                DenseMatrix &A) const
{
   const int dof = DS.Height(), dim = DS.Width();
   const double I1 = Dim2Invariant1(Jpt), I2 = Dim2Invariant2(Jpt);
   DenseMatrix dI1_dM(dim), dI1_dMdM(dim), dI2_dM(dim), dI2_dMdM(dim);
   Dim2Invariant1_dM(Jpt, dI1_dM);
   Dim2Invariant2_dM(Jpt, dI2_dM);

#if 0
   // Shorter version without using invariants.
   for (int i = 0; i < dof; i++)
   {
      for (int j = 0; j <= i; j++)
      {
         double a = 0.0;
         for (int d = 0; d < dim; d++)
         {
            a += DS(i,d)*DS(j,d);
         }
         a *= 0.5 * weight;
         for (int d = 0; d < dim; d++)
         {
            A(i+d*dof,j+d*dof) += a;
            if (i != j)
            {
               A(j+d*dof,i+d*dof) += a;
            }
         }
      }
   }
#endif

   // The first two go over the rows and cols of dP_dJ where P = dW_dJ.
   for (int r = 0; r < dim; r++)
   {
      for (int c = 0; c < dim; c++)
      {
         Dim2Invariant1_dMdM(Jpt, r, c, dI1_dMdM);
         Dim2Invariant2_dMdM(Jpt, r, c, dI2_dMdM);
         // Compute each entry of d(Prc)_dJ.
         for (int rr = 0; rr < dim; rr++)
         {
            for (int cc = 0; cc < dim; cc++)
            {
               const double entry_rr_cc =
                  dI1_dMdM(rr,cc) * I2 +
                  dI1_dM(r, c)    * dI2_dM(rr,cc) +
                  dI2_dMdM(rr,cc) * I1 +
                  dI2_dM(r, c)    * dI1_dM(rr,cc);

               for (int i = 0; i < dof; i++)
               {
                  for (int j = 0; j < dof; j++)
                  {
                     A(i+r*dof, j+rr*dof) +=
                        weight * DS(i, c) * DS(j, cc) * entry_rr_cc;
                  }
               }
            }
         }
      }
   }
}

double TMOP_Metric_002::EvalW(const DenseMatrix &Jpt) const
{
#ifdef MFEM_USE_OLD_TMOP_INVARIANTS
   return 0.5 * Dim2Invariant1(Jpt) - 1.0;
#else
   ie.SetJacobian(Jpt.GetData());
   return 0.5 * ie.Get_I1b() - 1.0;
#endif
}

void TMOP_Metric_002::EvalP(const DenseMatrix &Jpt,
                            DenseMatrix &P) const
{
#ifdef MFEM_USE_OLD_TMOP_INVARIANTS
   Dim2Invariant1_dM(Jpt, P);
   P *= 0.5;
#else
   ie.SetJacobian(Jpt.GetData());
   P = ie.Get_dI1b();
   P *= 0.5;
#endif
}

void TMOP_Metric_002::AssembleH(const DenseMatrix &Jpt,
                                const DenseMatrix &DS,
                                const double weight,
                                DenseMatrix &A) const
{
#ifdef MFEM_USE_OLD_TMOP_INVARIANTS
   const int dof = DS.Height(), dim = DS.Width();
   DenseMatrix dI1_dMdM(dim);

   // The first two go over the rows and cols of dP_dJ where P = dW_dJ.
   for (int r = 0; r < dim; r++)
   {
      for (int c = 0; c < dim; c++)
      {
         Dim2Invariant1_dMdM(Jpt, r, c, dI1_dMdM);

         // Compute each entry of d(Prc)_dJ.
         for (int rr = 0; rr < dim; rr++)
         {
            for (int cc = 0; cc < dim; cc++)
            {
               const double entry_rr_cc = 0.5 * dI1_dMdM(rr,cc);

               for (int i = 0; i < dof; i++)
               {
                  for (int j = 0; j < dof; j++)
                  {
                     A(i+r*dof, j+rr*dof) +=
                        weight * DS(i, c) * DS(j, cc) * entry_rr_cc;
                  }
               }
            }
         }
      }
   }
#else
   ie.SetJacobian(Jpt.GetData());
   ie.SetDerivativeMatrix(DS.Height(), DS.GetData());
   ie.Assemble_ddI1b(0.5*weight, A.GetData());
#endif
}

double TMOP_Metric_007::EvalW(const DenseMatrix &Jpt) const
{
   const double I2 = Dim2Invariant2(Jpt);
   return Dim2Invariant1(Jpt) * (I2 + 1.0 / I2) - 4.0;
}

void TMOP_Metric_007::EvalP(const DenseMatrix &Jpt,
                            DenseMatrix &P) const
{
   const double I1 = Dim2Invariant1(Jpt), I2 = Dim2Invariant2(Jpt);
   Dim2Invariant1_dM(Jpt, P);
   P *= (I2 + 1.0 / I2);

   DenseMatrix PP(P.Size());
   Dim2Invariant2_dM(Jpt, PP);
   PP *= I1 * (1.0 - 1.0 / (I2 * I2));

   P += PP;
}

void TMOP_Metric_007::AssembleH(const DenseMatrix &Jpt,
                                const DenseMatrix &DS,
                                const double weight,
                                DenseMatrix &A) const
{
   const int dof = DS.Height(), dim = DS.Width();
   const double I1 = Dim2Invariant1(Jpt),
                I2 = Dim2Invariant2(Jpt), iI2 = 1.0/I2;
   DenseMatrix dI1_dM(dim), dI1_dMdM(dim), dI2_dM(dim), dI2_dMdM(dim);
   Dim2Invariant1_dM(Jpt, dI1_dM);
   Dim2Invariant2_dM(Jpt, dI2_dM);

   // The first two go over the rows and cols of dP_dJ where P = dW_dJ.
   for (int r = 0; r < dim; r++)
   {
      for (int c = 0; c < dim; c++)
      {
         Dim2Invariant1_dMdM(Jpt, r, c, dI1_dMdM);
         Dim2Invariant2_dMdM(Jpt, r, c, dI2_dMdM);
         // Compute each entry of d(Prc)_dJ.
         for (int rr = 0; rr < dim; rr++)
         {
            for (int cc = 0; cc < dim; cc++)
            {
               const double entry_rr_cc =
                  dI1_dMdM(rr,cc) * (I2 + iI2) +
                  dI1_dM(r,c) * dI2_dM(rr,cc) * (1.0 - iI2 * iI2) +
                  dI1_dM(rr,cc) * dI2_dM(r,c) * (1.0 - iI2 * iI2) +
                  I1 * ( dI2_dMdM(rr,cc) * (1.0 - iI2 * iI2) +
                         dI2_dM(r,c) * iI2 * iI2 * iI2 * dI2_dM(rr,cc));

               for (int i = 0; i < dof; i++)
               {
                  for (int j = 0; j < dof; j++)
                  {
                     A(i+r*dof, j+rr*dof) +=
                        weight * DS(i, c) * DS(j, cc) * entry_rr_cc;
                  }
               }
            }
         }
      }
   }
}

double TMOP_Metric_009::EvalW(const DenseMatrix &Jpt) const
{
   const double I2 = Dim2Invariant2(Jpt);
   return I2*( Dim2Invariant1(Jpt) * (I2 + 1.0 / I2) - 4.0);
}

void TMOP_Metric_009::EvalP(const DenseMatrix &Jpt,
                            DenseMatrix &P) const
{
   const double I1 = Dim2Invariant1(Jpt), I2 = Dim2Invariant2(Jpt);
   Dim2Invariant1_dM(Jpt, P);
   P *= (I2*I2 + 1.0);

   DenseMatrix PP(P.Size());
   Dim2Invariant2_dM(Jpt, PP);
   PP *= (2.*I1*I2 - 4.);

   P += PP;
}

void TMOP_Metric_009::AssembleH(const DenseMatrix &Jpt,
                                const DenseMatrix &DS,
                                const double weight,
                                DenseMatrix &A) const
{
   const int dof = DS.Height(), dim = DS.Width();
   const double I1 = Dim2Invariant1(Jpt), I2 = Dim2Invariant2(Jpt);
   DenseMatrix dI1_dM(dim), dI1_dMdM(dim), dI2_dM(dim), dI2_dMdM(dim);
   Dim2Invariant1_dM(Jpt, dI1_dM);
   Dim2Invariant2_dM(Jpt, dI2_dM);

   // The first two go over the rows and cols of dP_dJ where P = dW_dJ.
   for (int r = 0; r < dim; r++)
   {
      for (int c = 0; c < dim; c++)
      {
         Dim2Invariant1_dMdM(Jpt, r, c, dI1_dMdM);
         Dim2Invariant2_dMdM(Jpt, r, c, dI2_dMdM);
         // Compute each entry of d(Prc)_dJ.
         for (int rr = 0; rr < dim; rr++)
         {
            for (int cc = 0; cc < dim; cc++)
            {
               const double entry_rr_cc = dI1_dMdM(rr,cc) * (I2*I2 + 1.)
                                          + 2.*I1*dI2_dM(rr,cc) * dI2_dM(r,c)
                                          + 2.*I2*dI1_dM(rr,cc) * dI2_dM(r,c)
                                          + 2.*I1*I2*dI2_dMdM(rr,cc)
                                          - 4.*dI2_dMdM(rr,cc);

               for (int i = 0; i < dof; i++)
               {
                  for (int j = 0; j < dof; j++)
                  {
                     A(i+r*dof, j+rr*dof) +=
                        weight * DS(i, c) * DS(j, cc) * entry_rr_cc;
                  }
               }
            }
         }
      }
   }
}

double TMOP_Metric_022::EvalW(const DenseMatrix &Jpt) const
{
   const double I1 = Dim2Invariant1(Jpt), I2 = Dim2Invariant2(Jpt);
   return  0.5 * (I1*I2 - 2.*I2) / (I2 - tau0);
}

void TMOP_Metric_022::EvalP(const DenseMatrix &Jpt,
                            DenseMatrix &P) const
{
   const double I1 = Dim2Invariant1(Jpt), I2 = Dim2Invariant2(Jpt);
   double alpha = (2.*I2 - 2.*tau0);

   Dim2Invariant1_dM(Jpt, P);
   P *= I2*alpha;

   DenseMatrix PP(P.Size());
   Dim2Invariant2_dM(Jpt, PP);
   PP *= (4.*tau0 - 2.*I1*tau0);

   P += PP;
   P *= 1./(alpha*alpha);
}

void TMOP_Metric_022::AssembleH(const DenseMatrix &Jpt,
                                const DenseMatrix &DS,
                                const double weight,
                                DenseMatrix &A) const
{
   const int dof = DS.Height(), dim = DS.Width();
   const double I1 = Dim2Invariant1(Jpt), I2 = Dim2Invariant2(Jpt);
   DenseMatrix dI1_dM(dim), dI1_dMdM(dim), dI2_dM(dim), dI2_dMdM(dim);
   Dim2Invariant1_dM(Jpt, dI1_dM);
   Dim2Invariant2_dM(Jpt, dI2_dM);
   double alpha = (2.*I2 - 2.*tau0);

   // The first two go over the rows and cols of dP_dJ where P = dW_dJ.
   for (int r = 0; r < dim; r++)
   {
      for (int c = 0; c < dim; c++)
      {
         Dim2Invariant1_dMdM(Jpt, r, c, dI1_dMdM);
         Dim2Invariant2_dMdM(Jpt, r, c, dI2_dMdM);
         // Compute each entry of d(Prc)_dJ.
         for (int rr = 0; rr < dim; rr++)
         {
            for (int cc = 0; cc < dim; cc++)
            {
               const double entry_rr_cc =
                  ( alpha*alpha*
                    (4.*dI2_dM(rr,cc)*I2*dI1_dM(r,c)
                     + 2.*I2*I2*dI1_dMdM(rr,cc)
                     - 2.*tau0*dI2_dM(rr,cc)*dI1_dM(r,c)
                     - 2.*tau0*I2*dI1_dMdM(rr,cc)
                     + 4.*tau0*dI2_dMdM(rr,cc)
                     - 2.*tau0*dI1_dM(rr,cc)*dI2_dM(r,c)
                     - 2.*tau0*I1*dI2_dMdM(rr,cc))
                    - (2.*I2*I2*dI1_dM(r,c)
                       - 2.*tau0*I2*dI1_dM(r,c)
                       + 4.*tau0*dI2_dM(r,c)
                       - 2.*I1*tau0*dI2_dM(r,c))*
                    (4.*alpha*dI2_dM(rr,cc)) ) / (alpha*alpha*alpha*alpha);

               for (int i = 0; i < dof; i++)
               {
                  for (int j = 0; j < dof; j++)
                  {
                     A(i+r*dof, j+rr*dof) +=
                        weight * DS(i, c) * DS(j, cc) * entry_rr_cc;
                  }
               }
            }
         }
      }
   }
}

double TMOP_Metric_050::EvalW(const DenseMatrix &Jpt) const
{
   const double I1 = Dim2Invariant1(Jpt);
   return (I1*I1*0.5 - 2.);
}

void TMOP_Metric_050::EvalP(const DenseMatrix &Jpt,
                            DenseMatrix &P) const
{
   const double I1 = Dim2Invariant1(Jpt);
   Dim2Invariant1_dM(Jpt, P);
   P *= I1;
}

void TMOP_Metric_050::AssembleH(const DenseMatrix &Jpt,
                                const DenseMatrix &DS,
                                const double weight,
                                DenseMatrix &A) const
{
   const int dof = DS.Height(), dim = DS.Width();
   const double I1 = Dim2Invariant1(Jpt);
   DenseMatrix dI1_dM(dim), dI1_dMdM(dim);
   Dim2Invariant1_dM(Jpt, dI1_dM);

   // The first two go over the rows and cols of dP_dJ where P = dW_dJ.
   for (int r = 0; r < dim; r++)
   {
      for (int c = 0; c < dim; c++)
      {
         Dim2Invariant1_dMdM(Jpt, r, c, dI1_dMdM);
         // Compute each entry of d(Prc)_dJ.
         for (int rr = 0; rr < dim; rr++)
         {
            for (int cc = 0; cc < dim; cc++)
            {
               const double entry_rr_cc = dI1_dM(rr,cc) * dI1_dM(r,c)
                                          + I1 * dI1_dMdM(rr,cc);
               for (int i = 0; i < dof; i++)
               {
                  for (int j = 0; j < dof; j++)
                  {
                     A(i+r*dof, j+rr*dof) +=
                        weight * DS(i, c) * DS(j, cc) * entry_rr_cc;
                  }
               }
            }
         }
      }
   }
}

double TMOP_Metric_052::EvalW(const DenseMatrix &Jpt) const
{
   const double I2 = Dim2Invariant2(Jpt);
   return  (I2*I2 - 2.*I2 + 1.) / (2.*I2 - 2.*tau0);
}

void TMOP_Metric_052::EvalP(const DenseMatrix &Jpt,
                            DenseMatrix &P) const
{
   const double I2 = Dim2Invariant2(Jpt);
   double alpha = (2.*I2 - 2.*tau0);
   double alphainv = 1./alpha;

   Dim2Invariant2_dM(Jpt, P);
   P *= (2*I2*I2 - 4.*tau0*I2 + 4.*tau0 - 2.)*alphainv*alphainv;
}

void TMOP_Metric_052::AssembleH(const DenseMatrix &Jpt,
                                const DenseMatrix &DS,
                                const double weight,
                                DenseMatrix &A) const
{
   const int dof = DS.Height(), dim = DS.Width();
   const double I2 = Dim2Invariant2(Jpt);
   DenseMatrix dI2_dM(dim), dI2_dMdM(dim);
   Dim3Invariant2_dM(Jpt, dI2_dM);
   double alpha = (2.*I2 - 2.*tau0);

   // The first two go over the rows and cols of dP_dJ where P = dW_dJ.
   for (int r = 0; r < dim; r++)
   {
      for (int c = 0; c < dim; c++)
      {
         Dim2Invariant2_dMdM(Jpt, r, c, dI2_dMdM);
         // Compute each entry of d(Prc)_dJ.
         for (int rr = 0; rr < dim; rr++)
         {
            for (int cc = 0; cc < dim; cc++)
            {
               const double entry_rr_cc =
                  ( alpha*alpha*
                    (4.*I2*dI2_dM(rr,cc)*dI2_dM(r,c)
                     - 4.*tau0*dI2_dM(r,c)*dI2_dM(rr,cc)
                     + 2.*I2*I2*dI2_dMdM(rr,cc)
                     - 4.*tau0*I2*dI2_dMdM(rr,cc)
                     + 4.*tau0*dI2_dMdM(rr,cc)
                     - 2.*dI2_dMdM(rr,cc))
                    - 4.*alpha*dI2_dM(rr,cc) *
                    (2*I2*I2-4.*tau0*I2 + 4.*tau0-2.) * dI2_dM(r,c))
                  / (alpha*alpha*alpha*alpha);

               for (int i = 0; i < dof; i++)
               {
                  for (int j = 0; j < dof; j++)
                  {
                     A(i+r*dof, j+rr*dof) +=
                        weight * DS(i, c) * DS(j, cc) * entry_rr_cc;
                  }
               }
            }
         }
      }
   }
}

double TMOP_Metric_055::EvalW(const DenseMatrix &Jpt) const
{
   const double I2 = Dim2Invariant2(Jpt);
   return  I2*I2 - 2.*I2 + 1;
}

void TMOP_Metric_055::EvalP(const DenseMatrix &Jpt,
                            DenseMatrix &P) const
{
   const double I2 = Dim2Invariant2(Jpt);

   Dim2Invariant2_dM(Jpt, P);
   P *= (2.*I2 - 1.);
}

void TMOP_Metric_055::AssembleH(const DenseMatrix &Jpt,
                                const DenseMatrix &DS,
                                const double weight,
                                DenseMatrix &A) const
{
   const int dof = DS.Height(), dim = DS.Width();
   const double I2 = Dim2Invariant2(Jpt);
   DenseMatrix dI2_dM(dim), dI2_dMdM(dim);
   Dim2Invariant2_dM(Jpt, dI2_dM);

   // The first two go over the rows and cols of dP_dJ where P = dW_dJ.
   for (int r = 0; r < dim; r++)
   {
      for (int c = 0; c < dim; c++)
      {
         Dim2Invariant2_dMdM(Jpt, r, c, dI2_dMdM);
         // Compute each entry of d(Prc)_dJ.
         for (int rr = 0; rr < dim; rr++)
         {
            for (int cc = 0; cc < dim; cc++)
            {
               const double entry_rr_cc = 2.*dI2_dMdM(rr,cc)*(I2 - 1.)
                                          + 2.*dI2_dM(rr,cc)*dI2_dM(r,c);
               for (int i = 0; i < dof; i++)
               {
                  for (int j = 0; j < dof; j++)
                  {
                     A(i+r*dof, j+rr*dof) +=
                        weight * DS(i, c) * DS(j, cc) * entry_rr_cc;
                  }
               }
            }
         }
      }
   }
}

double TMOP_Metric_056::EvalW(const DenseMatrix &Jpt) const
{
   const double I2 = Dim2Invariant2(Jpt);
   return  0.5*(I2 + (1./I2)) - 1.;
}

void TMOP_Metric_056::EvalP(const DenseMatrix &Jpt,
                            DenseMatrix &P) const
{
   const double I2 = Dim2Invariant2(Jpt);
   double alpha = 1./(I2*I2);

   Dim2Invariant2_dM(Jpt, P);
   P *= (0.5 - 0.5*alpha);
}

void TMOP_Metric_056::AssembleH(const DenseMatrix &Jpt,
                                const DenseMatrix &DS,
                                const double weight,
                                DenseMatrix &A) const
{
   const int dof = DS.Height(), dim = DS.Width();
   const double I2 = Dim2Invariant2(Jpt);
   DenseMatrix dI2_dM(dim), dI2_dMdM(dim);
   Dim2Invariant2_dM(Jpt, dI2_dM);
   double alpha = 1./(I2*I2*I2);

   // The first two go over the rows and cols of dP_dJ where P = dW_dJ.
   for (int r = 0; r < dim; r++)
   {
      for (int c = 0; c < dim; c++)
      {
         Dim2Invariant2_dMdM(Jpt, r, c, dI2_dMdM);
         // Compute each entry of d(Prc)_dJ.
         for (int rr = 0; rr < dim; rr++)
         {
            for (int cc = 0; cc < dim; cc++)
            {
               const double entry_rr_cc =
                  alpha * dI2_dM(rr,cc) * dI2_dM(r,c) +
                  dI2_dMdM(rr,cc) * (0.5 - 0.5/(I2*I2));

               for (int i = 0; i < dof; i++)
               {
                  for (int j = 0; j < dof; j++)
                  {
                     A(i+r*dof, j+rr*dof) +=
                        weight * DS(i, c) * DS(j, cc) * entry_rr_cc;
                  }
               }
            }
         }
      }
   }
}

double TMOP_Metric_058::EvalW(const DenseMatrix &Jpt) const
{
   const double I1 = Dim2Invariant1(Jpt);
   return  I1*(I1-2.);
}

void TMOP_Metric_058::EvalP(const DenseMatrix &Jpt,
                            DenseMatrix &P) const
{
   const double I1 = Dim2Invariant1(Jpt);

   Dim2Invariant2_dM(Jpt, P);
   P *= (2.*I1 - 2.);
}

void TMOP_Metric_058::AssembleH(const DenseMatrix &Jpt,
                                const DenseMatrix &DS,
                                const double weight,
                                DenseMatrix &A) const
{
   const int dof = DS.Height(), dim = DS.Width();
   const double I1 = Dim2Invariant1(Jpt);
   DenseMatrix dI1_dM(dim), dI1_dMdM(dim);
   Dim2Invariant1_dM(Jpt, dI1_dM);

   // The first two go over the rows and cols of dP_dJ where P = dW_dJ.
   for (int r = 0; r < dim; r++)
   {
      for (int c = 0; c < dim; c++)
      {
         Dim2Invariant1_dMdM(Jpt, r, c, dI1_dMdM);
         // Compute each entry of d(Prc)_dJ.
         for (int rr = 0; rr < dim; rr++)
         {
            for (int cc = 0; cc < dim; cc++)
            {
               const double entry_rr_cc =
                  2. * dI1_dM(rr,cc) * dI1_dM(r,c) +
                  dI1_dMdM(rr,cc) * (2.*I1 - 2.);

               for (int i = 0; i < dof; i++)
               {
                  for (int j = 0; j < dof; j++)
                  {
                     A(i+r*dof, j+rr*dof) +=
                        weight * DS(i, c) * DS(j, cc) * entry_rr_cc;
                  }
               }
            }
         }
      }
   }
}

double TMOP_Metric_077::EvalW(const DenseMatrix &Jpt) const
{
   const double I2 = Dim2Invariant2(Jpt);
   return  0.5*(I2*I2 + 1./(I2*I2) - 2.);
}

void TMOP_Metric_077::EvalP(const DenseMatrix &Jpt,
                            DenseMatrix &P) const
{
   const double I2 = Dim2Invariant2(Jpt);
   double alpha = 1./(I2*I2*I2);

   Dim2Invariant2_dM(Jpt, P);
   P *= (I2 - alpha);
}

void TMOP_Metric_077::AssembleH(const DenseMatrix &Jpt,
                                const DenseMatrix &DS,
                                const double weight,
                                DenseMatrix &A) const
{
   const int dof = DS.Height(), dim = DS.Width();
   const double I2 = Dim2Invariant2(Jpt);
   DenseMatrix dI2_dM(dim), dI2_dMdM(dim);
   Dim2Invariant2_dM(Jpt, dI2_dM);
   double alpha = 1./(I2*I2*I2);
   double alpha2 = 1./(I2*I2*I2*I2);

   // The first two go over the rows and cols of dP_dJ where P = dW_dJ.
   for (int r = 0; r < dim; r++)
   {
      for (int c = 0; c < dim; c++)
      {
         Dim2Invariant2_dMdM(Jpt, r, c, dI2_dMdM);
         // Compute each entry of d(Prc)_dJ.
         for (int rr = 0; rr < dim; rr++)
         {
            for (int cc = 0; cc < dim; cc++)
            {
               const double entry_rr_cc =
                  dI2_dMdM(rr,cc)*(I2 - alpha)
                  + dI2_dM(rr,cc)*dI2_dM(r,c)
                  + 3.*alpha2*dI2_dM(rr,cc)*dI2_dM(r,c);

               for (int i = 0; i < dof; i++)
               {
                  for (int j = 0; j < dof; j++)
                  {
                     A(i+r*dof, j+rr*dof) +=
                        weight * DS(i, c) * DS(j, cc) * entry_rr_cc;
                  }
               }
            }
         }
      }
   }
}

double TMOP_Metric_211::EvalW(const DenseMatrix &Jpt) const
{
   const double det = Dim2Invariant2(Jpt);
   return (det*det) -3.*det + sqrt(det*det + eps) + 1.;
}

void TMOP_Metric_211::EvalP(const DenseMatrix &Jpt,
                            DenseMatrix &P) const
{
   const double det = Dim2Invariant2(Jpt);
   const double alpha = det/sqrt(det*det + eps);

   Dim2Invariant2_dM(Jpt, P);
   P *= (2.*det - 3. + alpha);
}

void TMOP_Metric_211::AssembleH(const DenseMatrix &Jpt,
                                const DenseMatrix &DS,
                                const double weight,
                                DenseMatrix &A) const
{
   const int dof = DS.Height(), dim = DS.Width();
   const double I2 = Dim2Invariant2(Jpt);
   DenseMatrix dI2_dM(dim), dI2_dMdM(dim);
   Dim2Invariant2_dM(Jpt, dI2_dM);
   double det = I2;
   double alpha = det / sqrt(det * det + eps);
   double alpha2 = pow(det * det + eps, 1.5);

   // The first two go over the rows and cols of dP_dJ where P = dW_dJ.
   for (int r = 0; r < dim; r++)
   {
      for (int c = 0; c < dim; c++)
      {
         Dim2Invariant2_dMdM(Jpt, r, c, dI2_dMdM);
         // Compute each entry of d(Prc)_dJ.
         for (int rr = 0; rr < dim; rr++)
         {
            for (int cc = 0; cc < dim; cc++)
            {
               const double entry_rr_cc =
                  2.*I2*dI2_dMdM(rr,cc)
                  - 3.*dI2_dMdM(rr,cc)
                  + alpha*dI2_dMdM(rr,cc)
                  + 2.*dI2_dM(rr,cc)*dI2_dM(r,c)
                  + dI2_dM(rr,cc)*dI2_dM(r,c)*1./(sqrt(det*det+eps))
                  - (I2*I2/alpha2) * dI2_dM(r,c) * dI2_dM(rr,cc);

               for (int i = 0; i < dof; i++)
               {
                  for (int j = 0; j < dof; j++)
                  {
                     A(i+r*dof, j+rr*dof) +=
                        weight * DS(i, c) * DS(j, cc) * entry_rr_cc;
                  }
               }
            }
         }
      }
   }
}

double TMOP_Metric_301::EvalW(const DenseMatrix &Jpt) const
{
#ifdef MFEM_USE_OLD_TMOP_INVARIANTS
   const double I1 = Dim3Invariant1(Jpt), I2 = Dim3Invariant2(Jpt);
   return  pow(I1*I2,0.5)/3. - 1.;
#else
   ie.SetJacobian(Jpt.GetData());
   return std::sqrt(ie.Get_I1b()*ie.Get_I2b())/3. - 1.;
#endif
}

void TMOP_Metric_301::EvalP(const DenseMatrix &Jpt,
                            DenseMatrix &P) const
{
#ifdef MFEM_USE_OLD_TMOP_INVARIANTS
   const double I1 = Dim3Invariant1(Jpt), I2 = Dim3Invariant2(Jpt);
   double I1I2 = I1*I2;

   Dim3Invariant1_dM(Jpt, P);
   DenseMatrix PP(P.Size());
   Dim3Invariant2_dM(Jpt, PP);

   P *= I2;
   PP *= (I1);
   P += PP;
   P *= (1./6.)*pow(I1I2,-0.5);
#else
   //  W = (1/3)*sqrt(I1b*I2b) - 1
   // dW = (1/6)/sqrt(I1b*I2b)*[I2b*dI1b + I1b*dI2b]
   ie.SetJacobian(Jpt.GetData());
   const double a = 1./(6.*std::sqrt(ie.Get_I1b()*ie.Get_I2b()));
   Add(a*ie.Get_I2b(), ie.Get_dI1b(), a*ie.Get_I1b(), ie.Get_dI2b(), P);
#endif
}

void TMOP_Metric_301::AssembleH(const DenseMatrix &Jpt,
                                const DenseMatrix &DS,
                                const double weight,
                                DenseMatrix &A) const
{
#ifdef MFEM_USE_OLD_TMOP_INVARIANTS
   const int dof = DS.Height(), dim = DS.Width();
   const double I1 = Dim3Invariant1(Jpt), I2 = Dim3Invariant2(Jpt);
   DenseMatrix dI1_dM(dim), dI1_dMdM(dim), dI2_dM(dim), dI2_dMdM(dim);
   Dim3Invariant1_dM(Jpt, dI1_dM);
   Dim3Invariant2_dM(Jpt, dI2_dM);
   double I1I2 = I1*I2;

   // The first two go over the rows and cols of dP_dJ where P = dW_dJ.
   for (int r = 0; r < dim; r++)
   {
      for (int c = 0; c < dim; c++)
      {
         Dim3Invariant1_dMdM(Jpt, r, c, dI1_dMdM);
         Dim3Invariant2_dMdM(Jpt, r, c, dI2_dMdM);
         // Compute each entry of d(Prc)_dJ.
         for (int rr = 0; rr < dim; rr++)
         {
            for (int cc = 0; cc < dim; cc++)
            {
               const double entry_rr_cc =
                  - (1./12.)*pow(I1I2,-1.5) *
                  (dI1_dM(rr,cc)*I2 + I1*dI2_dM(rr,cc)) *
                  (dI1_dM(r,c)*I2 + I1*dI2_dM(r,c))
                  + (1./6)*pow(I1I2,-0.5) *
                  (dI1_dMdM(rr,cc)*I2
                   + dI1_dM(r,c)*dI2_dM(rr,cc)
                   + dI1_dM(rr,cc)*dI2_dM(r,c)
                   + I1*dI2_dMdM(rr,cc));

               for (int i = 0; i < dof; i++)
               {
                  for (int j = 0; j < dof; j++)
                  {
                     A(i+r*dof, j+rr*dof) +=
                        weight * DS(i, c) * DS(j, cc) * entry_rr_cc;
                  }
               }
            }
         }
      }
   }
#else
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
#endif
}

double TMOP_Metric_302::EvalW(const DenseMatrix &Jpt) const
{
   const double I1 = Dim3Invariant1(Jpt), I2 = Dim3Invariant2(Jpt);
   return  I1*I2/9. - 1;
}

void TMOP_Metric_302::EvalP(const DenseMatrix &Jpt,
                            DenseMatrix &P) const
{
   const double I1 = Dim3Invariant1(Jpt), I2 = Dim3Invariant2(Jpt);
   Dim3Invariant1_dM(Jpt, P);
   P *= I2;
   DenseMatrix PP(P.Size());
   Dim3Invariant2_dM(Jpt, PP);
   PP *= (I1);
   P += PP;
   P *= 1.0 / 9.0;
}

void TMOP_Metric_302::AssembleH(const DenseMatrix &Jpt,
                                const DenseMatrix &DS,
                                const double weight,
                                DenseMatrix &A) const
{
   const int dof = DS.Height(), dim = DS.Width();
   const double I1 = Dim3Invariant1(Jpt), I2 = Dim3Invariant2(Jpt);
   DenseMatrix dI1_dM(dim), dI1_dMdM(dim), dI2_dM(dim), dI2_dMdM(dim);
   Dim3Invariant1_dM(Jpt, dI1_dM);
   Dim3Invariant2_dM(Jpt, dI2_dM);

   // The first two go over the rows and cols of dP_dJ where P = dW_dJ.
   for (int r = 0; r < dim; r++)
   {
      for (int c = 0; c < dim; c++)
      {
         Dim3Invariant1_dMdM(Jpt, r, c, dI1_dMdM);
         Dim3Invariant2_dMdM(Jpt, r, c, dI2_dMdM);
         // Compute each entry of d(Prc)_dJ.
         for (int rr = 0; rr < dim; rr++)
         {
            for (int cc = 0; cc < dim; cc++)
            {
               const double entry_rr_cc =
                  (1./9.)*(dI1_dMdM(rr,cc)*I2
                           + dI1_dM(r,c)*dI2_dM(rr,cc)
                           + dI1_dM(rr,cc)*dI2_dM(r,c)
                           + dI2_dMdM(rr,cc)*I1);
               for (int i = 0; i < dof; i++)
               {
                  for (int j = 0; j < dof; j++)
                  {
                     A(i+r*dof, j+rr*dof) +=
                        weight * DS(i, c) * DS(j, cc) * entry_rr_cc;
                  }
               }
            }
         }
      }
   }
}

double TMOP_Metric_303::EvalW(const DenseMatrix &Jpt) const
{
#ifdef MFEM_USE_OLD_TMOP_INVARIANTS
   const double I1 = Dim3Invariant1(Jpt);
   return I1/3. - 1.0;
#else
   ie.SetJacobian(Jpt.GetData());
   return ie.Get_I1b()/3.0 - 1.0;
#endif
}

void TMOP_Metric_303::EvalP(const DenseMatrix &Jpt,
                            DenseMatrix &P) const
{
#ifdef MFEM_USE_OLD_TMOP_INVARIANTS
   Dim3Invariant1_dM(Jpt, P);
   P *= 1.0 / 3.0;
#else
   ie.SetJacobian(Jpt.GetData());
   P = ie.Get_dI1b();
   P *= 1./3.;
#endif
}

void TMOP_Metric_303::AssembleH(const DenseMatrix &Jpt,
                                const DenseMatrix &DS,
                                const double weight,
                                DenseMatrix &A) const
{
#ifdef MFEM_USE_OLD_TMOP_INVARIANTS
   const int dof = DS.Height(), dim = DS.Width();
   DenseMatrix dI1_dMdM(dim);

   // The first two go over the rows and cols of dP_dJ where P = dW_dJ.
   for (int r = 0; r < dim; r++)
   {
      for (int c = 0; c < dim; c++)
      {
         Dim3Invariant1_dMdM(Jpt, r, c, dI1_dMdM);
         // Compute each entry of d(Prc)_dJ.
         for (int rr = 0; rr < dim; rr++)
         {
            for (int cc = 0; cc < dim; cc++)
            {
               const double entry_rr_cc = (1./3.)*dI1_dMdM(rr,cc);

               for (int i = 0; i < dof; i++)
               {
                  for (int j = 0; j < dof; j++)
                  {
                     A(i+r*dof, j+rr*dof) +=
                        weight * DS(i, c) * DS(j, cc) * entry_rr_cc;
                  }
               }
            }
         }
      }
   }
#else
   ie.SetJacobian(Jpt.GetData());
   ie.SetDerivativeMatrix(DS.Height(), DS.GetData());
   ie.Assemble_ddI1b(weight/3., A.GetData());
#endif
}

double TMOP_Metric_315::EvalW(const DenseMatrix &Jpt) const
{
   const double I3 = Dim3Invariant3(Jpt);
   return  I3*I3 - 2.*I3 + 1;
}

void TMOP_Metric_315::EvalP(const DenseMatrix &Jpt,
                            DenseMatrix &P) const
{
   const double I3 = Dim3Invariant3(Jpt);

   Dim3Invariant3_dM(Jpt, P);
   P *= (2.*I3 - 1.);
}

void TMOP_Metric_315::AssembleH(const DenseMatrix &Jpt,
                                const DenseMatrix &DS,
                                const double weight,
                                DenseMatrix &A) const
{
   const int dof = DS.Height(), dim = DS.Width();
   const double I3 = Dim3Invariant3(Jpt);
   DenseMatrix dI3_dM(dim), dI3_dMdM(dim);
   Dim3Invariant3_dM(Jpt, dI3_dM);

   // The first two go over the rows and cols of dP_dJ where P = dW_dJ.
   for (int r = 0; r < dim; r++)
   {
      for (int c = 0; c < dim; c++)
      {
         Dim3Invariant3_dMdM(Jpt, r, c, dI3_dMdM);
         // Compute each entry of d(Prc)_dJ.
         for (int rr = 0; rr < dim; rr++)
         {
            for (int cc = 0; cc < dim; cc++)
            {
               const double entry_rr_cc = 2.*dI3_dMdM(rr,cc) * (I3 - 1.)
                                          + 2.*dI3_dM(rr,cc) * dI3_dM(r,c);
               for (int i = 0; i < dof; i++)
               {
                  for (int j = 0; j < dof; j++)
                  {
                     A(i+r*dof, j+rr*dof) +=
                        weight * DS(i, c) * DS(j, cc) * entry_rr_cc;
                  }
               }
            }
         }
      }
   }
}

double TMOP_Metric_316::EvalW(const DenseMatrix &Jpt) const
{
   const double I3 = Dim3Invariant3(Jpt);
   return  0.5*(I3 + (1./I3)) - 1.;
}

void TMOP_Metric_316::EvalP(const DenseMatrix &Jpt,
                            DenseMatrix &P) const
{
   const double I3 = Dim3Invariant3(Jpt);
   const double alpha = 1./(I3*I3);

   Dim3Invariant3_dM(Jpt, P);
   P *= (0.5 - 0.5*alpha);
}

void TMOP_Metric_316::AssembleH(const DenseMatrix &Jpt,
                                const DenseMatrix &DS,
                                const double weight,
                                DenseMatrix &A) const
{
   const int dof = DS.Height(), dim = DS.Width();
   const double I3 = Dim3Invariant3(Jpt);
   DenseMatrix dI3_dM(dim), dI3_dMdM(dim);
   Dim3Invariant3_dM(Jpt, dI3_dM);
   double alpha = 1./(I3*I3*I3);

   // The first two go over the rows and cols of dP_dJ where P = dW_dJ.
   for (int r = 0; r < dim; r++)
   {
      for (int c = 0; c < dim; c++)
      {
         Dim3Invariant3_dMdM(Jpt, r, c, dI3_dMdM);
         // Compute each entry of d(Prc)_dJ.
         for (int rr = 0; rr < dim; rr++)
         {
            for (int cc = 0; cc < dim; cc++)
            {
               const double entry_rr_cc =
                  alpha * dI3_dM(rr,cc) * dI3_dM(r,c)
                  + dI3_dMdM(rr,cc) * (0.5 - 0.5/(I3*I3));

               for (int i = 0; i < dof; i++)
               {
                  for (int j = 0; j < dof; j++)
                  {
                     A(i+r*dof, j+rr*dof) +=
                        weight * DS(i, c) * DS(j, cc) * entry_rr_cc;
                  }
               }
            }
         }
      }
   }
}

double TMOP_Metric_321::EvalW(const DenseMatrix &Jpt) const
{
   const double I1 = Dim3Invariant1(Jpt),
                I2 = Dim3Invariant2(Jpt),
                I3 = Dim3Invariant3(Jpt);
   return  pow(I3,2./3.)*I1 + pow(I3,-2./3.)*I2 - 6.;
}

void TMOP_Metric_321::EvalP(const DenseMatrix &Jpt,
                            DenseMatrix &P) const
{
   const double I1 = Dim3Invariant1(Jpt),
                I2 = Dim3Invariant2(Jpt),
                I3 = Dim3Invariant3(Jpt);
   DenseMatrix PP(P.Size());
   DenseMatrix PPP(P.Size());

   Dim3Invariant1_dM(Jpt, P); //DI1/DM
   Dim3Invariant2_dM(Jpt, PP); //DI2/DM
   Dim3Invariant3_dM(Jpt, PPP); //DI3/DM

   P *= pow(I3,2./3.);
   PP *= pow(I3,-2./3.);
   PPP *= I1*(2./3.)*pow(I3,-1./3.) + I2*(-2./3.)*pow(I3,-5./3.);

   P += PP;
   P += PPP;
}

void TMOP_Metric_321::AssembleH(const DenseMatrix &Jpt,
                                const DenseMatrix &DS,
                                const double weight,
                                DenseMatrix &A) const
{
   const int dof = DS.Height(), dim = DS.Width();
   const double I1 = Dim3Invariant1(Jpt),
                I2 = Dim3Invariant2(Jpt),
                I3 = Dim3Invariant3(Jpt);
   DenseMatrix dI1_dM(dim), dI1_dMdM(dim), dI2_dM(dim),
               dI2_dMdM(dim), dI3_dM(dim), dI3_dMdM(dim);
   Dim3Invariant1_dM(Jpt, dI1_dM);
   Dim3Invariant2_dM(Jpt, dI2_dM);
   Dim3Invariant3_dM(Jpt, dI3_dM);
   double pwn43 =pow(I3,-4./3.);
   double pwn13 =pow(I3,-1./3.);
   double pwn53 =pow(I3,-5./3.);
   double pwn23 =pow(I3,-2./3.);
   double pwn83 =pow(I3,-8./3.);

   // The first two go over the rows and cols of dP_dJ where P = dW_dJ.
   for (int r = 0; r < dim; r++)
   {
      for (int c = 0; c < dim; c++)
      {
         Dim3Invariant1_dMdM(Jpt, r, c, dI1_dMdM);
         Dim3Invariant2_dMdM(Jpt, r, c, dI2_dMdM);
         Dim3Invariant3_dMdM(Jpt, r, c, dI3_dMdM);
         // Compute each entry of d(Prc)_dJ.
         for (int rr = 0; rr < dim; rr++)
         {
            for (int cc = 0; cc < dim; cc++)
            {
               const double entry_rr_cc =
                  - (2./9.)*pwn43*dI3_dM(rr,cc)*dI3_dM(r,c)*I1
                  + (2./3.)*pwn13*dI3_dMdM(rr,cc)*I1
                  + (2./3.)*pwn13*dI3_dM(r,c)*dI1_dM(rr,cc)
                  + (2./3.)*pwn13*dI3_dM(rr,cc)*dI1_dM(r,c)
                  + pow(I3,2./3.)*dI1_dMdM(rr,cc)
                  + (10./9.)*pwn83*dI3_dM(rr,cc)*dI3_dM(r,c)*I2
                  - (2./3.)*pwn53*dI3_dMdM(rr,cc)*I2
                  - (2./3.)*pwn53*dI3_dM(r,c)*dI2_dM(rr,cc)
                  - (2./3.)*pwn53*dI3_dM(rr,cc)*dI2_dM(r,c)
                  + pwn23*dI2_dMdM(rr,cc);

               for (int i = 0; i < dof; i++)
               {
                  for (int j = 0; j < dof; j++)
                  {
                     A(i+r*dof, j+rr*dof) +=
                        weight * DS(i, c) * DS(j, cc) * entry_rr_cc;
                  }
               }
            }
         }
      }
   }
}

double TMOP_Metric_352::EvalW(const DenseMatrix &Jpt) const
{
   const double I3 = Dim3Invariant3(Jpt);
   return  (I3*I3 - 2.*I3 + 1.)/(2.*I3 - 2.*tau0);
}

void TMOP_Metric_352::EvalP(const DenseMatrix &Jpt,
                            DenseMatrix &P) const
{
   const double I3 = Dim3Invariant3(Jpt);
   double alpha = (2.*I3 - 2.*tau0);
   double alphainv = 1./alpha;

   Dim3Invariant3_dM(Jpt, P);
   P *= (2*I3*I3 - 4.*tau0*I3 + 4.*tau0 - 2.)*alphainv*alphainv;
}

void TMOP_Metric_352::AssembleH(const DenseMatrix &Jpt,
                                const DenseMatrix &DS,
                                const double weight,
                                DenseMatrix &A) const
{
   const int dof = DS.Height(), dim = DS.Width();
   const double I3 = Dim3Invariant3(Jpt);
   DenseMatrix dI3_dM(dim), dI3_dMdM(dim);
   Dim3Invariant3_dM(Jpt, dI3_dM);
   double alpha = (2.*I3 - 2.*tau0);

   // The first two go over the rows and cols of dP_dJ where P = dW_dJ.
   for (int r = 0; r < dim; r++)
   {
      for (int c = 0; c < dim; c++)
      {
         Dim3Invariant3_dMdM(Jpt, r, c, dI3_dMdM);
         // Compute each entry of d(Prc)_dJ.
         for (int rr = 0; rr < dim; rr++)
         {
            for (int cc = 0; cc < dim; cc++)
            {
               const double entry_rr_cc =
                  ( alpha*alpha*
                    (4.*I3*dI3_dM(rr,cc)*dI3_dM(r,c)
                     + 2*I3*I3*dI3_dMdM(rr,cc)
                     - 4.*tau0*dI3_dM(r,c)*dI3_dM(rr,cc)
                     - 4.*tau0*I3*dI3_dMdM(rr,cc)
                     + 4.*tau0*dI3_dMdM(rr,cc)
                     - 2.*dI3_dMdM(rr,cc)) -
                    4.*alpha*dI3_dM(rr,cc) *
                    (2*I3*I3-4.*tau0*I3 + 4.*tau0-2.) * dI3_dM(r,c) )
                  / (alpha*alpha*alpha*alpha);

               for (int i = 0; i < dof; i++)
               {
                  for (int j = 0; j < dof; j++)
                  {
                     A(i+r*dof, j+rr*dof) +=
                        weight * DS(i, c) * DS(j, cc) * entry_rr_cc;
                  }
               }
            }
         }
      }
   }
}


void TargetJacobian::SetInitialNodes(const GridFunction &n0)
{
   nodes0 = &n0;

   // TODO: note that this uses the mesh to compute the area, not nodes0.
   // Average cell volume.
   L2_FECollection fec(0, nodes0->FESpace()->GetFE(0)->GetDim());
   FiniteElementSpace fes(nodes0->FESpace()->GetMesh(), &fec);
   LinearForm lf(&fes);
   ConstantCoefficient one(1.0);
   lf.AddDomainIntegrator(new DomainLFIntegrator(one));
   lf.Assemble();
#ifdef MFEM_USE_MPI
   if (serial_use)
   {
      avg_volume0 = lf.Sum() / nodes0->FESpace()->GetNE();
   }
   else
   {
      double area_NE[4];
      area_NE[0] = lf.Sum(); area_NE[1] = nodes0->FESpace()->GetNE();
      MPI_Allreduce(area_NE, area_NE + 2, 2, MPI_DOUBLE, MPI_SUM, comm);
      avg_volume0 = area_NE[2] / area_NE[3];
   }
#else
   avg_volume0 = lf.Sum() / nodes0->FESpace()->GetNE();
#endif
}

void TargetJacobian::ComputeElementTargets(int e_id, const FiniteElement &fe,
                                           const IntegrationRule &ir,
                                           DenseTensor &Jtr) const
{
   switch (target_type)
   {
      case CURRENT:
      case TARGET_MESH:
      case IDEAL_INIT_SIZE:
      {
         const GridFunction *nds;
         if (target_type == CURRENT)
         { MFEM_VERIFY(nodes, "Nodes are not set!");          nds = nodes; }
         else if (target_type == TARGET_MESH)
         { MFEM_VERIFY(tnodes, "Target nodes are not set!");  nds = tnodes; }
         else
         { MFEM_VERIFY(nodes0, "Initial nodes are not set!"); nds = nodes0; }

         const int dim = fe.GetDim(), dof = fe.GetDof();
         DenseMatrix dshape(dof, dim), pos(dof, dim);
         Array<int> xdofs(dof * dim);
         Vector posV(pos.Data(), dof * dim);

         DenseMatrix *Wideal = NULL;
         if (target_type == IDEAL_INIT_SIZE)
         {
            Wideal = new DenseMatrix(dim);
            ConstructIdealJ(fe.GetGeomType(), *Wideal);
         }

         nds->FESpace()->GetElementVDofs(e_id, xdofs);
         nds->GetSubVector(xdofs, posV);
         for (int i = 0; i < ir.GetNPoints(); i++)
         {
            fe.CalcDShape(ir.IntPoint(i), dshape);

            // W = Jac(ref->physical) for CURRENT and TARGET_MESH.
            MultAtB(pos, dshape, Jtr(i));

            if (target_type == IDEAL_INIT_SIZE)
            {
               double det = Jtr(i).Det();
               MFEM_VERIFY(det > 0.0, "Initial mesh is inverted!");
               Jtr(i) = *Wideal;
               Jtr(i) *= sqrt(det / Wideal->Det());
            }
         }
         delete Wideal;
         break;
      }
      case IDEAL:
      {
         DenseMatrix Wideal(fe.GetDim());
         ConstructIdealJ(fe.GetGeomType(), Wideal);
         for (int i = 0; i < ir.GetNPoints(); i++) { Jtr(i) = Wideal; }
         break;
      }
      case IDEAL_EQ_SIZE:
      {
         DenseMatrix Wideal(fe.GetDim());
         ConstructIdealJ(fe.GetGeomType(), Wideal);
         Wideal *= sqrt(size_scale * avg_volume0 / Wideal.Det());
         for (int i = 0; i < ir.GetNPoints(); i++) { Jtr(i) = Wideal; }
         break;
      }
   }
}

void TargetJacobian::ConstructIdealJ(int geom, DenseMatrix &J)
{
   switch (geom)
   {
      case Geometry::SQUARE:
      case Geometry::CUBE:
         J = 0.0;
         for (int i = 0; i < J.Size(); i++) { J(i, i) = 1.0; }
         break;
      case Geometry::TRIANGLE:
      {
         const double r3 = sqrt(3.0);
         J(0, 0) = 1.0; J(0, 1) = 0.5;
         J(1, 0) = 0.0; J(1, 1) = 0.5*r3;
         break;
      }
      case Geometry::TETRAHEDRON:
      {
         const double r3 = sqrt(3.0), r6 = sqrt(6.0);
         J(0, 0) = 1.0; J(0, 1) = 0.5;    J(0, 2) = 0.5;
         J(1, 0) = 0.0; J(1, 1) = 0.5*r3; J(1, 2) = 0.5*r3;
         J(2, 0) = 0.0; J(2, 1) = 0.0;    J(2, 2) = r6/3.0;
         break;
      }
   }
}


double TMOP_Integrator::GetElementEnergy(const FiniteElement &el,
                                         ElementTransformation &Ttr,
                                         const Vector &elfun)
{
   int dof = el.GetDof(), dim = el.GetDim();
   double energy;

   DSh.SetSize(dof, dim);
   Jrt.SetSize(dim);
   Jpr.SetSize(dim);
   Jpt.SetSize(dim);
   PMatI.UseExternalData(elfun.GetData(), dof, dim);

   const IntegrationRule *ir = IntRule;
   if (!ir)
   {
      ir = &(IntRules.Get(el.GetGeomType(), 2*el.GetOrder() + 3)); // <---
   }

   energy = 0.0;
   model->SetTransformation(Ttr);
   DenseTensor *Jtr = NULL;
   if (targetJ)
   {
      Jtr = new DenseTensor(dim, dim, ir->GetNPoints());
      targetJ->ComputeElementTargets(Ttr.ElementNo, el, *ir, *Jtr);
   }

   // Limited case.
   DenseMatrix *pos0 = NULL;
   if (limited)
   {
      pos0 = new DenseMatrix(dof, dim);
      Vector pos0V(pos0->Data(), dof * dim);
      Array<int> pos_dofs;
      nodes0->FESpace()->GetElementVDofs(Ttr.ElementNo, pos_dofs);
      nodes0->GetSubVector(pos_dofs, pos0V);
   }

   // Define ref->physical transformation, for the case when coeff is used.
   IsoparametricTransformation *Tpr = NULL;
   if (coeff)
   {
      Tpr = new IsoparametricTransformation;
      Tpr->SetFE(&el);
      Tpr->ElementNo = Ttr.ElementNo;
      Tpr->Attribute = Ttr.Attribute;
      Tpr->GetPointMat().SetSize(dim, dof);
      for (int i = 0; i < dof; i++)
      {
         for (int d = 0; d < dim; d++)
         {
            Tpr->GetPointMat()(d, i) = PMatI(i, d);
         }
      }
   }

   double weight;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      if (targetJ)
      {
         const DenseMatrix &Jtr_i = (*Jtr)(i);
         model->SetTargetJacobian(Jtr_i);
         CalcInverse(Jtr_i, Jrt);
         weight = Jtr_i.Det();
      }
      else
      {
         Ttr.SetIntPoint(&ip);
         CalcInverse(Ttr.Jacobian(), Jrt);
         weight = Ttr.Weight();
      }

      el.CalcDShape(ip, DSh);
      MultAtB(PMatI, DSh, Jpr);
      Mult(Jpr, Jrt, Jpt);

      double val = model->EvalW(Jpt);
      if (limited)
      {
         val *= eps;
         Vector shape(dof), p(dim), p0(dim);
         el.CalcShape(ip, shape);
         PMatI.MultTranspose(shape, p);
         pos0->MultTranspose(shape, p0);
         for (int d = 0; d < dim; d++)
         {
            const double diff = p(d) - p0(d);
            val += 0.5 * diff * diff;
         }
      }

      if (coeff) { weight *= coeff->Eval(*Tpr, ip); }
      energy += ip.weight * weight * val;
   }
   delete Tpr;
   delete pos0;
   delete Jtr;
   return energy;
}

void TMOP_Integrator::AssembleElementVector(const FiniteElement &el,
                                            ElementTransformation &Ttr,
                                            const Vector &elfun, Vector &elvect)
{
   int dof = el.GetDof(), dim = el.GetDim();

   DSh.SetSize(dof, dim);
   DS.SetSize(dof, dim);
   Jrt.SetSize(dim);
   Jpt.SetSize(dim);
   P.SetSize(dim);
   PMatI.UseExternalData(elfun.GetData(), dof, dim);
   elvect.SetSize(dof*dim);
   PMatO.UseExternalData(elvect.GetData(), dof, dim);

   const IntegrationRule *ir = IntRule;
   if (!ir)
   {
      ir = &(IntRules.Get(el.GetGeomType(), 2*el.GetOrder() + 3)); // <---
   }

   elvect = 0.0;
   model->SetTransformation(Ttr);
   DenseTensor *Jtr = NULL;
   if (targetJ)
   {
      Jtr = new DenseTensor(dim, dim, ir->GetNPoints());
      targetJ->ComputeElementTargets(Ttr.ElementNo, el, *ir, *Jtr);
   }

   // Limited case.
   DenseMatrix *pos0 = NULL;
   if (limited)
   {
      pos0 = new DenseMatrix(dof, dim);
      Vector pos0V(pos0->Data(), dof * dim);
      Array<int> pos_dofs;
      nodes0->FESpace()->GetElementVDofs(Ttr.ElementNo, pos_dofs);
      nodes0->GetSubVector(pos_dofs, pos0V);
   }

   // Define ref->physical transformation, for the case when coeff is used.
   IsoparametricTransformation *Tpr = NULL;
   if (coeff)
   {
      Tpr = new IsoparametricTransformation;
      Tpr->SetFE(&el);
      Tpr->ElementNo = Ttr.ElementNo;
      Tpr->Attribute = Ttr.Attribute;
      Tpr->GetPointMat().SetSize(dim, dof);
      for (int i = 0; i < dof; i++)
      {
         for (int d = 0; d < dim; d++)
         {
            Tpr->GetPointMat()(d, i) = PMatI(i, d);
         }
      }
   }

   double weight;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      if (targetJ)
      {
         const DenseMatrix &Jtr_i = (*Jtr)(i);
         model->SetTargetJacobian(Jtr_i);
         CalcInverse(Jtr_i, Jrt);
         weight = Jtr_i.Det();
      }
      else
      {
         Ttr.SetIntPoint(&ip);
         CalcInverse(Ttr.Jacobian(), Jrt);
         weight = Ttr.Weight();
      }

      el.CalcDShape(ip, DSh);
      Mult(DSh, Jrt, DS);
      MultAtB(PMatI, DS, Jpt);

      model->EvalP(Jpt, P);

      if (coeff) { weight *= coeff->Eval(*Tpr, ip); }

      P *= ip.weight * weight;
      if (limited) { P *= eps; }
      AddMultABt(DS, P, PMatO);

      if (limited)
      {
         Vector shape(dof), p(dim), p0(dim);
         el.CalcShape(ip, shape);
         PMatI.MultTranspose(shape, p);
         pos0->MultTranspose(shape, p0);
         for (int d = 0; d < dim; d++)
         {
            Vector s(shape);
            s *= ip.weight * weight * (p(d) - p0(d));
            Vector tmp;
            PMatO.GetColumnReference(d, tmp);
            tmp += s;
         }
      }
   }
   delete Tpr;
   delete pos0;
   delete Jtr;
}


void TMOP_Integrator::AssembleElementGrad(const FiniteElement &el,
                                          ElementTransformation &Ttr,
                                          const Vector &elfun,
                                          DenseMatrix &elmat)
{
   int dof = el.GetDof(), dim = el.GetDim();

   DSh.SetSize(dof, dim);
   DS.SetSize(dof, dim);
   Jrt.SetSize(dim);
   Jpt.SetSize(dim);
   PMatI.UseExternalData(elfun.GetData(), dof, dim);
   elmat.SetSize(dof*dim);

   const IntegrationRule *ir = IntRule;
   if (!ir)
   {
      ir = &(IntRules.Get(el.GetGeomType(), 2*el.GetOrder() + 3)); // <---
   }

   elmat = 0.0;
   DenseTensor *Jtr = NULL;
   if (targetJ)
   {
      Jtr = new DenseTensor(dim, dim, ir->GetNPoints());
      targetJ->ComputeElementTargets(Ttr.ElementNo, el, *ir, *Jtr);
   }

   // Define ref->physical transformation, for the case when coeff is used.
   IsoparametricTransformation *Tpr = NULL;
   if (coeff)
   {
      Tpr = new IsoparametricTransformation;
      Tpr->SetFE(&el);
      Tpr->ElementNo = Ttr.ElementNo;
      Tpr->Attribute = Ttr.Attribute;
      Tpr->GetPointMat().SetSize(dim, dof);
      for (int i = 0; i < dof; i++)
      {
         for (int d = 0; d < dim; d++)
         {
            Tpr->GetPointMat()(d, i) = PMatI(i, d);
         }
      }
   }

   if (!targetJ)
   {
      model->SetTransformation(Ttr); // FIXME
   }
   double weight;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      if (targetJ)
      {
         const DenseMatrix &Jtr_i = (*Jtr)(i);
         model->SetTargetJacobian(Jtr_i);
         CalcInverse(Jtr_i, Jrt);
         weight = Jtr_i.Det();
      }
      else
      {
         Ttr.SetIntPoint(&ip);
         CalcInverse(Ttr.Jacobian(), Jrt);
         weight = Ttr.Weight();
      }

      el.CalcDShape(ip, DSh);
      Mult(DSh, Jrt, DS);
      MultAtB(PMatI, DS, Jpt);

      if (coeff) { weight *= coeff->Eval(*Tpr, ip); }

      if (!limited)
      { model->AssembleH(Jpt, DS, ip.weight * weight, elmat); }
      else
      {
         model->AssembleH(Jpt, DS, eps * ip.weight * weight, elmat);
         Vector shape(dof);
         el.CalcShape(ip, shape);
         for (int i = 0; i < dof; i++)
         {
            for (int j = 0; j <= i; j++)
            {
               double a = shape(i) * shape(j) * ip.weight * weight;
               for (int d = 0; d < dim; d++)
               {
                  elmat(i+d*dof, j+d*dof) += a;
                  if (i != j) { elmat(j+d*dof, i+d*dof) += a; }
               }
            }
         }
      }
   }
   delete Tpr;
   delete Jtr;
}

TMOP_Integrator::~TMOP_Integrator()
{
   delete targetJ;
}


void InterpolateTMOP_QualityMetric(TMOP_QualityMetric &metric,
                                   const TargetJacobian &tj,
                                   const Mesh &mesh, GridFunction &gf)
{
   const int NE = mesh.GetNE();
   const GridFunction &nodes = *mesh.GetNodes();

   for (int i = 0; i < NE; i++)
   {
      const FiniteElement &fe_pos = *nodes.FESpace()->GetFE(i);
      const IntegrationRule &ir = gf.FESpace()->GetFE(i)->GetNodes();
      const int dim = fe_pos.GetDim(), nsp = ir.GetNPoints(),
                dof = fe_pos.GetDof();

      DenseTensor W(dim, dim, nsp);
      tj.ComputeElementTargets(i, fe_pos, ir, W);

      DenseMatrix dshape(dof, dim), Winv(dim), T(dim), A(dim), pos(dof, dim);
      Array<int> pos_dofs(dof * dim), gf_dofs(nsp);
      Vector posV(pos.Data(), dof * dim);

      gf.FESpace()->GetElementDofs(i, gf_dofs);
      nodes.FESpace()->GetElementVDofs(i, pos_dofs);
      nodes.GetSubVector(pos_dofs, posV);

      for (int j = 0; j < nsp; j++)
      {
         const DenseMatrix &Wj = W(j);
         metric.SetTargetJacobian(Wj);
         CalcInverse(Wj, Winv);

         const IntegrationPoint &ip = ir.IntPoint(j);
         fe_pos.CalcDShape(ip, dshape);
         MultAtB(pos, dshape, A);
         Mult(A, Winv, T);

         gf(gf_dofs[j]) = metric.EvalW(T);
      }
   }
}

} // namespace mfem
