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

#include "tmop.hpp"
#include "linearform.hpp"
#include "pgridfunc.hpp"
#include "tmop_tools.hpp"
#include "../general/forall.hpp"
#include "../linalg/dual.hpp"

namespace mfem
{

/* AD related definitions below ========================================*/

/// MFEM native AD-type for first derivatives
using AD1Type = future::dual<real_t, real_t>;
/// MFEM native AD-type for second derivatives
using AD2Type = future::dual<AD1Type, AD1Type>;

/*
Functions for 2x2 DenseMatrix cast as std::vector<type>, assuming column-major storage
*/
template <typename type>
type fnorm2_2D(const std::vector<type> &u)
{
   return u[0]*u[0] + u[1]*u[1] + u[2]*u[2] + u[3]*u[3];
}

template <typename type>
type fnorm2_3D(const std::vector<type> &u)
{
   return u[0]*u[0] + u[1]*u[1] + u[2]*u[2] + u[3]*u[3] + u[4]*u[4] +
          u[5]*u[5] + u[6]*u[6] + u[7]*u[7] + u[8]*u[8];
}

template <typename type>
type det_2D(const std::vector<type> &u)
{
   return u[0]*u[3] - u[1]*u[2];
}

template <typename type>
type det_3D(const std::vector<type> &u)
{
   return u[0]*(u[4]*u[8] - u[5]*u[7]) -
          u[1]*(u[3]*u[8] - u[5]*u[6]) +
          u[2]*(u[3]*u[7] - u[4]*u[6]);
}

template <typename type>
void mult_2D(const std::vector<type> &u, const std::vector<type> &M,
             std::vector<type> &mat)
{
   mat.resize(u.size());

   mat[0] = u[0]*M[0] + u[2]*M[1];
   mat[1] = u[1]*M[0] + u[3]*M[1];
   mat[2] = u[0]*M[2] + u[2]*M[3];
   mat[3] = u[1]*M[2] + u[3]*M[3];
}

template <typename type>
void mult_aTa_2D(const std::vector<type> &in, std::vector<type> &outm)
{
   outm.resize(in.size());
   outm[0] = in[0]*in[0];
   outm[1] = in[0]*in[2] + in[1]*in[3];
   outm[2] = in[0]*in[2] + in[1]*in[3];
   outm[3] = in[3]*in[3];
}

template <typename scalartype, typename type>
void add_2D(const scalartype &scalar, const std::vector<type> &u,
            const DenseMatrix *M, std::vector<type> &mat)
{
   mat.resize(u.size());
   mat[0] = u[0] + scalar * M->Elem(0,0);
   mat[1] = u[1] + scalar * M->Elem(1,0);
   mat[2] = u[2] + scalar * M->Elem(0,1);
   mat[3] = u[3] + scalar * M->Elem(1,1);
}

template <typename scalartype, typename type>
void add_2D(const scalartype &scalar, const std::vector<type> &u,
            const std::vector<type> &M, std::vector<type> &mat)
{
   mat.resize(M.size());
   mat[0] = u[0] + scalar * M[0];
   mat[1] = u[1] + scalar * M[1];
   mat[2] = u[2] + scalar * M[2];
   mat[3] = u[3] + scalar * M[3];
}

template <typename type>
void adjoint_2D(const std::vector<type> &in, std::vector<type> &outm)
{
   outm.resize(in.size());
   outm[0] = in[3];
   outm[1] = -in[1];
   outm[2] = -in[2];
   outm[3] = in[0];
}

template <typename type>
void transpose_2D(const std::vector<type> &in, std::vector<type> &outm)
{
   outm.resize(in.size());
   outm[0] = in[0];
   outm[1] = in[2];
   outm[2] = in[1];
   outm[3] = in[3];
}

template <typename scalartype, typename type>
void add_3D(const scalartype &scalar, const std::vector<type> &u,
            const DenseMatrix *M, std::vector<type> &mat)
{
   mat.resize(u.size());
   mat[0] = u[0] + scalar * M->Elem(0,0);
   mat[1] = u[1] + scalar * M->Elem(1,0);
   mat[2] = u[2] + scalar * M->Elem(2,0);

   mat[3] = u[3] + scalar * M->Elem(0,1);
   mat[4] = u[4] + scalar * M->Elem(1,1);
   mat[5] = u[5] + scalar * M->Elem(2,1);

   mat[6] = u[6] + scalar * M->Elem(0,2);
   mat[7] = u[7] + scalar * M->Elem(1,2);
   mat[8] = u[8] + scalar * M->Elem(2,2);
}

/* Metric definitions */

// W = ||T||^2 - 2*det(T).
template <typename type>
type mu4_ad(const std::vector<type> &T, const std::vector<type> &W)
{
   auto fnorm2 = fnorm2_2D(T);
   auto det = det_2D(T);
   return fnorm2 - 2*det;
};

// W = ||T-I||^2.
template <typename type>
type mu14_ad(const std::vector<type> &T, const std::vector<type> &W)
{
   DenseMatrix Id(2,2); Id = 0.0;
   Id(0,0) = 1; Id(1,1) = 1;

   std::vector<type> Mat;
   add_2D(real_t{-1.0}, T, &Id, Mat);

   return fnorm2_2D(Mat);
};

// W = (det(T)-1)^2.
template <typename type>
type mu55_ad(const std::vector<type> &T, const std::vector<type> &W)
{
   auto det = det_2D(T);
   return pow(det-1.0, 2.0);
};

// W = |T-T'|^2, where T'= |T|*I/sqrt(2).
template <typename type>
type mu85_ad(const std::vector<type> &T, const std::vector<type> &W)
{
   auto fnorm = sqrt(fnorm2_2D(T));
   return T[1]*T[1] + T[2]*T[2] +
          (T[0] - fnorm/sqrt(2))*(T[0] - fnorm/sqrt(2)) +
          (T[3] - fnorm/sqrt(2))*(T[3] - fnorm/sqrt(2));
};

// W = 1/tau |T-I|^2.
template <typename type>
type mu98_ad(const std::vector<type> &T, const std::vector<type> &W)
{
   DenseMatrix Id(2,2); Id = 0.0;
   Id(0,0) = 1; Id(1,1) = 1;

   std::vector<type> Mat;
   add_2D(real_t{-1.0}, T, &Id, Mat);

   return fnorm2_2D(Mat)/det_2D(T);
};

template <typename type>
type make_one_type()
{
   return 1.0;
}
// add specialization for AD1Type
template <>
AD1Type make_one_type<AD1Type>()
{
   return AD1Type{1.0, 0.0};
}
// add specialization for AD2Type
template <>
AD2Type make_one_type<AD2Type>()
{
   return AD2Type{AD1Type{1.0, 0.0}, AD1Type{0.0, 0.0}};
}

using TWCUO = TMOP_WorstCaseUntangleOptimizer_Metric;
template <typename type>
type wcuo_ad(type mu,
             const std::vector<type> &T, const std::vector<type> &W,
             real_t alpha, real_t min_detT, real_t detT_ep,
             int exponent, real_t max_muT, real_t muT_ep,
             TWCUO::BarrierType bt,
             TWCUO::WorstCaseType wct)
{
   type one = make_one_type<type>();
   type zero = 0.0*one;
   type denom = one;
   if (bt == TWCUO::BarrierType::Shifted)
   {
      auto val1 = alpha*min_detT-detT_ep < 0.0 ?
                  (alpha*min_detT-detT_ep)*one :
                  zero;
      denom = 2.0*(det_2D(T)-val1);
   }
   else if (bt == TWCUO::BarrierType::Pseudo)
   {
      auto detT = det_2D(T);
      denom = detT + sqrt(detT*detT + detT_ep*detT_ep);
   }
   mu = mu/denom;

   if (wct == TWCUO::WorstCaseType::PMean)
   {
      auto exp = exponent*one;
      mu = pow(mu, exp);
   }
   else if (wct == TWCUO::WorstCaseType::Beta)
   {
      auto beta = (max_muT+muT_ep)*one;
      mu = mu/(beta-mu);
   }
   return mu;
}

// W = 1/(tau^0.5) |T-I|^2.
template <typename type>
type mu342_ad(const std::vector<type> &T, const std::vector<type> &W)
{
   DenseMatrix Id(3,3); Id = 0.0;
   Id(0,0) = 1; Id(1,1) = 1; Id(2,2) = 1;

   std::vector<type> Mat;
   add_3D(real_t{-1.0}, T, &Id, Mat);

   return fnorm2_3D(Mat)/sqrt(det_3D(T));
};

// (1/4 alpha) | A - (adj A)^t W^t W / omega |^2
template <typename type>
type nu11_ad(const std::vector<type> &T, const std::vector<type> &W)
{
   std::vector<type> A;   // T*W = A
   std::vector<type> AdjA,AdjAt, WtW, WRK, WRK2;

   mult_2D(T,W,A); // We assume that both A and W are nonsingular.

   auto alpha = det_2D(A);
   auto omega = det_2D(W);
   adjoint_2D(A, AdjA);
   transpose_2D(AdjA, AdjAt);

   mult_aTa_2D(W, WtW);
   mult_2D(AdjAt, WtW, WRK);

   add_2D(-1.0/omega, A, WRK, WRK2);
   auto fnorm =  fnorm2_2D(WRK2);

   return 0.25 / (alpha) * fnorm;
};

// 0.5 * ( sqrt(alpha/omega) - sqrt(omega/alpha) )^2
template <typename type>
type nu14_ad(const std::vector<type> &T, const std::vector<type> &W)
{
   std::vector<type> A;   // T*W = A
   mult_2D(T,W,A);

   auto sqalpha = sqrt(det_2D(A));
   auto sqomega = sqrt(det_2D(W));

   return 0.5*pow(sqalpha/sqomega - sqomega/sqalpha, 2.0);
};

// (1/alpha) | A - W |^2
template <typename type>
type nu36_ad(const std::vector<type> &T, const std::vector<type> &W)
{
   std::vector<type> A;   // T*W = A
   std::vector<type> AminusW;  // A-W

   mult_2D(T,W,A);
   add_2D(-1.0,A,W,AminusW);
   auto fnorm =  fnorm2_2D(AminusW);

   return 1.0 / (det_2D(A)) * fnorm;
};

// [ 1.0 - cos( phi_A - phi_W ) ] / (sin phi_A * sin phi_W)
template <typename type>
type nu50_ad(const std::vector<type> &T, const std::vector<type> &W)
{
   // We assume that both A and W are nonsingular.
   std::vector<type> A;
   mult_2D(T,W,A);
   auto l1_A = sqrt(A[0]*A[0] + A[1]*A[1]);
   auto l2_A = sqrt(A[2]*A[2] + A[3]*A[3]);
   auto prod_A = l1_A*l2_A;
   auto det_A = A[0]*A[3] - A[1]*A[2];
   auto sin_A = det_A/prod_A;
   auto cos_A = (A[0]*A[2] + A[1]*A[3])/prod_A;

   auto l1_W = sqrt(W[0]*W[0] + W[1]*W[1]);
   auto l2_W = sqrt(W[2]*W[2] + W[3]*W[3]);
   auto prod_W = l1_W*l2_W;
   auto det_W = W[0]*W[3] - W[1]*W[2];
   auto sin_W = det_W/prod_W;
   auto cos_W = (W[0]*W[2] + W[1]*W[3])/prod_W;

   return (1.0 - cos_A*cos_W - sin_A*sin_W)/(sin_A*sin_W);
};

// [ 0.5 * (ups_A / ups_W + ups_W / ups_A) - cos(phi_A - phi_W) ] /
// (sin phi_A * sin phi_W), where ups = l_1 l_2 sin(phi)
template <typename type>
type nu51_ad(const std::vector<type> &T, const std::vector<type> &W)
{
   std::vector<type> A;
   mult_2D(T,W,A);
   // We assume that both A and W are nonsingular.
   auto l1_A = sqrt(A[0]*A[0] + A[1]*A[1]);
   auto l2_A = sqrt(A[2]*A[2] + A[3]*A[3]);
   auto prod_A = l1_A*l2_A;
   auto det_A = A[0]*A[3] - A[1]*A[2];
   auto sin_A = det_A/prod_A;
   auto cos_A = (A[0]*A[2] + A[1]*A[3])/prod_A;
   auto ups_A = l1_A*l2_A*sin_A;

   auto l1_W = sqrt(W[0]*W[0] + W[1]*W[1]);
   auto l2_W = sqrt(W[2]*W[2] + W[3]*W[3]);
   auto prod_W = l1_W*l2_W;
   auto det_W = W[0]*W[3] - W[1]*W[2];
   auto sin_W = det_W/prod_W;
   auto cos_W = (W[0]*W[2] + W[1]*W[3])/prod_W;
   auto ups_W = l1_W*l2_W*sin_W;

   return (0.5 * (ups_A / ups_W + ups_W / ups_A) - cos_A*cos_W - sin_A*sin_W) /
          (sin_A*sin_W);
};

// (1/2 alpha) | A - (|A|/|W|) W |^2
template <typename type>
type nu107_ad(const std::vector<type> &T, const std::vector<type> &W)
{
   std::vector<type> A;   // T*W = A
   std::vector<type> Mat;  // A-W
   mult_2D(T,W,A);

   auto alpha = det_2D(A);
   auto aw = sqrt(fnorm2_2D(A))/sqrt(fnorm2_2D(W));

   add_2D(-aw, A, W, Mat);
   return (0.5/alpha)*fnorm2_2D(Mat);
};

// 0.5[ 1.0 - cos( phi_A - phi_W ) ]
template <typename type>
type skew2D_ad(const std::vector<type> &T, const std::vector<type> &W)
{
   // We assume that both A and W are nonsingular.
   std::vector<type> A;
   mult_2D(T,W,A);
   auto l1_A = sqrt(A[0]*A[0] + A[1]*A[1]);
   auto l2_A = sqrt(A[2]*A[2] + A[3]*A[3]);
   auto prod_A = l1_A*l2_A;
   auto det_A = A[0]*A[3] - A[1]*A[2];
   auto sin_A = det_A/prod_A;
   auto cos_A = (A[0]*A[2] + A[1]*A[3])/prod_A;

   auto l1_W = sqrt(W[0]*W[0] + W[1]*W[1]);
   auto l2_W = sqrt(W[2]*W[2] + W[3]*W[3]);
   auto prod_W = l1_W*l2_W;
   auto det_W = W[0]*W[3] - W[1]*W[2];
   auto sin_W = det_W/prod_W;
   auto cos_W = (W[0]*W[2] + W[1]*W[3])/prod_W;

   return 0.5*(1.0 - cos_A*cos_W - sin_A*sin_W);
};

// Given mu(X,Y), compute dmu/dX or dmu/dY. Y is an optional parameter when
// computing dmu/dX.
void ADGrad(std::function<AD1Type(std::vector<AD1Type>&,
                                  std::vector<AD1Type>&)>mu_ad,
            DenseMatrix &dmu, //output
            const DenseMatrix &X, // parameter 1
            const DenseMatrix *Y = nullptr, //parameter 2
            const bool dX = true /*derivative with respect to X*/)
{
   int matsize = X.TotalSize();
   std::vector<AD1Type> adX(matsize), adY(matsize);

   for (int i=0; i<matsize; i++) { adX[i] = AD1Type{X.GetData()[i], 0.0}; }
   if (Y)
   {
      for (int i=0; i<matsize; i++) { adY[i] = AD1Type{Y->GetData()[i], 0.0}; }
   }

   if (dX)
   {
      for (int i=0; i<matsize; i++)
      {
         adX[i] = AD1Type{X.GetData()[i], 1.0};
         AD1Type rez = mu_ad(adX, adY);
         dmu.GetData()[i] = rez.gradient;
         adX[i] = AD1Type{X.GetData()[i], 0.0};
      }
   }
   else
   {
      MFEM_VERIFY(Y, "Y cannot be nullptr when dX = false.");
      for (int i=0; i<matsize; i++)
      {
         adY[i] = AD1Type{Y->GetData()[i], 1.0};
         AD1Type rez = mu_ad(adX,adY);
         dmu.GetData()[i] = rez.gradient;
         adY[i] = AD1Type{Y->GetData()[i], 0.0};
      }
   }
}

// Given mu(X,Y), compute d2mu/dX2, where Y is an optional parameter.
void ADHessian(std::function<AD2Type(std::vector<AD2Type>&,
                                     std::vector<AD2Type>&)> mu_ad,
               DenseTensor &d2mu_dX2,
               const DenseMatrix &X,
               const DenseMatrix *Y = nullptr)
{
   const int matsize = X.TotalSize();

   //use forward-forward mode
   std::vector<AD2Type> aduu(matsize), adY(matsize);
   for (int ii = 0; ii < matsize; ii++)
   {
      aduu[ii].value = AD1Type{X.GetData()[ii], 0.0};
      aduu[ii].gradient = AD1Type{0.0, 0.0};
   }
   if (Y)
   {
      for (int ii=0; ii<matsize; ii++)
      {
         adY[ii].value = AD1Type{Y->GetData()[ii], 0.0};
         adY[ii].gradient = AD1Type{0.0, 0.0};
      }
   }

   for (int ii = 0; ii < matsize; ii++)
   {
      aduu[ii].value = AD1Type{X.GetData()[ii], 1.0};
      for (int jj = 0; jj < (ii + 1); jj++)
      {
         aduu[jj].gradient = AD1Type{1.0, 0.0};
         AD2Type rez = mu_ad(aduu, adY);
         d2mu_dX2(ii).GetData()[jj] = rez.gradient.gradient;
         d2mu_dX2(jj).GetData()[ii] = rez.gradient.gradient;
         aduu[jj].gradient = AD1Type{0.0, 0.0};
      }
      aduu[ii].value = AD1Type{X.GetData()[ii], 0.0};
   }
   return;
}
/* end AD related definitions ========================================*/

// Target-matrix optimization paradigm (TMOP) mesh quality metrics.

void TMOP_QualityMetric::DefaultAssembleH(const DenseTensor &H,
                                          const DenseMatrix &DS,
                                          const real_t weight,
                                          DenseMatrix &A) const
{
   const int dof = DS.Height(), dim = DS.Width();

   // The first two go over the rows and cols of dP_dJ where P = dW_dJ.
   for (int r = 0; r < dim; r++)
   {
      for (int c = 0; c < dim; c++)
      {
         DenseMatrix Hrc = H(r+c*dim);

         // Compute each entry of d(Prc)_dJ.
         for (int rr = 0; rr < dim; rr++)
         {
            for (int cc = 0; cc < dim; cc++)
            {
               const real_t entry_rr_cc = Hrc(rr, cc);

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

real_t TMOP_Combo_QualityMetric::EvalWMatrixForm(const DenseMatrix &Jpt) const
{
   real_t metric = 0.;
   for (int i = 0; i < tmop_q_arr.Size(); i++)
   {
      metric += wt_arr[i]*tmop_q_arr[i]->EvalWMatrixForm(Jpt);
   }
   return metric;
}

real_t TMOP_Combo_QualityMetric::EvalW(const DenseMatrix &Jpt) const
{
   real_t metric = 0.;
   for (int i = 0; i < tmop_q_arr.Size(); i++)
   {
      metric += wt_arr[i]*tmop_q_arr[i]->EvalW(Jpt);
   }
   return metric;
}

void TMOP_Combo_QualityMetric::EvalP(const DenseMatrix &Jpt,
                                     DenseMatrix &P) const
{
   DenseMatrix Pt(P.Size());
   P = 0.0;
   for (int i = 0; i < tmop_q_arr.Size(); i++)
   {
      tmop_q_arr[i]->EvalP(Jpt, Pt);
      P.Add(wt_arr[i], Pt);
   }
}

void TMOP_Combo_QualityMetric::EvalPW(const DenseMatrix &Jpt,
                                      DenseMatrix &PW) const
{
   DenseMatrix Pt(PW.Size());
   PW = 0.0;
   for (int i = 0; i < tmop_q_arr.Size(); i++)
   {
      tmop_q_arr[i]->EvalPW(Jpt, Pt);
      PW.Add(wt_arr[i], Pt);
   }
}

AD1Type TMOP_Combo_QualityMetric::EvalW_AD1(const std::vector<AD1Type> &T,
                                            const std::vector<AD1Type> &W)
const
{
   AD1Type metric = {0., 0.};
   for (int i = 0; i < tmop_q_arr.Size(); i++)
   {
      metric += wt_arr[i]*tmop_q_arr[i]->EvalW_AD1(T, W);
   }
   return metric;
}

AD2Type TMOP_Combo_QualityMetric::EvalW_AD2(const std::vector<AD2Type> &T,
                                            const std::vector<AD2Type> &W)
const
{
   AD2Type metric = {{0., 0.},{0., 0.}};
   for (int i = 0; i < tmop_q_arr.Size(); i++)
   {
      metric += wt_arr[i]*tmop_q_arr[i]->EvalW_AD2(T, W);
   }
   return metric;
}

void TMOP_Combo_QualityMetric::AssembleH(const DenseMatrix &Jpt,
                                         const DenseMatrix &DS,
                                         const real_t weight,
                                         DenseMatrix &A) const
{
   DenseMatrix At(A.Size());
   for (int i = 0; i < tmop_q_arr.Size(); i++)
   {
      At = 0.0;
      tmop_q_arr[i]->AssembleH(Jpt, DS, weight * wt_arr[i], At);
      A += At;
   }
}

void TMOP_Combo_QualityMetric::ComputeBalancedWeights(
   const GridFunction &nodes, const TargetConstructor &tc,
   Vector &weights, bool use_pa, const IntegrationRule *IntRule) const
{
   const int m_cnt = tmop_q_arr.Size();
   Vector averages;
   ComputeAvgMetrics(nodes, tc, averages, use_pa, IntRule);
   weights.SetSize(m_cnt);

   // For [ combo_A_B_C = a m_A + b m_B + c m_C ] we would have:
   // a = BC / (AB + AC + BC), b = AC / (AB + AC + BC), c = AB / (AB + AC + BC),
   // where A = avg_m_A, B = avg_m_B, C = avg_m_C.
   // Nested loop to avoid division, as some avg may be 0.
   Vector products_no_m(m_cnt); products_no_m = 1.0;
   for (int m_p = 0; m_p < m_cnt; m_p++)
   {
      for (int m_a = 0; m_a < m_cnt; m_a++)
      {
         if (m_p != m_a) { products_no_m(m_p) *= averages(m_a); }
      }
   }
   const real_t pnm_sum = products_no_m.Sum();

   if (pnm_sum == 0.0) { weights = 1.0 / m_cnt; return; }
   for (int m = 0; m < m_cnt; m++) { weights(m) = products_no_m(m) / pnm_sum; }

   MFEM_ASSERT(fabs(weights.Sum() - 1.0) < 1e-14,
               "Error: sum should be 1 always: " << weights.Sum());
}

void TMOP_Combo_QualityMetric::ComputeAvgMetrics(
   const GridFunction &nodes, const TargetConstructor &tc,
   Vector &averages, bool use_pa, const IntegrationRule *IntRule) const
{
   const int m_cnt = tmop_q_arr.Size(),
             NE    = nodes.FESpace()->GetNE(),
             dim   = nodes.FESpace()->GetMesh()->Dimension();

   averages.SetSize(m_cnt);

   auto fe = nodes.FESpace()->GetTypicalFE();
   const IntegrationRule &ir =
      (IntRule) ? *IntRule : IntRules.Get(fe->GetGeomType(), 2*fe->GetOrder());

   // Integrals of all metrics.
   averages = 0.0;
   real_t volume = 0.0;
   if (use_pa)
   {
      for (int m = 0; m < m_cnt; m++)
      {
         if (dim == 2)
         {
            GetLocalEnergyPA_2D(nodes, tc, m, averages(m), volume, ir);
         }
         else
         {
            GetLocalEnergyPA_3D(nodes, tc, m, averages(m), volume, ir);
         }
      }
   }
   else
   {
      Array<int> pos_dofs;
      for (int e = 0; e < NE; e++)
      {
         const FiniteElement &fe_pos = *nodes.FESpace()->GetFE(e);
         const int nsp = ir.GetNPoints(), dof = fe_pos.GetDof();

         DenseMatrix dshape(dof, dim);
         DenseMatrix pos(dof, dim);
         pos.SetSize(dof, dim);
         Vector posV(pos.Data(), dof * dim);

         nodes.FESpace()->GetElementVDofs(e, pos_dofs);
         nodes.GetSubVector(pos_dofs, posV);

         DenseTensor W(dim, dim, nsp);
         DenseMatrix Winv(dim), T(dim), A(dim);
         tc.ComputeElementTargets(e, fe_pos, ir, posV, W);

         for (int q = 0; q < nsp; q++)
         {
            const DenseMatrix &Wj = W(q);
            CalcInverse(Wj, Winv);

            const IntegrationPoint &ip = ir.IntPoint(q);
            fe_pos.CalcDShape(ip, dshape);
            MultAtB(pos, dshape, A);
            Mult(A, Winv, T);

            const real_t w_detA = ip.weight * A.Det();
            for (int m = 0; m < m_cnt; m++)
            {
               tmop_q_arr[m]->SetTargetJacobian(Wj);
               averages(m) += tmop_q_arr[m]->EvalW(T) * w_detA;
            }
            volume += w_detA;
         }
      }
   }

   // Parallel case.
#ifdef MFEM_USE_MPI
   auto par_nodes = dynamic_cast<const ParGridFunction *>(&nodes);
   if (par_nodes)
   {
      MPI_Allreduce(MPI_IN_PLACE, averages.GetData(), m_cnt,
                    MPITypeMap<real_t>::mpi_type, MPI_SUM, par_nodes->ParFESpace()->GetComm());
      MPI_Allreduce(MPI_IN_PLACE, &volume, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                    par_nodes->ParFESpace()->GetComm());
   }
#endif

   averages /= volume;
}

real_t TMOP_WorstCaseUntangleOptimizer_Metric::EvalW(const DenseMatrix &Jpt)
const
{
   real_t metric_tilde = EvalWBarrier(Jpt);
   real_t metric = metric_tilde;
   if (wctype == WorstCaseType::PMean)
   {
      metric = std::pow(metric_tilde, exponent);
   }
   else if (wctype == WorstCaseType::Beta)
   {
      real_t beta = max_muT+muT_ep;
      metric = metric_tilde/(beta-metric_tilde);
   }
   return metric;
}

real_t TMOP_WorstCaseUntangleOptimizer_Metric::EvalWBarrier(
   const DenseMatrix &Jpt) const
{
   real_t denominator = 1.0;
   if (btype == BarrierType::Shifted)
   {
      denominator = 2.0*(Jpt.Det()-std::min(alpha*min_detT-detT_ep, (real_t) 0.0));
   }
   else if (btype == BarrierType::Pseudo)
   {
      real_t detT = Jpt.Det();
      denominator = detT + std::sqrt(detT*detT + detT_ep*detT_ep);
   }
   return tmop_metric.EvalW(Jpt)/denominator;
}

AD1Type TMOP_WorstCaseUntangleOptimizer_Metric::EvalW_AD1(
   const std::vector<AD1Type> &T,
   const std::vector<AD1Type> &W) const
{
   return wcuo_ad(tmop_metric.EvalW_AD1(T,W), T, W, alpha, min_detT, detT_ep,
                  exponent, max_muT, muT_ep, btype, wctype);
}

AD2Type TMOP_WorstCaseUntangleOptimizer_Metric::EvalW_AD2(
   const std::vector<AD2Type> &T,
   const std::vector<AD2Type> &W) const
{
   return wcuo_ad(tmop_metric.EvalW_AD2(T,W), T, W, alpha, min_detT, detT_ep,
                  exponent, max_muT, muT_ep, btype, wctype);
}

void TMOP_WorstCaseUntangleOptimizer_Metric::EvalP(const DenseMatrix &Jpt,
                                                   DenseMatrix &P) const
{
   auto mu_ad_fn = [this](std::vector<AD1Type> &T, std::vector<AD1Type> &W)
   {
      return EvalW_AD1(T,W);
   };
   if (tmop_metric.Id() == 4 || tmop_metric.Id() == 14 ||
       tmop_metric.Id() == 66)
   {
      ADGrad(mu_ad_fn, P, Jpt);
      return;
   }
   MFEM_ABORT("EvalW_AD1 not implemented with this metric for "
              "TMOP_WorstCaseUntangleOptimizer_Metric. "
              "Please use metric 4/14/66.");
}

void TMOP_WorstCaseUntangleOptimizer_Metric::AssembleH(
   const DenseMatrix &Jpt,
   const DenseMatrix &DS,
   const real_t weight,
   DenseMatrix &A) const
{
   DenseTensor H(Jpt.Height(), Jpt.Height(), Jpt.TotalSize());
   H = 0.0;
   auto mu_ad_fn = [this](std::vector<AD2Type> &T, std::vector<AD2Type> &W)
   {
      return EvalW_AD2(T,W);
   };
   if (tmop_metric.Id() == 4 || tmop_metric.Id() == 14 ||
       tmop_metric.Id() == 66)
   {
      ADHessian(mu_ad_fn, H, Jpt);
      this->DefaultAssembleH(H,DS,weight,A);
      return;
   }
   MFEM_ABORT("EvalW_AD1 not implemented with this metric for "
              "TMOP_WorstCaseUntangleOptimizer_Metric. "
              "Please use metric 4/14/66.");
}

real_t TMOP_Metric_001::EvalW(const DenseMatrix &Jpt) const
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
                                const real_t weight,
                                DenseMatrix &A) const
{
   ie.SetJacobian(Jpt.GetData());
   ie.SetDerivativeMatrix(DS.Height(), DS.GetData());
   ie.Assemble_ddI1(weight, A.GetData());
}

real_t TMOP_Metric_skew2D::EvalWMatrixForm(const DenseMatrix &Jpt) const
{
   MFEM_VERIFY(Jtr != NULL,
               "Requires a target Jacobian, use SetTargetJacobian().");
   int matsize = Jpt.TotalSize();
   std::vector<AD1Type> T(matsize), W(matsize);
   for (int i=0; i<matsize; i++)
   {
      T[i] = AD1Type{Jpt.GetData()[i], 0.0};
      W[i] = AD1Type{Jtr->GetData()[i], 0.0};
   }
   return skew2D_ad(T, W).value;
}

void TMOP_Metric_skew2D::EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const
{
   ADGrad(skew2D_ad<AD1Type>, P, Jpt, Jtr);
   return;
}

void TMOP_Metric_skew2D::EvalPW(const DenseMatrix &Jpt, DenseMatrix &PW) const
{
   ADGrad(skew2D_ad<AD1Type>, PW, Jpt, Jtr, false);
   return;
}

void TMOP_Metric_skew2D::AssembleH(const DenseMatrix &Jpt,
                                   const DenseMatrix &DS,
                                   const real_t weight,
                                   DenseMatrix &A) const
{
   const int dim = Jpt.Height();
   DenseTensor H(dim, dim, dim*dim); H = 0.0;
   ADHessian(skew2D_ad<AD2Type>, H, Jpt, Jtr);
   this->DefaultAssembleH(H,DS,weight,A);
}

real_t TMOP_Metric_skew3D::EvalW(const DenseMatrix &Jpt) const
{
   MFEM_VERIFY(Jtr != NULL,
               "Requires a target Jacobian, use SetTargetJacobian().");

   DenseMatrix Jpr(3, 3);
   Mult(Jpt, *Jtr, Jpr);

   Vector col1, col2, col3;
   Jpr.GetColumn(0, col1);
   Jpr.GetColumn(1, col2);
   Jpr.GetColumn(2, col3);
   real_t norm_c1 = col1.Norml2(),
          norm_c2 = col2.Norml2(),
          norm_c3 = col3.Norml2();
   real_t cos_Jpr_12 = (col1 * col2) / (norm_c1 * norm_c2),
          cos_Jpr_13 = (col1 * col3) / (norm_c1 * norm_c3),
          cos_Jpr_23 = (col2 * col3) / (norm_c2 * norm_c3);
   real_t sin_Jpr_12 = std::sqrt(1.0 - cos_Jpr_12 * cos_Jpr_12),
          sin_Jpr_13 = std::sqrt(1.0 - cos_Jpr_13 * cos_Jpr_13),
          sin_Jpr_23 = std::sqrt(1.0 - cos_Jpr_23 * cos_Jpr_23);

   Jtr->GetColumn(0, col1);
   Jtr->GetColumn(1, col2);
   Jtr->GetColumn(2, col3);
   norm_c1 = col1.Norml2();
   norm_c2 = col2.Norml2(),
   norm_c3 = col3.Norml2();
   real_t cos_Jtr_12 = (col1 * col2) / (norm_c1 * norm_c2),
          cos_Jtr_13 = (col1 * col3) / (norm_c1 * norm_c3),
          cos_Jtr_23 = (col2 * col3) / (norm_c2 * norm_c3);
   real_t sin_Jtr_12 = std::sqrt(1.0 - cos_Jtr_12 * cos_Jtr_12),
          sin_Jtr_13 = std::sqrt(1.0 - cos_Jtr_13 * cos_Jtr_13),
          sin_Jtr_23 = std::sqrt(1.0 - cos_Jtr_23 * cos_Jtr_23);

   return (3.0 - cos_Jpr_12 * cos_Jtr_12 - sin_Jpr_12 * sin_Jtr_12
           - cos_Jpr_13 * cos_Jtr_13 - sin_Jpr_13 * sin_Jtr_13
           - cos_Jpr_23 * cos_Jtr_23 - sin_Jpr_23 * sin_Jtr_23) / 6.0;
}

real_t TMOP_Metric_aspratio2D::EvalW(const DenseMatrix &Jpt) const
{
   MFEM_VERIFY(Jtr != NULL,
               "Requires a target Jacobian, use SetTargetJacobian().");

   DenseMatrix Jpr(2, 2);
   Mult(Jpt, *Jtr, Jpr);

   Vector col1, col2;
   Jpr.GetColumn(0, col1);
   Jpr.GetColumn(1, col2);
   const real_t ratio_Jpr = col2.Norml2() / col1.Norml2();

   Jtr->GetColumn(0, col1);
   Jtr->GetColumn(1, col2);
   const real_t ratio_Jtr = col2.Norml2() / col1.Norml2();

   return 0.5 * (ratio_Jpr / ratio_Jtr + ratio_Jtr / ratio_Jpr) - 1.0;
}

real_t TMOP_Metric_aspratio3D::EvalW(const DenseMatrix &Jpt) const
{
   MFEM_VERIFY(Jtr != NULL,
               "Requires a target Jacobian, use SetTargetJacobian().");

   DenseMatrix Jpr(3, 3);
   Mult(Jpt, *Jtr, Jpr);

   Vector col1, col2, col3;
   Jpr.GetColumn(0, col1);
   Jpr.GetColumn(1, col2);
   Jpr.GetColumn(2, col3);
   real_t norm_c1 = col1.Norml2(),
          norm_c2 = col2.Norml2(),
          norm_c3 = col3.Norml2();
   real_t ratio_Jpr_1 = norm_c1 / std::sqrt(norm_c2 * norm_c3),
          ratio_Jpr_2 = norm_c2 / std::sqrt(norm_c1 * norm_c3),
          ratio_Jpr_3 = norm_c3 / std::sqrt(norm_c1 * norm_c2);

   Jtr->GetColumn(0, col1);
   Jtr->GetColumn(1, col2);
   Jtr->GetColumn(2, col3);
   norm_c1 = col1.Norml2();
   norm_c2 = col2.Norml2();
   norm_c3 = col3.Norml2();
   real_t ratio_Jtr_1 = norm_c1 / std::sqrt(norm_c2 * norm_c3),
          ratio_Jtr_2 = norm_c2 / std::sqrt(norm_c1 * norm_c3),
          ratio_Jtr_3 = norm_c3 / std::sqrt(norm_c1 * norm_c2);

   return ( 0.5 * (ratio_Jpr_1 / ratio_Jtr_1 + ratio_Jtr_1 / ratio_Jpr_1) +
            0.5 * (ratio_Jpr_2 / ratio_Jtr_2 + ratio_Jtr_2 / ratio_Jpr_2) +
            0.5 * (ratio_Jpr_3 / ratio_Jtr_3 + ratio_Jtr_3 / ratio_Jpr_3) - 3.0
          ) / 3.0;
}

real_t TMOP_Metric_002::EvalWMatrixForm(const DenseMatrix &Jpt) const
{
   return 0.5 * Jpt.FNorm2() / Jpt.Det() - 1.0;
}

real_t TMOP_Metric_002::EvalW(const DenseMatrix &Jpt) const
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
                                const real_t weight,
                                DenseMatrix &A) const
{
   ie.SetJacobian(Jpt.GetData());
   ie.SetDerivativeMatrix(DS.Height(), DS.GetData());
   ie.Assemble_ddI1b(0.5*weight, A.GetData());
}

real_t TMOP_Metric_004::EvalW(const DenseMatrix &Jpt) const
{
   ie.SetJacobian(Jpt.GetData());
   return ie.Get_I1() - 2.0*ie.Get_I2b();
}

void TMOP_Metric_004::EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const
{
   ie.SetJacobian(Jpt.GetData());
   Add(1.0, ie.Get_dI1(), -2.0, ie.Get_dI2b(), P);
}

void TMOP_Metric_004::AssembleH(const DenseMatrix &Jpt,
                                const DenseMatrix &DS,
                                const real_t weight,
                                DenseMatrix &A) const
{
   ie.SetJacobian(Jpt.GetData());
   ie.SetDerivativeMatrix(DS.Height(), DS.GetData());

   ie.Assemble_ddI1(weight, A.GetData());
   ie.Assemble_ddI2b(-2.0*weight, A.GetData());
}

template <typename type>
type TMOP_Metric_004::EvalW_AD_impl(const std::vector<type> &T,
                                    const std::vector<type> &W) const
{
   return mu4_ad(T, W);
}

AD1Type TMOP_Metric_004::EvalW_AD1(const std::vector<AD1Type> &T,
                                   const std::vector<AD1Type> &W) const
{
   return EvalW_AD_impl<AD1Type>(T,W);
}

AD2Type TMOP_Metric_004::EvalW_AD2(const std::vector<AD2Type> &T,
                                   const std::vector<AD2Type> &W) const
{
   return EvalW_AD_impl<AD2Type>(T,W);
}

real_t TMOP_Metric_007::EvalW(const DenseMatrix &Jpt) const
{
   // mu_7 = |J-J^{-t}|^2 = |J|^2 + |J^{-1}|^2 - 4
   ie.SetJacobian(Jpt.GetData());
   return ie.Get_I1()*(1. + 1./ie.Get_I2()) - 4.0;
}

void TMOP_Metric_007::EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const
{
   // P = d(I1*(1 + 1/I2)) = (1 + 1/I2) dI1 - I1/I2^2 dI2
   ie.SetJacobian(Jpt.GetData());
   const real_t I2 = ie.Get_I2();
   Add(1. + 1./I2, ie.Get_dI1(), -ie.Get_I1()/(I2*I2), ie.Get_dI2(), P);
}

void TMOP_Metric_007::AssembleH(const DenseMatrix &Jpt,
                                const DenseMatrix &DS,
                                const real_t weight,
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
   const real_t c1 = 1./ie.Get_I2();
   const real_t c2 = weight*c1*c1;
   const real_t c3 = ie.Get_I1()*c2;
   ie.Assemble_ddI1(weight*(1. + c1), A.GetData());
   ie.Assemble_ddI2(-c3, A.GetData());
   ie.Assemble_TProd(-c2, ie.Get_dI1(), ie.Get_dI2(), A.GetData());
   ie.Assemble_TProd(2*c1*c3, ie.Get_dI2(), A.GetData());
}

real_t TMOP_Metric_009::EvalW(const DenseMatrix &Jpt) const
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
                                const real_t weight,
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

real_t TMOP_Metric_014::EvalWMatrixForm(const DenseMatrix &Jpt) const
{
   // mu_14 = |J - I|^2.
   DenseMatrix Mat(Jpt);
   Mat(0,0) -= 1.0;
   Mat(1,1) -= 1.0;
   return Mat.FNorm2();
}

real_t TMOP_Metric_014::EvalW(const DenseMatrix &Jpt) const
{
   // mu_14 = |J - I|^2 = I1[J-I].
   DenseMatrix Mat(Jpt);
   Mat(0,0) -= 1.0;
   Mat(1,1) -= 1.0;

   ie.SetJacobian(Mat.GetData());
   return ie.Get_I1();
}

void TMOP_Metric_014::EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const
{
   // P = dI1[J-I] d/dJ[J-I] = dI1[J-I].
   DenseMatrix JptMinusId = Jpt;
   for (int i = 0; i < Jpt.Size(); i++)
   {
      JptMinusId(i, i) -= 1.0;
   }
   ie.SetJacobian(JptMinusId.GetData());
   P = ie.Get_dI1();
}

void TMOP_Metric_014::AssembleH(const DenseMatrix &Jpt,
                                const DenseMatrix &DS,
                                const real_t weight,
                                DenseMatrix &A) const
{
   // dP = ddI1[J-I].
   DenseMatrix JptMinusId = Jpt;
   for (int i = 0; i < Jpt.Size(); i++)
   {
      JptMinusId(i, i) -= 1.0;
   }
   ie.SetJacobian(JptMinusId.GetData());
   ie.SetDerivativeMatrix(DS.Height(), DS.GetData());
   ie.Assemble_ddI1(weight, A.GetData());
}

template <typename type>
type TMOP_Metric_014::EvalW_AD_impl(const std::vector<type> &T,
                                    const std::vector<type> &W) const
{
   return mu14_ad(T, W);
}

AD1Type TMOP_Metric_014::EvalW_AD1(const std::vector<AD1Type> &T,
                                   const std::vector<AD1Type> &W) const
{
   return EvalW_AD_impl<AD1Type>(T,W);
}

AD2Type TMOP_Metric_014::EvalW_AD2(const std::vector<AD2Type> &T,
                                   const std::vector<AD2Type> &W) const
{
   return EvalW_AD_impl<AD2Type>(T,W);
}

real_t TMOP_Metric_022::EvalW(const DenseMatrix &Jpt) const
{
   // mu_22 = (0.5*|J|^2 - det(J)) / (det(J) - tau0)
   //       = (0.5*I1 - I2b) / (I2b - tau0)
   ie.SetJacobian(Jpt.GetData());
   const real_t I2b = ie.Get_I2b();

   real_t d = I2b - min_detT;
   if (d < 0.0 && min_detT == 0.0)
   {
      // The mesh has been untangled, but it's still possible to get negative
      // detJ in FD calculations, as they move the nodes around with some small
      // increments and can produce negative determinants. Thus we put a small
      // value in the denominator. Note that here I2b < 0.
      d = - I2b * 0.1;
   }

   return (0.5*ie.Get_I1() - I2b) / d;
}

void TMOP_Metric_022::EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const
{
   // mu_22 = (0.5*I1 - I2b) / (I2b - tau0)
   // P = 1/(I2b - tau0)*(0.5*dI1 - dI2b) - (0.5*I1 - I2b)/(I2b - tau0)^2*dI2b
   //   = 0.5/(I2b - tau0)*dI1 + (tau0 - 0.5*I1)/(I2b - tau0)^2*dI2b
   ie.SetJacobian(Jpt.GetData());
   const real_t c1 = 1.0/(ie.Get_I2b() - min_detT);
   Add(c1/2, ie.Get_dI1(), (min_detT - ie.Get_I1()/2)*c1*c1, ie.Get_dI2b(), P);
}

void TMOP_Metric_022::AssembleH(const DenseMatrix &Jpt,
                                const DenseMatrix &DS,
                                const real_t weight,
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
   const real_t c1 = 1.0/(ie.Get_I2b() - min_detT);
   const real_t c2 = weight*c1/2;
   const real_t c3 = c1*c2;
   const real_t c4 = (2*min_detT - ie.Get_I1())*c3; // weight*z
   ie.Assemble_TProd(-c3, ie.Get_dI1(), ie.Get_dI2b(), A.GetData());
   ie.Assemble_TProd(-2*c1*c4, ie.Get_dI2b(), A.GetData());
   ie.Assemble_ddI1(c2, A.GetData());
   ie.Assemble_ddI2b(c4, A.GetData());
}

real_t TMOP_Metric_050::EvalWMatrixForm(const DenseMatrix &Jpt) const
{
   // mu_50 = 0.5 |J^t J|^2 / det(J)^2 - 1.
   DenseMatrix JtJ(2);
   MultAAt(Jpt, JtJ);
   JtJ.Transpose();
   real_t det = Jpt.Det();

   return 0.5 * JtJ.FNorm2()/(det*det) - 1.0;
}

real_t TMOP_Metric_050::EvalW(const DenseMatrix &Jpt) const
{
   // mu_50 = 0.5*|J^t J|^2/det(J)^2 - 1
   //       = 0.5*(l1^4 + l2^4)/(l1*l2)^2 - 1
   //       = 0.5*((l1/l2)^2 + (l2/l1)^2) - 1 = 0.5*(l1/l2 - l2/l1)^2
   //       = 0.5*(l1/l2 + l2/l1)^2 - 2 = 0.5*I1b^2 - 2.
   ie.SetJacobian(Jpt.GetData());
   const real_t I1b = ie.Get_I1b();
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
                                const real_t weight,
                                DenseMatrix &A) const
{
   // P  = I1b*dI1b
   // dP = dI1b x dI1b + I1b*ddI1b
   ie.SetJacobian(Jpt.GetData());
   ie.SetDerivativeMatrix(DS.Height(), DS.GetData());
   ie.Assemble_TProd(weight, ie.Get_dI1b(), A.GetData());
   ie.Assemble_ddI1b(weight*ie.Get_I1b(), A.GetData());
}

real_t TMOP_Metric_055::EvalW(const DenseMatrix &Jpt) const
{
   // mu_55 = (det(J) - 1)^2 = (I2b - 1)^2
   ie.SetJacobian(Jpt.GetData());
   const real_t c1 = ie.Get_I2b() - 1.0;
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
                                const real_t weight,
                                DenseMatrix &A) const
{
   // P  = 2*(I2b - 1)*dI2b
   // dP = 2*(dI2b x dI2b) + 2*(I2b - 1)*ddI2b
   ie.SetJacobian(Jpt.GetData());
   ie.SetDerivativeMatrix(DS.Height(), DS.GetData());
   ie.Assemble_TProd(2*weight, ie.Get_dI2b(), A.GetData());
   ie.Assemble_ddI2b(2*weight*(ie.Get_I2b() - 1.0), A.GetData());
}

template <typename type>
type TMOP_Metric_055::EvalW_AD_impl(const std::vector<type> &T,
                                    const std::vector<type> &W) const
{
   return mu55_ad(T, W);
}

AD1Type TMOP_Metric_055::EvalW_AD1(const std::vector<AD1Type> &T,
                                   const std::vector<AD1Type> &W) const
{
   return EvalW_AD_impl<AD1Type>(T,W);
}

AD2Type TMOP_Metric_055::EvalW_AD2(const std::vector<AD2Type> &T,
                                   const std::vector<AD2Type> &W) const
{
   return EvalW_AD_impl<AD2Type>(T,W);
}

real_t TMOP_Metric_056::EvalWMatrixForm(const DenseMatrix &Jpt) const
{
   // mu_56 = 0.5 (det(J) + 1 / det(J)) - 1.
   const real_t d = Jpt.Det();
   return 0.5 * (d + 1.0 / d) - 1.0;
}

real_t TMOP_Metric_056::EvalW(const DenseMatrix &Jpt) const
{
   // mu_56 = 0.5*(I2b + 1/I2b) - 1.
   ie.SetJacobian(Jpt.GetData());
   const real_t I2b = ie.Get_I2b();
   return 0.5*(I2b + 1.0/I2b) - 1.0;
}

void TMOP_Metric_056::EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const
{
   // mu_56 = 0.5*(I2b + 1/I2b) - 1.
   // P = 0.5*(1 - 1/I2b^2)*dI2b.
   ie.SetJacobian(Jpt.GetData());
   P.Set(0.5 - 0.5/ie.Get_I2(), ie.Get_dI2b());
}

void TMOP_Metric_056::AssembleH(const DenseMatrix &Jpt,
                                const DenseMatrix &DS,
                                const real_t weight,
                                DenseMatrix &A) const
{
   // P  = 0.5*(1 - 1/I2b^2)*dI2b.
   // dP = (1/I2b^3)*(dI2b x dI2b) + (0.5 - 0.5/I2)*ddI2b.
   ie.SetJacobian(Jpt.GetData());
   ie.SetDerivativeMatrix(DS.Height(), DS.GetData());
   ie.Assemble_TProd(weight/(ie.Get_I2()*ie.Get_I2b()),
                     ie.Get_dI2b(), A.GetData());
   ie.Assemble_ddI2b(weight*(0.5 - 0.5/ie.Get_I2()), A.GetData());
}

real_t TMOP_Metric_058::EvalWMatrixForm(const DenseMatrix &Jpt) const
{
   // mu_58 = |J^t J|^2 / det(J)^2 - 2|J|^2 / det(J) + 2.
   DenseMatrix JtJ(2);
   MultAAt(Jpt, JtJ);
   JtJ.Transpose();
   real_t det = Jpt.Det();

   return JtJ.FNorm2()/(det*det) - 2*Jpt.FNorm2()/det + 2.0;
}

real_t TMOP_Metric_058::EvalW(const DenseMatrix &Jpt) const
{
   // mu_58 = I1b*(I1b - 2)
   ie.SetJacobian(Jpt.GetData());
   const real_t I1b = ie.Get_I1b();
   return I1b*(I1b - 2.0);
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
                                const real_t weight,
                                DenseMatrix &A) const
{
   // P  = (2*I1b - 2)*dI1b
   // dP =  2*(dI1b x dI1b) + (2*I1b - 2)*ddI1b
   ie.SetJacobian(Jpt.GetData());
   ie.SetDerivativeMatrix(DS.Height(), DS.GetData());
   ie.Assemble_TProd(2*weight, ie.Get_dI1b(), A.GetData());
   ie.Assemble_ddI1b(weight*(2*ie.Get_I1b() - 2.0), A.GetData());
}

real_t TMOP_Metric_077::EvalWMatrixForm(const DenseMatrix &Jpt) const
{
   // mu_77 = 0.5 (det(J)^2 + 1 / det(J)^2) - 1.
   const real_t d = Jpt.Det();
   return 0.5 * (d*d + 1.0/(d*d)) - 1.0;
}
real_t TMOP_Metric_077::EvalW(const DenseMatrix &Jpt) const
{
   // mu_77 = 0.5 (I2 + 1 / I2) - 1.0.
   ie.SetJacobian(Jpt.GetData());
   const real_t I2 = ie.Get_I2();
   return  0.5*(I2 + 1.0/I2) - 1.0;
}

void TMOP_Metric_077::EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const
{
   // mu_77 = 0.5 (I2 + 1 / I2) - 1.0.
   // P = 1/2 (1 - 1/I2^2) dI2_dJ.
   ie.SetJacobian(Jpt.GetData());
   const real_t I2 = ie.Get_I2();
   P.Set(0.5 * (1.0 - 1.0 / (I2 * I2)), ie.Get_dI2());
}

void TMOP_Metric_077::AssembleH(const DenseMatrix &Jpt,
                                const DenseMatrix &DS,
                                const real_t weight,
                                DenseMatrix &A) const
{
   ie.SetJacobian(Jpt.GetData());
   ie.SetDerivativeMatrix(DS.Height(), DS.GetData());
   const real_t I2 = ie.Get_I2(), I2inv_sq = 1.0 / (I2 * I2);
   ie.Assemble_ddI2(weight*0.5*(1.0 - I2inv_sq), A.GetData());
   ie.Assemble_TProd(weight * I2inv_sq / I2, ie.Get_dI2(), A.GetData());
}

// mu_85 = |T-T'|^2, where T'= |T|*I/sqrt(2)
real_t TMOP_Metric_085::EvalWMatrixForm(const DenseMatrix &Jpt) const
{
   int matsize = Jpt.TotalSize();
   std::vector<AD1Type> T(matsize), W(matsize);
   for (int i=0; i<matsize; i++) { T[i] = AD1Type{Jpt.GetData()[i], 0.0}; }
   return mu85_ad(T, W).value;
}

void TMOP_Metric_085::EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const
{
   ADGrad(mu85_ad<AD1Type>, P, Jpt);
   return;
}

void TMOP_Metric_085::AssembleH(const DenseMatrix &Jpt,
                                const DenseMatrix &DS,
                                const real_t weight,
                                DenseMatrix &A) const
{
   const int dim = Jpt.Height();
   DenseTensor H(dim, dim, dim*dim); H = 0.0;
   ADHessian(mu85_ad<AD2Type>, H, Jpt);
   this->DefaultAssembleH(H,DS,weight,A);
}

// mu_98 = 1/(tau)|T-I|^2
real_t TMOP_Metric_098::EvalWMatrixForm(const DenseMatrix &Jpt) const
{
   int matsize = Jpt.TotalSize();
   std::vector<AD1Type> T(matsize), W(matsize);
   for (int i=0; i<matsize; i++) { T[i] = AD1Type{Jpt.GetData()[i], 0.0}; }
   return mu98_ad(T, W).value;
}

void TMOP_Metric_098::EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const
{
   ADGrad(mu98_ad<AD1Type>, P, Jpt);
   return;
}

void TMOP_Metric_098::AssembleH(const DenseMatrix &Jpt,
                                const DenseMatrix &DS,
                                const real_t weight,
                                DenseMatrix &A) const
{
   const int dim = Jpt.Height();
   DenseTensor H(dim, dim, dim*dim); H = 0.0;
   ADHessian(mu98_ad<AD2Type>, H, Jpt);
   this->DefaultAssembleH(H,DS,weight,A);
}

real_t TMOP_Metric_211::EvalW(const DenseMatrix &Jpt) const
{
   // mu_211 = (det(J) - 1)^2 - det(J) + (det(J)^2 + eps)^{1/2}
   //        = (I2b - 1)^2 - I2b + sqrt(I2b^2 + eps)
   ie.SetJacobian(Jpt.GetData());
   const real_t I2b = ie.Get_I2b();
   return (I2b - 1.0)*(I2b - 1.0) - I2b + std::sqrt(I2b*I2b + eps);
}

void TMOP_Metric_211::EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const
{
   MFEM_ABORT("Metric not implemented yet. Use metric mu_55 instead.");
}

void TMOP_Metric_211::AssembleH(const DenseMatrix &Jpt,
                                const DenseMatrix &DS,
                                const real_t weight,
                                DenseMatrix &A) const
{
   MFEM_ABORT("Metric not implemented yet. Use metric mu_55 instead.");
}

real_t TMOP_Metric_252::EvalW(const DenseMatrix &Jpt) const
{
   // mu_252 = 0.5*(det(J) - 1)^2 / (det(J) - tau0).
   ie.SetJacobian(Jpt.GetData());
   const real_t I2b = ie.Get_I2b();
   return 0.5*(I2b - 1.0)*(I2b - 1.0)/(I2b - tau0);
}

void TMOP_Metric_252::EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const
{
   // mu_252 = 0.5*(det(J) - 1)^2 / (det(J) - tau0)
   // P = (c - 0.5*c*c) * dI2b
   //
   // c = (I2b - 1)/(I2b - tau0), see TMOP_Metric_352 for details
   ie.SetJacobian(Jpt.GetData());
   const real_t I2b = ie.Get_I2b();
   const real_t c = (I2b - 1.0)/(I2b - tau0);
   P.Set(c - 0.5*c*c, ie.Get_dI2b());
}

void TMOP_Metric_252::AssembleH(const DenseMatrix &Jpt,
                                const DenseMatrix &DS,
                                const real_t weight,
                                DenseMatrix &A) const
{
   // c = (I2b - 1)/(I2b - tau0), see TMOP_Metric_352 for details
   //
   // P  = (c - 0.5*c*c) * dI2b
   // dP = (1 - c)^2/(I2b - tau0)*(dI2b x dI2b) + (c - 0.5*c*c)*ddI2b
   ie.SetJacobian(Jpt.GetData());
   ie.SetDerivativeMatrix(DS.Height(), DS.GetData());
   const real_t I2b = ie.Get_I2b();
   const real_t c0 = 1.0/(I2b - tau0);
   const real_t c = c0*(I2b - 1.0);
   ie.Assemble_TProd(weight*c0*(1.0 - c)*(1.0 - c), ie.Get_dI2b(), A.GetData());
   ie.Assemble_ddI2b(weight*(c - 0.5*c*c), A.GetData());
}

real_t TMOP_Metric_301::EvalWMatrixForm(const DenseMatrix &Jpt) const
{
   // mu_301 = 1/3 |J| |J^-1| - 1.
   ie.SetJacobian(Jpt.GetData());
   DenseMatrix inv(3);
   CalcInverse(Jpt, inv);
   return Jpt.FNorm() * inv.FNorm() / 3.0 - 1.0;
}

real_t TMOP_Metric_301::EvalW(const DenseMatrix &Jpt) const
{
   // mu_301 = 1/3 sqrt(I1b * I2b) - 1
   ie.SetJacobian(Jpt.GetData());
   return std::sqrt(ie.Get_I1b()*ie.Get_I2b())/3. - 1.;
}

void TMOP_Metric_301::EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const
{
   //  W = (1/3)*sqrt(I1b*I2b) - 1
   // dW = (1/6)/sqrt(I1b*I2b)*[I2b*dI1b + I1b*dI2b]
   ie.SetJacobian(Jpt.GetData());
   const real_t a = 1./(6.*std::sqrt(ie.Get_I1b()*ie.Get_I2b()));
   Add(a*ie.Get_I2b(), ie.Get_dI1b(), a*ie.Get_I1b(), ie.Get_dI2b(), P);
}

void TMOP_Metric_301::AssembleH(const DenseMatrix &Jpt,
                                const DenseMatrix &DS,
                                const real_t weight,
                                DenseMatrix &A) const
{
   //  dW = (1/6)/sqrt(I1b*I2b)*[I2b*dI1b + I1b*dI2b]
   //  dW = (1/6)*[z2*dI1b + z1*dI2b], z1 = sqrt(I1b/I2b), z2 = sqrt(I2b/I1b)
   // ddW = (1/6)*[dI1b x dz2 + z2*ddI1b + dI2b x dz1 + z1*ddI2b]
   //
   // dz1 = (1/2)*sqrt(I2b/I1b) [ (1/I2b)*dI1b - (I1b/(I2b*I2b))*dI2b ]
   //     = (1/2)/sqrt(I1b*I2b) [ dI1b - (I1b/I2b)*dI2b ]
   // dz2 = (1/2)/sqrt(I1b*I2b) [ dI2b - (I2b/I1b)*dI1b ]
   //
   // dI1b x dz2 + dI2b x dz1 =
   //    (1/2)/sqrt(I1b*I2b) dI1b x [ dI2b - (I2b/I1b)*dI1b ] +
   //    (1/2)/sqrt(I1b*I2b) dI2b x [ dI1b - (I1b/I2b)*dI2b ] =
   //    (1/2)/sqrt(I1b*I2b) [sqrt(I1b/I2b)*dI2b - sqrt(I2b/I1b)*dI1b] x
   //                        [sqrt(I2b/I1b)*dI1b - sqrt(I1b/I2b)*dI2b] =
   //    (1/2)*(I1b*I2b)^{-3/2} (I1b*dI2b - I2b*dI1b) x (I2b*dI1b - I1b*dI2b)
   //      and the last two parentheses are the same up to a sign.
   //
   // z1 = I1b/sqrt(I1b*I2b), z2 = I2b/sqrt(I1b*I2b)

   ie.SetJacobian(Jpt.GetData());
   ie.SetDerivativeMatrix(DS.Height(), DS.GetData());
   real_t X_data[9];
   DenseMatrix X(X_data, 3, 3);
   Add(- ie.Get_I2b(), ie.Get_dI1b(), ie.Get_I1b(), ie.Get_dI2b(), X);
   const real_t I1b_I2b = ie.Get_I1b()*ie.Get_I2b();
   const real_t a = weight/(6*std::sqrt(I1b_I2b));
   ie.Assemble_ddI1b(a*ie.Get_I2b(), A.GetData());
   ie.Assemble_ddI2b(a*ie.Get_I1b(), A.GetData());
   ie.Assemble_TProd(-a/(2*I1b_I2b), X_data, A.GetData());
}

real_t TMOP_Metric_302::EvalWMatrixForm(const DenseMatrix &Jpt) const
{
   // mu_301 = |J|^2 |J^{-1}|^2 / 9 - 1.
   ie.SetJacobian(Jpt.GetData());
   DenseMatrix inv(3);
   CalcInverse(Jpt, inv);
   return Jpt.FNorm2() * inv.FNorm2() / 9.0 - 1.0;
}

real_t TMOP_Metric_302::EvalW(const DenseMatrix &Jpt) const
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
                                const real_t weight,
                                DenseMatrix &A) const
{
   // P  = (I1b/9)*dI2b + (I2b/9)*dI1b
   // dP = (dI2b x dI1b)/9 + (I1b/9)*ddI2b + (dI1b x dI2b)/9 + (I2b/9)*ddI1b
   //    = (dI2b x dI1b + dI1b x dI2b)/9 + (I1b/9)*ddI2b + (I2b/9)*ddI1b
   ie.SetJacobian(Jpt.GetData());
   ie.SetDerivativeMatrix(DS.Height(), DS.GetData());
   const real_t c1 = weight/9;
   ie.Assemble_TProd(c1, ie.Get_dI1b(), ie.Get_dI2b(), A.GetData());
   ie.Assemble_ddI2b(c1*ie.Get_I1b(), A.GetData());
   ie.Assemble_ddI1b(c1*ie.Get_I2b(), A.GetData());
}

real_t TMOP_Metric_303::EvalWMatrixForm(const DenseMatrix &Jpt) const
{
   // mu_303 = |J|^2 / 3 / det(J)^(2/3) - 1.
   ie.SetJacobian(Jpt.GetData());
   return Jpt.FNorm2() / 3.0 / pow(Jpt.Det(), 2.0 / 3.0) - 1.0;
}

real_t TMOP_Metric_303::EvalW(const DenseMatrix &Jpt) const
{
   // mu_303 = |J|^2 / 3 / det(J)^(2/3) - 1 = I1b/3 - 1.
   ie.SetJacobian(Jpt.GetData());
   return ie.Get_I1b()/3.0 - 1.0;
}

void TMOP_Metric_303::EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const
{
   // mu_304 = I1b/3 - 1.
   // P      = dI1b/3.
   ie.SetJacobian(Jpt.GetData());
   P.Set(1./3., ie.Get_dI1b());
}

void TMOP_Metric_303::AssembleH(const DenseMatrix &Jpt,
                                const DenseMatrix &DS,
                                const real_t weight,
                                DenseMatrix &A) const
{
   // P  = dI1b/3.
   // dP = ddI1b/3.
   ie.SetJacobian(Jpt.GetData());
   ie.SetDerivativeMatrix(DS.Height(), DS.GetData());
   ie.Assemble_ddI1b(weight/3., A.GetData());
}

real_t TMOP_Metric_304::EvalWMatrixForm(const DenseMatrix &Jpt) const
{
   // mu_304 = |J|^3 / 3^(3/2) / det(J) - 1
   const real_t fnorm = Jpt.FNorm();
   return fnorm * fnorm * fnorm / pow(3.0, 1.5) / Jpt.Det() - 1.0;
}

real_t TMOP_Metric_304::EvalW(const DenseMatrix &Jpt) const
{
   // mu_304 = (I1b/3)^3/2 - 1.
   ie.SetJacobian(Jpt.GetData());
   return pow(ie.Get_I1b()/3.0, 1.5) - 1.0;
}

void TMOP_Metric_304::EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const
{
   // mu_304 = (I1b/3)^3/2 - 1.
   // P      = 3/2 * (I1b/3)^1/2 * dI1b / 3 = 1/2 * (I1b/3)^1/2 * dI1b.
   ie.SetJacobian(Jpt.GetData());
   P.Set(0.5 * sqrt(ie.Get_I1b()/3.0), ie.Get_dI1b());
}

void TMOP_Metric_304::AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                                const real_t weight, DenseMatrix &A) const
{
   // P  = 1/2 * (I1b/3)^1/2 * dI1b.
   // dP = 1/12 * (I1b/3)^(-1/2) * (dI1b x dI1b) + 1/2 * (I1b/3)^1/2 * ddI1b.
   ie.SetJacobian(Jpt.GetData());
   ie.SetDerivativeMatrix(DS.Height(), DS.GetData());
   ie.Assemble_TProd(weight / 12.0 / sqrt(ie.Get_I1b()/3.0),
                     ie.Get_dI1b(), A.GetData());
   ie.Assemble_ddI1b(weight / 2.0 * sqrt(ie.Get_I1b()/3.0), A.GetData());
}

real_t TMOP_Metric_311::EvalW(const DenseMatrix &Jpt) const
{
   // mu_311 = (det(J) - 1)^2 - det(J) + (det(J)^2 + eps)^{1/2}
   //        = (I3b - 1)^2 - I3b + sqrt(I3b^2 + eps)
   ie.SetJacobian(Jpt.GetData());
   const real_t I3b = ie.Get_I3b();
   return (I3b - 1.0)*(I3b - 1.0) - I3b + std::sqrt(I3b*I3b + eps);
}

void TMOP_Metric_311::EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const
{
   ie.SetJacobian(Jpt.GetData());
   const real_t I3b = ie.Get_I3b();
   const real_t c = 2*I3b-3+(I3b)/(std::pow((I3b*I3b+eps),0.5));
   P.Set(c, ie.Get_dI3b());
}

void TMOP_Metric_311::AssembleH(const DenseMatrix &Jpt,
                                const DenseMatrix &DS,
                                const real_t weight,
                                DenseMatrix &A) const
{
   ie.SetJacobian(Jpt.GetData());
   ie.SetDerivativeMatrix(DS.Height(), DS.GetData());
   const real_t I3b = ie.Get_I3b();
   const real_t c0 = I3b*I3b+eps;
   const real_t c1 = 2 + 1/(pow(c0,0.5)) - I3b*I3b/(pow(c0,1.5));
   const real_t c2 = 2*I3b - 3 + I3b/(pow(c0,0.5));
   ie.Assemble_TProd(weight*c1, ie.Get_dI3b(), A.GetData());
   ie.Assemble_ddI3b(c2*weight, A.GetData());
}

real_t TMOP_Metric_313::EvalW(const DenseMatrix &Jpt) const
{
   ie.SetJacobian(Jpt.GetData());

   const real_t I3b = ie.Get_I3b();
   real_t d = I3b - min_detT;
   if (d < 0.0 && min_detT == 0.0)
   {
      // The mesh has been untangled, but it's still possible to get negative
      // detJ in FD calculations, as they move the nodes around with some small
      // increments and can produce negative determinants. Thus we put a small
      // value in the denominator. Note that here I3b < 0.
      d = - I3b * 0.1;
   }

   const real_t c = std::pow(d, -2.0/3.0);

   return ie.Get_I1() * c / 3.0;
}

void TMOP_Metric_313::EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const
{
   MFEM_ABORT("Metric not implemented yet.");
}

void TMOP_Metric_313::AssembleH(const DenseMatrix &Jpt,
                                const DenseMatrix &DS,
                                const real_t weight,
                                DenseMatrix &A) const
{
   MFEM_ABORT("Metric not implemented yet.");
}

real_t TMOP_Metric_315::EvalW(const DenseMatrix &Jpt) const
{
   // mu_315 = mu_15_3D = (det(J) - 1)^2
   ie.SetJacobian(Jpt.GetData());
   const real_t c1 = ie.Get_I3b() - 1.0;
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
                                const real_t weight,
                                DenseMatrix &A) const
{
   // P  = 2*(I3b - 1)*dI3b
   // dP = 2*(dI3b x dI3b) + 2*(I3b - 1)*ddI3b
   ie.SetJacobian(Jpt.GetData());
   ie.SetDerivativeMatrix(DS.Height(), DS.GetData());
   ie.Assemble_TProd(2*weight, ie.Get_dI3b(), A.GetData());
   ie.Assemble_ddI3b(2*weight*(ie.Get_I3b() - 1.0), A.GetData());
}

real_t TMOP_Metric_316::EvalWMatrixForm(const DenseMatrix &Jpt) const
{
   // mu_316 = 0.5 (det(J) + 1/det(J)) - 1.
   return 0.5 * (Jpt.Det() + 1.0 / Jpt.Det()) - 1.0;
}

real_t TMOP_Metric_316::EvalW(const DenseMatrix &Jpt) const
{
   // mu_316 = mu_16_3D = 0.5*(I3b + 1/I3b) - 1
   ie.SetJacobian(Jpt.GetData());
   const real_t I3b = ie.Get_I3b();
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
                                const real_t weight,
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

real_t TMOP_Metric_318::EvalWMatrixForm(const DenseMatrix &Jpt) const
{
   // mu_318 = 0.5 (det(J)^2 + 1/det(J)^2) - 1.
   real_t d = Jpt.Det();
   return 0.5 * (d*d + 1.0 / (d*d)) - 1.0;
}

real_t TMOP_Metric_318::EvalW(const DenseMatrix &Jpt) const
{
   // mu_318 = mu_77_3D = 0.5 * (I3 + 1/I3) - 1.
   ie.SetJacobian(Jpt.GetData());
   const real_t I3 = ie.Get_I3();
   return 0.5*(I3 + 1.0/I3) - 1.0;
}

void TMOP_Metric_318::EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const
{
   // mu_318 = mu_77_3D = 0.5*(I3 + 1/I3) - 1.
   // P = 0.5*(1 - 1/I3^2)*dI3 = (0.5 - 0.5/I3^2)*dI3.
   ie.SetJacobian(Jpt.GetData());
   P.Set(0.5 - 0.5/(ie.Get_I3()*ie.Get_I3()), ie.Get_dI3());
}

void TMOP_Metric_318::AssembleH(const DenseMatrix &Jpt,
                                const DenseMatrix &DS,
                                const real_t weight,
                                DenseMatrix &A) const
{
   // P = (0.5 - 0.5/I3^2)*dI3.
   // dP =  (1/I3^3)*(dI3 x dI3) +(0.5 - 0.5/I3^2)*ddI3
   ie.SetJacobian(Jpt.GetData());
   ie.SetDerivativeMatrix(DS.Height(), DS.GetData());
   const real_t i3 = ie.Get_I3();
   ie.Assemble_TProd(weight/(i3 * i3 * i3), ie.Get_dI3(), A.GetData());
   ie.Assemble_ddI3(weight*(0.5 - 0.5 / (i3 * i3)), A.GetData());
}

real_t TMOP_Metric_321::EvalWMatrixForm(const DenseMatrix &Jpt) const
{
   // mu_321 = |J - J^-t|^2.
   ie.SetJacobian(Jpt.GetData());
   DenseMatrix invt(3);
   CalcInverseTranspose(Jpt, invt);
   invt.Add(-1.0, Jpt);
   return invt.FNorm2();
}

real_t TMOP_Metric_321::EvalW(const DenseMatrix &Jpt) const
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
   const real_t I3 = ie.Get_I3();
   Add(1.0/I3, ie.Get_dI2(),
       -2*ie.Get_I2()/(I3*ie.Get_I3b()), ie.Get_dI3b(), P);
   P += ie.Get_dI1();
}

void TMOP_Metric_321::AssembleH(const DenseMatrix &Jpt,
                                const DenseMatrix &DS,
                                const real_t weight,
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
   const real_t c0 = 1.0/ie.Get_I3b();
   const real_t c1 = weight*c0*c0;
   const real_t c2 = -2*c0*c1;
   const real_t c3 = c2*ie.Get_I2();
   ie.Assemble_ddI1(weight, A.GetData());
   ie.Assemble_ddI2(c1, A.GetData());
   ie.Assemble_ddI3b(c3, A.GetData());
   ie.Assemble_TProd(c2, ie.Get_dI2(), ie.Get_dI3b(), A.GetData());
   ie.Assemble_TProd(-3*c0*c3, ie.Get_dI3b(), A.GetData());
}

real_t TMOP_Metric_322::EvalWMatrixForm(const DenseMatrix &Jpt) const
{
   // mu_322 = 1 / (6 det(J)) |J - adj(J)^t|^2
   DenseMatrix adj_J_t(3);
   CalcAdjugateTranspose(Jpt, adj_J_t);
   adj_J_t *= -1.0;
   adj_J_t.Add(1.0, Jpt);
   return 1.0 / 6.0 / Jpt.Det() * adj_J_t.FNorm2();
}

real_t TMOP_Metric_322::EvalW(const DenseMatrix &Jpt) const
{
   // mu_322 = 1 / (6 det(J)) |J - adj(J)^t|^2
   //        = 1 / (6 det(J)) |J|^2 + 1/6 det(J) |J^{-1}|^2 - 1
   //        = I1b / (I3b^-1/3) / 6 + I2b (I3b^1/3) / 6 - 1
   ie.SetJacobian(Jpt.GetData());

   return ie.Get_I1b() / pow(ie.Get_I3b(), 1.0/3.0) / 6.0 +
          ie.Get_I2b() * pow(ie.Get_I3b(), 1.0/3.0) / 6.0 - 1.0;
}

void TMOP_Metric_322::EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const
{
   // mu_322 = I1b (I3b^-1/3) / 6 + I2b (I3b^1/3) / 6 - 1
   // P      =   1/6 (I3b^-1/3) dI1b - 1/18 I1b (I3b^-4/3) dI3b
   //          + 1/6 (I3b^1/3) dI2b  + 1/18 I2b (I3b^-2/3) dI3b
   ie.SetJacobian(Jpt.GetData());
   P.Set(1.0/6.0 * pow(ie.Get_I3b(), -1.0/3.0),
         ie.Get_dI1b());
   P.Add(-1.0/18.0 * ie.Get_I1b() * pow(ie.Get_I3b(), -4.0/3.0),
         ie.Get_dI3b());
   P.Add(1.0/6.0 * pow(ie.Get_I3b(), 1.0/3.0),
         ie.Get_dI2b());
   P.Add(1.0/18.0 * ie.Get_I2b() * pow(ie.Get_I3b(), -2.0/3.0),
         ie.Get_dI3b());
}

void TMOP_Metric_322::AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                                const real_t weight, DenseMatrix &A) const
{
   //  P =   1/6 (I3b^-1/3) dI1b - 1/18 I1b (I3b^-4/3) dI3b
   //      + 1/6 (I3b^1/3) dI2b  + 1/18 I2b (I3b^-2/3) dI3b
   // dP =   1/6 (I3b^-1/3) ddI1b - 1/18 (I3b^-4/3) (dI1b x dI3b)
   //      - 1/18 I1b (I3b^-4/3) ddI3b
   //      - 1/18 (I3b^-4/3) (dI3b x dI1b)
   //      + 2/27 I1b (I3b^-7/3) (dI3b x dI3b)
   //      + 1/6 (I3b^1/3) ddI2b + 1/18 (I3b^-2/3) (dI2b x dI3b)
   //      + 1/18 I2b (I3b^-2/3) ddI3b
   //      + 1/18 (I3b^-2/3) (dI3b x dI2b)
   //      - 1/27 I2b (I3b^-5/3) (dI3b x dI3b)
   ie.SetJacobian(Jpt.GetData());
   ie.SetDerivativeMatrix(DS.Height(), DS.GetData());
   const real_t p13 = weight * pow(ie.Get_I3b(),  1.0/3.0),
                m13 = weight * pow(ie.Get_I3b(), -1.0/3.0),
                m23 = weight * pow(ie.Get_I3b(), -2.0/3.0),
                m43 = weight * pow(ie.Get_I3b(), -4.0/3.0),
                m53 = weight * pow(ie.Get_I3b(), -5.0/3.0),
                m73 = weight * pow(ie.Get_I3b(), -7.0/3.0);
   ie.Assemble_ddI1b(1.0/6.0 * m13, A.GetData());
   // Combines - 1/18 (I3b^-4/3) (dI1b x dI3b) - 1/18 (I3b^-4/3) (dI3b x dI1b).
   ie.Assemble_TProd(-1.0/18.0 * m43,
                     ie.Get_dI1b(), ie.Get_dI3b(), A.GetData());
   ie.Assemble_ddI3b(-1.0/18.0 * ie.Get_I1b() * m43, A.GetData());
   ie.Assemble_TProd(2.0/27.0 * ie.Get_I1b() * m73,
                     ie.Get_dI3b(), A.GetData());
   ie.Assemble_ddI2b(1.0/6.0 * p13, A.GetData());
   // Combines + 1/18 (I3b^-2/3) (dI2b x dI3b) + 1/18 (I3b^-2/3) (dI3b x dI2b).
   ie.Assemble_TProd(1.0/18.0 * m23,
                     ie.Get_dI2b(), ie.Get_dI3b(), A.GetData());
   ie.Assemble_ddI3b(1.0/18.0 * ie.Get_I2b() * m23, A.GetData());
   ie.Assemble_TProd(-1.0/27.0 * ie.Get_I2b() * m53,
                     ie.Get_dI3b(), A.GetData());
}

real_t TMOP_Metric_323::EvalWMatrixForm(const DenseMatrix &Jpt) const
{
   // mu_323 = |J|^3 - 3 sqrt(3) ln(det(J)) - 3 sqrt(3).
   real_t fnorm = Jpt.FNorm();
   return fnorm * fnorm * fnorm - 3.0 * sqrt(3.0) * (log(Jpt.Det()) + 1.0);
}

real_t TMOP_Metric_323::EvalW(const DenseMatrix &Jpt) const
{
   // mu_323 = I1^3/2 - 3 sqrt(3) ln(I3b) - 3 sqrt(3).
   ie.SetJacobian(Jpt.GetData());
   return pow(ie.Get_I1(), 1.5) - 3.0 * sqrt(3.0) * (log(ie.Get_I3b()) + 1.0);
}

void TMOP_Metric_323::EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const
{
   // mu_323 = I1^3/2 - 3 sqrt(3) ln(I3b) - 3 sqrt(3).
   // P      = 3/2 (I1^1/2) dI1 - 3 sqrt(3) (I3b^-1) dI3b.
   ie.SetJacobian(Jpt.GetData());
   P.Set(1.5 * sqrt(ie.Get_I1()), ie.Get_dI1());
   P.Add(- 3.0 * sqrt(3.0) / ie.Get_I3b(), ie.Get_dI3b());
}

void TMOP_Metric_323::AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                                const real_t weight, DenseMatrix &A) const
{
   // P  =   3/2 (I1^1/2) dI1 - 3 sqrt(3) (I3b^-1) dI3b
   // dP =   3/2 (I1^1/2) ddI1 + 3/4 (I1^-1/2) (dI1 x dI1)
   //      - 3 sqrt(3) (I3b^-1) ddI3b + 3 sqrt(3) (I3b^-2) (dI3b x dI3b)
   ie.SetJacobian(Jpt.GetData());
   ie.SetDerivativeMatrix(DS.Height(), DS.GetData());
   ie.Assemble_ddI1(weight * 1.5 * sqrt(ie.Get_I1()), A.GetData());
   ie.Assemble_TProd(weight * 0.75 / sqrt(ie.Get_I1()),
                     ie.Get_dI1(), A.GetData());
   ie.Assemble_ddI3b(- weight * 3.0 * sqrt(3.0) / ie.Get_I3b(), A.GetData());
   ie.Assemble_TProd(weight * 3.0 * sqrt(3.0) / ie.Get_I3b() / ie.Get_I3b(),
                     ie.Get_dI3b(), A.GetData());
}

// mu_342 = 1/(tau^0.5)|T-I|^2
real_t TMOP_Metric_342::EvalWMatrixForm(const DenseMatrix &Jpt) const
{
   int matsize = Jpt.TotalSize();
   std::vector<AD1Type> T(matsize), W(matsize);
   for (int i=0; i<matsize; i++) { T[i] = AD1Type{Jpt.GetData()[i], 0.0}; }
   return mu342_ad(T, W).value;
}

void TMOP_Metric_342::EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const
{
   ADGrad(mu342_ad<AD1Type>, P, Jpt);
   return;
}

void TMOP_Metric_342::AssembleH(const DenseMatrix &Jpt,
                                const DenseMatrix &DS,
                                const real_t weight,
                                DenseMatrix &A) const
{
   const int dim = Jpt.Height();
   DenseTensor H(dim, dim, dim*dim); H = 0.0;
   ADHessian(mu342_ad<AD2Type>, H, Jpt);
   this->DefaultAssembleH(H,DS,weight,A);
}

real_t TMOP_Metric_352::EvalW(const DenseMatrix &Jpt) const
{
   // mu_352 = 0.5*(det(J) - 1)^2 / (det(J) - tau0)
   ie.SetJacobian(Jpt.GetData());
   const real_t I3b = ie.Get_I3b();
   return 0.5*(I3b - 1.0)*(I3b - 1.0)/(I3b - tau0);
}

void TMOP_Metric_352::EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const
{
   // mu_352 = 0.5*(det(J) - 1)^2 / (det(J) - tau0)
   // P = (I3b - 1)/(I3b - tau0)*dI3b + 0.5*(I3b - 1)^2*(-1/(I3b - tau0)^2)*dI3b
   //   = [ (I3b - 1)/(I3b - tau0) - 0.5*(I3b - 1)^2/(I3b - tau0)^2 ] * dI3b
   //   = (c - 0.5*c*c) * dI3b
   ie.SetJacobian(Jpt.GetData());
   const real_t I3b = ie.Get_I3b();
   const real_t c = (I3b - 1.0)/(I3b - tau0);
   P.Set(c - 0.5*c*c, ie.Get_dI3b());
}

void TMOP_Metric_352::AssembleH(const DenseMatrix &Jpt,
                                const DenseMatrix &DS,
                                const real_t weight,
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
   const real_t I3b = ie.Get_I3b();
   const real_t c0 = 1.0/(I3b - tau0);
   const real_t c = c0*(I3b - 1.0);
   ie.Assemble_TProd(weight*c0*(1.0 - c)*(1.0 - c), ie.Get_dI3b(), A.GetData());
   ie.Assemble_ddI3b(weight*(c - 0.5*c*c), A.GetData());
}

real_t TMOP_Metric_360::EvalWMatrixForm(const DenseMatrix &Jpt) const
{
   // mu_360 = |J|^3 / 3^(3/2) - det(J)
   const real_t fnorm = Jpt.FNorm();
   return fnorm * fnorm * fnorm / pow(3.0, 1.5) - Jpt.Det();
}

real_t TMOP_Metric_360::EvalW(const DenseMatrix &Jpt) const
{
   // mu_360 = (I1/3)^(3/2) - I3b.
   ie.SetJacobian(Jpt.GetData());
   return pow(ie.Get_I1()/3.0, 1.5) - ie.Get_I3b();
}

void TMOP_Metric_360::EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const
{
   // mu_360 = (I1/3)^(3/2) - I3b.
   // P      = 3/2 * (I1/3)^1/2 * dI1 / 3 - dI3b
   //        = 1/2 * (I1/3)^1/2 * dI1 - dI3b.
   ie.SetJacobian(Jpt.GetData());
   Add(0.5 * sqrt(ie.Get_I1()/3.0), ie.Get_dI1(), -1.0, ie.Get_dI3b(), P);
}

void TMOP_Metric_360::AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                                const real_t weight, DenseMatrix &A) const
{
   // P  = 1/2 * (I1/3)^1/2 * dI1 - dI3b.
   // dP = 1/12 * (I1/3)^(-1/2) * (dI1 x dI1) + 1/2 * (I1/3)^1/2 * ddI1 - ddI3b
   ie.SetJacobian(Jpt.GetData());
   ie.SetDerivativeMatrix(DS.Height(), DS.GetData());
   ie.Assemble_TProd(weight / 12.0 / sqrt(ie.Get_I1()/3.0),
                     ie.Get_dI1(), A.GetData());
   ie.Assemble_ddI1(weight / 2.0 * sqrt(ie.Get_I1()/3.0), A.GetData());
   ie.Assemble_ddI3b(-weight, A.GetData());
}

real_t TMOP_AMetric_011::EvalWMatrixForm(const DenseMatrix &Jpt) const
{
   MFEM_VERIFY(Jtr != NULL,
               "Requires a target Jacobian, use SetTargetJacobian().");
   int matsize = Jpt.TotalSize();
   std::vector<AD1Type> T(matsize), W(matsize);
   for (int i=0; i<matsize; i++)
   {
      T[i] = AD1Type{Jpt.GetData()[i], 0.0};
      W[i] = AD1Type{Jtr->GetData()[i], 0.0};
   }
   return nu11_ad(T, W).value;
}

void TMOP_AMetric_011::EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const
{
   ADGrad(nu11_ad<AD1Type>, P, Jpt, Jtr);
   return;
}

void TMOP_AMetric_011::EvalPW(const DenseMatrix &Jpt, DenseMatrix &PW) const
{
   ADGrad(nu11_ad<AD1Type>, PW, Jpt, Jtr, false);
   return;
}

void TMOP_AMetric_011::AssembleH(const DenseMatrix &Jpt,
                                 const DenseMatrix &DS,
                                 const real_t weight,
                                 DenseMatrix &A) const
{
   const int dim = Jpt.Height();
   DenseTensor H(dim, dim, dim*dim); H = 0.0;
   ADHessian(nu11_ad<AD2Type>, H, Jpt, Jtr);
   this->DefaultAssembleH(H,DS,weight,A);
}

real_t TMOP_AMetric_014::EvalWMatrixForm(const DenseMatrix &Jpt) const
{
   MFEM_VERIFY(Jtr != NULL,
               "Requires a target Jacobian, use SetTargetJacobian().");
   int matsize = Jpt.TotalSize();
   std::vector<AD1Type> T(matsize), W(matsize);
   for (int i=0; i<matsize; i++)
   {
      T[i] = AD1Type{Jpt.GetData()[i], 0.0};
      W[i] = AD1Type{Jtr->GetData()[i], 0.0};
   }
   return nu14_ad(T, W).value;
}

void TMOP_AMetric_014::EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const
{
   ADGrad(nu14_ad<AD1Type>, P, Jpt, Jtr);
   return;
}

void TMOP_AMetric_014::EvalPW(const DenseMatrix &Jpt, DenseMatrix &PW) const
{
   ADGrad(nu14_ad<AD1Type>, PW, Jpt, Jtr, false);
   return;
}

void TMOP_AMetric_014::AssembleH(const DenseMatrix &Jpt,
                                 const DenseMatrix &DS,
                                 const real_t weight,
                                 DenseMatrix &A) const
{
   const int dim = Jpt.Height();
   DenseTensor H(dim, dim, dim*dim); H = 0.0;
   ADHessian(nu14_ad<AD2Type>, H, Jpt, Jtr);
   this->DefaultAssembleH(H,DS,weight,A);
}

real_t TMOP_AMetric_036::EvalWMatrixForm(const DenseMatrix &Jpt) const
{
   MFEM_VERIFY(Jtr != NULL,
               "Requires a target Jacobian, use SetTargetJacobian().");
   int matsize = Jpt.TotalSize();
   std::vector<AD1Type> T(matsize), W(matsize);
   for (int i=0; i<matsize; i++)
   {
      T[i] = AD1Type{Jpt.GetData()[i], 0.0};
      W[i] = AD1Type{Jtr->GetData()[i], 0.0};
   }
   return nu36_ad(T, W).value;
}

void TMOP_AMetric_036::EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const
{
   ADGrad(nu36_ad<AD1Type>, P, Jpt, Jtr);
   return;
}

void TMOP_AMetric_036::EvalPW(const DenseMatrix &Jpt, DenseMatrix &PW) const
{
   ADGrad(nu36_ad<AD1Type>, PW, Jpt, Jtr, false);
   return;
}

void TMOP_AMetric_036::AssembleH(const DenseMatrix &Jpt,
                                 const DenseMatrix &DS,
                                 const real_t weight,
                                 DenseMatrix &A) const
{
   const int dim = Jpt.Height();
   DenseTensor H(dim, dim, dim*dim); H = 0.0;
   ADHessian(nu36_ad<AD2Type>, H, Jpt, Jtr);
   this->DefaultAssembleH(H,DS,weight,A);
}

real_t TMOP_AMetric_050::EvalWMatrixForm(const DenseMatrix &Jpt) const
{
   MFEM_VERIFY(Jtr != NULL,
               "Requires a target Jacobian, use SetTargetJacobian().");
   int matsize = Jpt.TotalSize();
   std::vector<AD1Type> T(matsize), W(matsize);
   for (int i=0; i<matsize; i++)
   {
      T[i] = AD1Type{Jpt.GetData()[i], 0.0};
      W[i] = AD1Type{Jtr->GetData()[i], 0.0};
   }
   return nu50_ad(T, W).value;
}

void TMOP_AMetric_050::EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const
{
   ADGrad(nu50_ad<AD1Type>, P, Jpt, Jtr);
   return;
}

void TMOP_AMetric_050::EvalPW(const DenseMatrix &Jpt, DenseMatrix &PW) const
{
   ADGrad(nu50_ad<AD1Type>, PW, Jpt, Jtr, false);
   return;
}

void TMOP_AMetric_050::AssembleH(const DenseMatrix &Jpt,
                                 const DenseMatrix &DS,
                                 const real_t weight,
                                 DenseMatrix &A) const
{
   const int dim = Jpt.Height();
   DenseTensor H(dim, dim, dim*dim); H = 0.0;
   ADHessian(nu50_ad<AD2Type>, H, Jpt, Jtr);
   this->DefaultAssembleH(H,DS,weight,A);
}

real_t TMOP_AMetric_051::EvalWMatrixForm(const DenseMatrix &Jpt) const
{
   MFEM_VERIFY(Jtr != NULL,
               "Requires a target Jacobian, use SetTargetJacobian().");
   int matsize = Jpt.TotalSize();
   std::vector<AD1Type> T(matsize), W(matsize);
   for (int i=0; i<matsize; i++)
   {
      T[i] = AD1Type{Jpt.GetData()[i], 0.0};
      W[i] = AD1Type{Jtr->GetData()[i], 0.0};
   }
   return nu51_ad(T, W).value;
}

void TMOP_AMetric_051::EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const
{
   ADGrad(nu51_ad<AD1Type>, P, Jpt, Jtr);
   return;
}

void TMOP_AMetric_051::EvalPW(const DenseMatrix &Jpt, DenseMatrix &PW) const
{
   ADGrad(nu51_ad<AD1Type>, PW, Jpt, Jtr, false);
   return;
}

void TMOP_AMetric_051::AssembleH(const DenseMatrix &Jpt,
                                 const DenseMatrix &DS,
                                 const real_t weight,
                                 DenseMatrix &A) const
{
   const int dim = Jpt.Height();
   DenseTensor H(dim, dim, dim*dim); H = 0.0;
   ADHessian(nu51_ad<AD2Type>, H, Jpt, Jtr);
   this->DefaultAssembleH(H,DS,weight,A);
}

real_t TMOP_AMetric_107::EvalWMatrixForm(const DenseMatrix &Jpt) const
{
   MFEM_VERIFY(Jtr != NULL,
               "Requires a target Jacobian, use SetTargetJacobian().");
   int matsize = Jpt.TotalSize();
   std::vector<AD1Type> T(matsize), W(matsize);
   for (int i=0; i<matsize; i++)
   {
      T[i] = AD1Type{Jpt.GetData()[i], 0.0};
      W[i] = AD1Type{Jtr->GetData()[i], 0.0};
   }
   return nu107_ad(T, W).value;
}

void TMOP_AMetric_107::EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const
{
   ADGrad(nu107_ad<AD1Type>, P, Jpt, Jtr);
   return;
}

void TMOP_AMetric_107::EvalPW(const DenseMatrix &Jpt, DenseMatrix &PW) const
{
   ADGrad(nu107_ad<AD1Type>, PW, Jpt, Jtr, false);
   return;
}

void TMOP_AMetric_107::AssembleH(const DenseMatrix &Jpt,
                                 const DenseMatrix &DS,
                                 const real_t weight,
                                 DenseMatrix &A) const
{
   const int dim = Jpt.Height();
   DenseTensor H(dim, dim, dim*dim); H = 0.0;
   ADHessian(nu107_ad<AD2Type>, H, Jpt, Jtr);
   this->DefaultAssembleH(H,DS,weight,A);
}

void TargetConstructor::ComputeAvgVolume() const
{
   MFEM_VERIFY(nodes, "Nodes are not given!");
   MFEM_ASSERT(avg_volume == 0.0, "The average volume is already computed!");

   Mesh *mesh = nodes->FESpace()->GetMesh();
   const int NE = mesh->GetNE();
   IsoparametricTransformation Tr;
   real_t volume = 0.0;

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
      real_t area_NE[4];
      area_NE[0] = volume; area_NE[1] = NE;
      MPI_Allreduce(area_NE, area_NE + 2, 2, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                    comm);
      avg_volume = (ncmesh == NULL) ?
                   area_NE[2] / area_NE[3] : area_NE[2] / ncmesh->GetNumRootElements();
   }
#endif
}

void TargetConstructor::ComputeAllElementTargets_Fallback(
   const FiniteElementSpace &fes,
   const IntegrationRule &ir,
   const Vector &xe,
   DenseTensor &Jtr) const
{
   // Fallback to the 1-element method, ComputeElementTargets()

   // When UsesPhysicalCoordinates() == true, we assume 'xe' uses
   // ElementDofOrdering::LEXICOGRAPHIC iff 'fe' is a TensorFiniteElement.

   const Mesh *mesh = fes.GetMesh();
   const int NE = mesh->GetNE();
   // Quick return for empty processors:
   if (NE == 0) { return; }
   const int dim = mesh->Dimension();
   MFEM_VERIFY(mesh->GetNumGeometries(dim) <= 1,
               "mixed meshes are not supported");
   MFEM_VERIFY(!fes.IsVariableOrder(), "variable orders are not supported");
   const FiniteElement &fe = *fes.GetTypicalFE();
   const int sdim = fes.GetVDim();
   const int nvdofs = sdim*fe.GetDof();
   MFEM_VERIFY(!UsesPhysicalCoordinates() ||
               xe.Size() == NE*nvdofs, "invalid input Vector 'xe'!");
   const int NQ = ir.GetNPoints();
   const Array<int> *dof_map = nullptr;
   if (UsesPhysicalCoordinates())
   {
      const TensorBasisElement *tfe =
         dynamic_cast<const TensorBasisElement *>(&fe);
      if (tfe)
      {
         dof_map = &tfe->GetDofMap();
         if (dof_map->Size() == 0) { dof_map = nullptr; }
      }
   }

   Vector elfun_lex, elfun_nat;
   DenseTensor J;
   xe.HostRead();
   Jtr.HostWrite();
   if (UsesPhysicalCoordinates() && dof_map != nullptr)
   {
      elfun_nat.SetSize(nvdofs);
   }
   for (int e = 0; e < NE; e++)
   {
      if (UsesPhysicalCoordinates())
      {
         if (!dof_map)
         {
            elfun_nat.SetDataAndSize(xe.GetData()+e*nvdofs, nvdofs);
         }
         else
         {
            elfun_lex.SetDataAndSize(xe.GetData()+e*nvdofs, nvdofs);
            const int ndofs = fe.GetDof();
            for (int d = 0; d < sdim; d++)
            {
               for (int i_lex = 0; i_lex < ndofs; i_lex++)
               {
                  elfun_nat[(*dof_map)[i_lex]+d*ndofs] =
                     elfun_lex[i_lex+d*ndofs];
               }
            }
         }
      }
      J.UseExternalData(Jtr(e*NQ).Data(), sdim, dim, NQ);
      ComputeElementTargets(e, fe, ir, elfun_nat, J);
   }
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
   MFEM_CONTRACT_VAR(elfun);
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
         real_t el_volume = avg_volume;
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
         real_t detW;

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
               const real_t det = Jtr(i).Det();
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

void TargetConstructor::ComputeElementTargetsGradient(
   const IntegrationRule &ir,
   const Vector &elfun,
   IsoparametricTransformation &Tpr,
   DenseTensor &dJtr) const
{
   MFEM_CONTRACT_VAR(elfun);
   MFEM_ASSERT(target_type == IDEAL_SHAPE_UNIT_SIZE || nodes != NULL, "");

   // TODO: Compute derivative for targets with GIVEN_SHAPE or/and GIVEN_SIZE
   for (int i = 0; i < Tpr.GetFE()->GetDim()*ir.GetNPoints(); i++)
   { dJtr(i) = 0.; }
}

void AnalyticAdaptTC::SetAnalyticTargetSpec(Coefficient *sspec,
                                            VectorCoefficient *vspec,
                                            TMOPMatrixCoefficient *mspec)
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

void AnalyticAdaptTC::ComputeElementTargetsGradient(const IntegrationRule &ir,
                                                    const Vector &elfun,
                                                    IsoparametricTransformation &Tpr,
                                                    DenseTensor &dJtr) const
{
   const FiniteElement *fe = Tpr.GetFE();
   DenseMatrix point_mat;
   point_mat.UseExternalData(elfun.GetData(), fe->GetDof(), fe->GetDim());

   switch (target_type)
   {
      case GIVEN_FULL:
      {
         MFEM_VERIFY(matrix_tspec != NULL,
                     "Target type GIVEN_FULL requires a TMOPMatrixCoefficient.");

         for (int d = 0; d < fe->GetDim(); d++)
         {
            for (int i = 0; i < ir.GetNPoints(); i++)
            {
               const IntegrationPoint &ip = ir.IntPoint(i);
               Tpr.SetIntPoint(&ip);
               DenseMatrix &dJtr_i = dJtr(i + d*ir.GetNPoints());
               matrix_tspec->EvalGrad(dJtr_i, Tpr, ip, d);
            }
         }
         break;
      }
      default:
         MFEM_ABORT("Incompatible target type for analytic adaptation!");
   }
}

namespace internal
{

// mfem::forall-based copy kernel -- used by protected methods below.
// Needed as a workaround for the nvcc restriction that methods with mfem::forall
// in them must to be public.
static inline void device_copy(real_t *d_dest, const real_t *d_src, int size)
{
   mfem::forall(size, [=] MFEM_HOST_DEVICE (int i) { d_dest[i] = d_src[i]; });
}

} // namespace internal

#ifdef MFEM_USE_MPI
void DiscreteAdaptTC::FinalizeParDiscreteTargetSpec(const ParGridFunction &t)
{
   MFEM_VERIFY(adapt_eval, "SetAdaptivityEvaluator() has not been called!")
   MFEM_VERIFY(ncomp > 0, "No target specifications have been set!");

   ParFiniteElementSpace *ptspec_fes = t.ParFESpace();

   tspec_sav = tspec;

   delete tspec_fesv;
   tspec_fesv = new FiniteElementSpace(ptspec_fes->GetMesh(),
                                       ptspec_fes->FEColl(), ncomp);

   delete ptspec_fesv;
   ptspec_fesv = new ParFiniteElementSpace(ptspec_fes->GetParMesh(),
                                           ptspec_fes->FEColl(), ncomp);

   delete tspec_pgf;
   tspec_pgf = new ParGridFunction(ptspec_fesv, tspec);
   tspec_gf = tspec_pgf;

   adapt_eval->SetParMetaInfo(*ptspec_fes->GetParMesh(), *ptspec_fesv);
   adapt_eval->SetInitialField(*ptspec_fes->GetMesh()->GetNodes(), tspec);
}

void DiscreteAdaptTC::ParUpdateAfterMeshTopologyChange()
{
   ptspec_fesv->Update();
   if (tspec_fesv)
   {
      delete tspec_fesv;
      tspec_fesv = new FiniteElementSpace(ptspec_fesv->GetMesh(),
                                          ptspec_fesv->FEColl(), ncomp);
   }
   tspec_pgf->Update();
   tspec_gf = tspec_pgf;
   tspec.SetDataAndSize(tspec_pgf->GetData(), tspec_pgf->Size());
   tspec_sav = tspec;

   adapt_eval->SetParMetaInfo(*ptspec_fesv->GetParMesh(), *ptspec_fesv);
   adapt_eval->SetInitialField(*ptspec_fesv->GetMesh()->GetNodes(), tspec);
}

void DiscreteAdaptTC::SetTspecAtIndex(int idx, const ParGridFunction &tspec_)
{
   const int vdim = tspec_.FESpace()->GetVDim(),
             ndof = tspec_.FESpace()->GetNDofs();
   MFEM_VERIFY(ndof == tspec.Size()/ncomp, "Inconsistency in SetTspecAtIndex.");

   const auto tspec__d = tspec_.Read();
   auto tspec_d = tspec.ReadWrite();
   const int offset = idx*ndof;
   internal::device_copy(tspec_d + offset, tspec__d, ndof*vdim);
   FinalizeParDiscreteTargetSpec(tspec_);
}

void DiscreteAdaptTC::SetParDiscreteTargetSize(const ParGridFunction &tspec_)
{
   MFEM_VERIFY(tspec_.FESpace()->GetOrdering() == Ordering::byNODES,
               "Discrete target size should be ordered byNodes.");
   if (sizeidx > -1) { SetTspecAtIndex(sizeidx, tspec_); return; }
   sizeidx = ncomp;
   SetDiscreteTargetBase(tspec_);
   FinalizeParDiscreteTargetSpec(tspec_);
}

void DiscreteAdaptTC::SetParDiscreteTargetSkew(const ParGridFunction &tspec_)
{
   MFEM_VERIFY(tspec_.FESpace()->GetOrdering() == Ordering::byNODES,
               "Discrete target skewness should be ordered byNodes.");
   if (skewidx > -1) { SetTspecAtIndex(skewidx, tspec_); return; }
   skewidx = ncomp;
   SetDiscreteTargetBase(tspec_);
   FinalizeParDiscreteTargetSpec(tspec_);
}

void DiscreteAdaptTC::SetParDiscreteTargetAspectRatio(const ParGridFunction &ar)
{
   MFEM_VERIFY(ar.FESpace()->GetOrdering() == Ordering::byNODES,
               "Discrete target aspect ratio should be ordered byNodes.");
   if (aspectratioidx > -1) { SetTspecAtIndex(aspectratioidx, ar); return; }
   aspectratioidx = ncomp;
   SetDiscreteTargetBase(ar);
   FinalizeParDiscreteTargetSpec(ar);
}

void DiscreteAdaptTC::SetParDiscreteTargetOrientation(const ParGridFunction &o)
{
   MFEM_VERIFY(o.FESpace()->GetOrdering() == Ordering::byNODES,
               "Discrete target orientation should be ordered byNodes.");
   if (orientationidx > -1) { SetTspecAtIndex(orientationidx, o); return; }
   orientationidx = ncomp;
   SetDiscreteTargetBase(o);
   FinalizeParDiscreteTargetSpec(o);
}

void DiscreteAdaptTC::SetParDiscreteTargetSpec(const ParGridFunction &tspec_)
{
   SetParDiscreteTargetSize(tspec_);
}
#endif // MFEM_USE_MPI

void DiscreteAdaptTC::SetDiscreteTargetBase(const GridFunction &tspec_)
{
   const int vdim = tspec_.FESpace()->GetVDim(),
             ndof = tspec_.FESpace()->GetNDofs();
   ncomp += vdim;

   // need to append data to tspec
   // make a copy of tspec->tspec_temp, increase its size, and
   // copy data from tspec_temp -> tspec, then add new entries
   Vector tspec_temp = tspec;
   tspec.UseDevice(true);
   tspec_sav.UseDevice(true);
   tspec.SetSize(ncomp*ndof);

   const auto tspec_temp_d = tspec_temp.Read();
   auto tspec_d = tspec.ReadWrite();
   internal::device_copy(tspec_d, tspec_temp_d, tspec_temp.Size());

   const auto tspec__d = tspec_.Read();
   const int offset = (ncomp-vdim)*ndof;
   internal::device_copy(tspec_d + offset, tspec__d, ndof*vdim);
}

void DiscreteAdaptTC::SetTspecAtIndex(int idx, const GridFunction &tspec_)
{
   const int vdim = tspec_.FESpace()->GetVDim(),
             ndof = tspec_.FESpace()->GetNDofs();
   MFEM_VERIFY(ndof == tspec.Size()/ncomp, "Inconsistency in SetTspecAtIndex.");

   const auto tspec__d = tspec_.Read();
   auto tspec_d = tspec.ReadWrite();
   const int offset = idx*ndof;
   internal::device_copy(tspec_d + offset, tspec__d, ndof*vdim);
   FinalizeSerialDiscreteTargetSpec(tspec_);
}

void DiscreteAdaptTC::SetSerialDiscreteTargetSize(const GridFunction &tspec_)
{
   MFEM_VERIFY(tspec_.FESpace()->GetOrdering() == Ordering::byNODES,
               "Discrete target size should be ordered byNodes.");
   if (sizeidx > -1) { SetTspecAtIndex(sizeidx, tspec_); return; }
   sizeidx = ncomp;
   SetDiscreteTargetBase(tspec_);
   FinalizeSerialDiscreteTargetSpec(tspec_);
}

void DiscreteAdaptTC::SetSerialDiscreteTargetSkew(const GridFunction &tspec_)
{
   MFEM_VERIFY(tspec_.FESpace()->GetOrdering() == Ordering::byNODES,
               "Discrete target skewness should be ordered byNodes.");
   if (skewidx > -1) { SetTspecAtIndex(skewidx, tspec_); return; }
   skewidx = ncomp;
   SetDiscreteTargetBase(tspec_);
   FinalizeSerialDiscreteTargetSpec(tspec_);
}

void DiscreteAdaptTC::SetSerialDiscreteTargetAspectRatio(const GridFunction &ar)
{
   MFEM_VERIFY(ar.FESpace()->GetOrdering() == Ordering::byNODES,
               "Discrete target aspect ratio should be ordered byNodes.");
   if (aspectratioidx > -1) { SetTspecAtIndex(aspectratioidx, ar); return; }
   aspectratioidx = ncomp;
   SetDiscreteTargetBase(ar);
   FinalizeSerialDiscreteTargetSpec(ar);
}

void DiscreteAdaptTC::SetSerialDiscreteTargetOrientation(const GridFunction &o)
{
   MFEM_VERIFY(o.FESpace()->GetOrdering() == Ordering::byNODES,
               "Discrete target orientation should be ordered byNodes.");
   if (orientationidx > -1) { SetTspecAtIndex(orientationidx, o); return; }
   orientationidx = ncomp;
   SetDiscreteTargetBase(o);
   FinalizeSerialDiscreteTargetSpec(o);
}

void DiscreteAdaptTC::FinalizeSerialDiscreteTargetSpec(const GridFunction &t)
{
   MFEM_VERIFY(adapt_eval, "SetAdaptivityEvaluator() has not been called!")
   MFEM_VERIFY(ncomp > 0, "No target specifications have been set!");

   const FiniteElementSpace *tspec_fes = t.FESpace();

   tspec_sav = tspec;

   delete tspec_fesv;
   tspec_fesv = new FiniteElementSpace(tspec_fes->GetMesh(),
                                       tspec_fes->FEColl(), ncomp,
                                       Ordering::byNODES);

   delete tspec_gf;
   tspec_gf = new GridFunction(tspec_fesv, tspec);

   adapt_eval->SetSerialMetaInfo(*tspec_fes->GetMesh(), *tspec_fesv);
   adapt_eval->SetInitialField(*tspec_fes->GetMesh()->GetNodes(), tspec);
}

void DiscreteAdaptTC::GetDiscreteTargetSpec(GridFunction &tspec_, int idx)
{
   if (idx < 0) { return; }
   const int ndof = tspec_.FESpace()->GetNDofs(),
             vdim = tspec_.FESpace()->GetVDim();
   MFEM_VERIFY(ndof == tspec.Size()/ncomp,
               "Inconsistency in GetSerialDiscreteTargetSpec.");

   for (int i = 0; i < ndof*vdim; i++)
   {
      tspec_(i) = tspec(i + idx*ndof);
   }
}

void DiscreteAdaptTC::UpdateAfterMeshTopologyChange()
{
   tspec_fesv->Update();
   tspec_gf->Update();
   tspec.SetDataAndSize(tspec_gf->GetData(), tspec_gf->Size());
   tspec_sav = tspec;

   adapt_eval->SetSerialMetaInfo(*tspec_fesv->GetMesh(), *tspec_fesv);
   adapt_eval->SetInitialField(*tspec_fesv->GetMesh()->GetNodes(), tspec);
}

void DiscreteAdaptTC::SetSerialDiscreteTargetSpec(const GridFunction &tspec_)
{
   SetSerialDiscreteTargetSize(tspec_);
}


void DiscreteAdaptTC::UpdateTargetSpecification(const Vector &new_x,
                                                bool reuse_flag,
                                                int new_x_ordering)
{
   if (reuse_flag && good_tspec) { return; }

   MFEM_VERIFY(tspec.Size() > 0, "Target specification is not set!");
   adapt_eval->ComputeAtNewPosition(new_x, tspec, new_x_ordering);
   tspec_sav = tspec;

   good_tspec = reuse_flag;
}

void DiscreteAdaptTC::UpdateTargetSpecification(Vector &new_x,
                                                Vector &IntData,
                                                int new_x_ordering)
{
   adapt_eval->ComputeAtNewPosition(new_x, IntData, new_x_ordering);
}

void DiscreteAdaptTC::UpdateTargetSpecificationAtNode(const FiniteElement &el,
                                                      ElementTransformation &T,
                                                      int dofidx, int dir,
                                                      const Vector &IntData)
{
   MFEM_VERIFY(tspec.Size() > 0, "Target specification is not set!");

   Array<int> dofs;
   tspec_fesv->GetElementDofs(T.ElementNo, dofs);
   const int cnt = tspec.Size()/ncomp; // dofs per scalar-field

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
   tspec_fesv->GetElementDofs(T.ElementNo, dofs);
   const int cnt = tspec.Size()/ncomp;
   for (int i = 0; i < ncomp; i++)
   {
      tspec(dofs[dofidx] + i*cnt) = tspec_sav(dofs[dofidx] + i*cnt);
   }
}

void DiscreteAdaptTC::SetTspecFromIntRule(int e_id,
                                          const IntegrationRule &intrule)
{
   switch (target_type)
   {
      case IDEAL_SHAPE_GIVEN_SIZE:
      case GIVEN_SHAPE_AND_SIZE:
      {
         const int ndofs = tspec_fesv->GetFE(e_id)->GetDof(),
                   ntspec_dofs = ndofs*ncomp;

         Vector tspec_vals(ntspec_dofs);

         Array<int> dofs;
         tspec_fesv->GetElementVDofs(e_id, dofs);
         tspec.GetSubVector(dofs, tspec_vals);
         DenseMatrix tr;
         tspec_gf->GetVectorValues(e_id, intrule, tspec_refine, tr);
         tspec_refine.Transpose();
         break;
      }
      default:
         MFEM_ABORT("Incompatible target type for discrete adaptation!");
   }
}

void DiscreteAdaptTC::SetTspecDataForDerefinement(FiniteElementSpace *fes)
{
   coarse_tspec_fesv = fes;
   const Operator *c_op = fes->GetUpdateOperator();
   tspec_derefine.SetSize(c_op->Height());
   c_op->Mult(tspec, tspec_derefine);
}

void DiscreteAdaptTC::ComputeElementTargets(int e_id, const FiniteElement &fe,
                                            const IntegrationRule &ir,
                                            const Vector &elfun,
                                            DenseTensor &Jtr) const
{
   MFEM_VERIFY(tspec_fesv, "No target specifications have been set.");
   const int dim = fe.GetDim(),
             nqp = ir.GetNPoints();
   Jtrcomp.SetSize(dim, dim, 4*nqp);

   FiniteElementSpace *src_fes = tspec_fesv;

   switch (target_type)
   {
      case IDEAL_SHAPE_GIVEN_SIZE:
      case GIVEN_SHAPE_AND_SIZE:
      {
         const DenseMatrix &Wideal =
            Geometries.GetGeomToPerfGeomJac(fe.GetGeomType());
         const int ndofs = tspec_fesv->GetFE(e_id)->GetDof(),
                   ntspec_dofs = ndofs*ncomp;

         Vector shape(ndofs), tspec_vals(ntspec_dofs), par_vals,
                par_vals_c1, par_vals_c2, par_vals_c3;

         Array<int> dofs;
         DenseMatrix D_rho(dim), Q_phi(dim), R_theta(dim);
         tspec_fesv->GetElementVDofs(e_id, dofs);
         tspec.UseDevice(true);
         tspec.GetSubVector(dofs, tspec_vals);
         if (tspec_refine.NumCols() > 0) // Refinement
         {
            MFEM_VERIFY(amr_el >= 0, " Target being constructed for an AMR element.");
            for (int i = 0; i < ncomp; i++)
            {
               for (int j = 0; j < ndofs; j++)
               {
                  tspec_vals(j + i*ndofs) = tspec_refine(j + amr_el*ndofs, i);
               }
            }
         }
         else if (tspec_derefine.Size() > 0) // Derefinement
         {
            dofs.SetSize(0);
            coarse_tspec_fesv->GetElementVDofs(e_id, dofs);
            tspec_derefine.GetSubVector(dofs, tspec_vals);
            src_fes = coarse_tspec_fesv;
         }

         for (int q = 0; q < nqp; q++)
         {
            const IntegrationPoint &ip = ir.IntPoint(q);
            src_fes->GetFE(e_id)->CalcShape(ip, shape);
            Jtr(q) = Wideal; // Initialize to identity
            for (int d = 0; d < 4; d++)
            {
               DenseMatrix Jtrcomp_q(Jtrcomp.GetData(d + 4*q), dim, dim);
               Jtrcomp_q = Wideal; // Initialize to identity
            }

            if (sizeidx != -1) // Set size
            {
               par_vals.SetDataAndSize(tspec_vals.GetData()+sizeidx*ndofs, ndofs);
               real_t min_size = par_vals.Min();
               if (lim_min_size > 0.) { min_size = lim_min_size; }
               MFEM_VERIFY(min_size > 0.0,
                           "Non-positive size propagated in the target definition.");

               real_t size = std::max(shape * par_vals, min_size);
               NCMesh *ncmesh = tspec_fesv->GetMesh()->ncmesh;
               if (ncmesh)
               {
                  size /= ncmesh->GetElementSizeReduction(e_id);
               }
               Jtr(q).Set(std::pow(size, 1.0/dim), Jtr(q));
               DenseMatrix Jtrcomp_q(Jtrcomp.GetData(0 + 4*q), dim, dim);
               Jtrcomp_q = Jtr(q);
            } // Done size

            if (target_type == IDEAL_SHAPE_GIVEN_SIZE) { continue; }

            if (aspectratioidx != -1) // Set aspect ratio
            {
               if (dim == 2)
               {
                  par_vals.SetDataAndSize(tspec_vals.GetData()+
                                          aspectratioidx*ndofs, ndofs);
                  const real_t min_size = par_vals.Min();
                  MFEM_VERIFY(min_size > 0.0,
                              "Non-positive aspect-ratio propagated in the target definition.");

                  const real_t aspectratio = shape * par_vals;
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

                  const real_t rho1 = shape * par_vals_c1;
                  const real_t rho2 = shape * par_vals_c2;
                  const real_t rho3 = shape * par_vals_c3;
                  D_rho = 0.;
                  D_rho(0,0) = pow(rho1,2./3.);
                  D_rho(1,1) = pow(rho2,2./3.);
                  D_rho(2,2) = pow(rho3,2./3.);
               }
               DenseMatrix Jtrcomp_q(Jtrcomp.GetData(1 + 4*q), dim, dim);
               Jtrcomp_q = D_rho;
               DenseMatrix Temp = Jtr(q);
               Mult(D_rho, Temp, Jtr(q));
            } // Done aspect ratio

            if (skewidx != -1) // Set skew
            {
               if (dim == 2)
               {
                  par_vals.SetDataAndSize(tspec_vals.GetData()+
                                          skewidx*ndofs, ndofs);

                  const real_t skew = shape * par_vals;

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

                  const real_t phi12  = shape * par_vals_c1;
                  const real_t phi13  = shape * par_vals_c2;
                  const real_t chi = shape * par_vals_c3;

                  Q_phi = 0.;
                  Q_phi(0,0) = 1.;
                  Q_phi(0,1) = cos(phi12);
                  Q_phi(0,2) = cos(phi13);

                  Q_phi(1,1) = sin(phi12);
                  Q_phi(1,2) = sin(phi13)*cos(chi);

                  Q_phi(2,2) = sin(phi13)*sin(chi);
               }
               DenseMatrix Jtrcomp_q(Jtrcomp.GetData(2 + 4*q), dim, dim);
               Jtrcomp_q = Q_phi;
               DenseMatrix Temp = Jtr(q);
               Mult(Q_phi, Temp, Jtr(q));
            } // Done skew

            if (orientationidx != -1) // Set orientation
            {
               if (dim == 2)
               {
                  par_vals.SetDataAndSize(tspec_vals.GetData()+
                                          orientationidx*ndofs, ndofs);

                  const real_t theta = shape * par_vals;
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

                  const real_t theta = shape * par_vals_c1;
                  const real_t psi   = shape * par_vals_c2;
                  const real_t beta  = shape * par_vals_c3;

                  real_t ct = cos(theta), st = sin(theta),
                         cp = cos(psi),   sp = sin(psi),
                         cb = cos(beta),  sb = sin(beta);

                  R_theta = 0.;
                  R_theta(0,0) = ct*sp;
                  R_theta(1,0) = st*sp;
                  R_theta(2,0) = cp;

                  R_theta(0,1) = -st*cb + ct*cp*sb;
                  R_theta(1,1) = ct*cb + st*cp*sb;
                  R_theta(2,1) = -sp*sb;

                  R_theta(0,0) = -st*sb - ct*cp*cb;
                  R_theta(1,0) = ct*sb - st*cp*cb;
                  R_theta(2,0) = sp*cb;
               }
               DenseMatrix Jtrcomp_q(Jtrcomp.GetData(3 + 4*q), dim, dim);
               Jtrcomp_q = R_theta;
               DenseMatrix Temp = Jtr(q);
               Mult(R_theta, Temp, Jtr(q));
            } // Done orientation
         }
         break;
      }
      default:
         MFEM_ABORT("Incompatible target type for discrete adaptation!");
   }
}

void DiscreteAdaptTC::ComputeElementTargetsGradient(const IntegrationRule &ir,
                                                    const Vector &elfun,
                                                    IsoparametricTransformation &Tpr,
                                                    DenseTensor &dJtr) const
{
   MFEM_ASSERT(target_type == IDEAL_SHAPE_UNIT_SIZE || nodes != NULL, "");

   MFEM_VERIFY(tspec_fesv, "No target specifications have been set.");

   dJtr = 0.;
   const int e_id = Tpr.ElementNo;
   const FiniteElement *fe = Tpr.GetFE();

   switch (target_type)
   {
      case IDEAL_SHAPE_GIVEN_SIZE:
      case GIVEN_SHAPE_AND_SIZE:
      {
         const DenseMatrix &Wideal =
            Geometries.GetGeomToPerfGeomJac(fe->GetGeomType());
         const int dim = Wideal.Height(),
                   ndofs = fe->GetDof(),
                   ntspec_dofs = ndofs*ncomp;

         Vector shape(ndofs), tspec_vals(ntspec_dofs), par_vals,
                par_vals_c1(ndofs), par_vals_c2(ndofs), par_vals_c3(ndofs);

         Array<int> dofs;
         DenseMatrix dD_rho(dim), dQ_phi(dim), dR_theta(dim);
         DenseMatrix dQ_phi13(dim), dQ_phichi(dim); // dQ_phi is used for dQ/dphi12 in 3D
         DenseMatrix dR_psi(dim), dR_beta(dim);
         tspec_fesv->GetElementVDofs(e_id, dofs);
         tspec.GetSubVector(dofs, tspec_vals);

         DenseMatrix grad_e_c1(ndofs, dim),
                     grad_e_c2(ndofs, dim),
                     grad_e_c3(ndofs, dim);
         Vector grad_ptr_c1(grad_e_c1.GetData(), ndofs*dim),
                grad_ptr_c2(grad_e_c2.GetData(), ndofs*dim),
                grad_ptr_c3(grad_e_c3.GetData(), ndofs*dim);

         DenseMatrix grad_phys; // This will be (dof x dim, dof).
         fe->ProjectGrad(*fe, Tpr, grad_phys);

         for (int i = 0; i < ir.GetNPoints(); i++)
         {
            const IntegrationPoint &ip = ir.IntPoint(i);
            DenseMatrix Jtrcomp_s(Jtrcomp.GetData(0 + 4*i), dim, dim); // size
            DenseMatrix Jtrcomp_d(Jtrcomp.GetData(1 + 4*i), dim, dim); // aspect-ratio
            DenseMatrix Jtrcomp_q(Jtrcomp.GetData(2 + 4*i), dim, dim); // skew
            DenseMatrix Jtrcomp_r(Jtrcomp.GetData(3 + 4*i), dim, dim); // orientation
            DenseMatrix work1(dim), work2(dim), work3(dim);

            if (sizeidx != -1) // Set size
            {
               par_vals.SetDataAndSize(tspec_vals.GetData()+sizeidx*ndofs, ndofs);

               grad_phys.Mult(par_vals, grad_ptr_c1);
               Vector grad_q(dim);
               tspec_fesv->GetFE(e_id)->CalcShape(ip, shape);
               grad_e_c1.MultTranspose(shape, grad_q);

               const real_t min_size = par_vals.Min();
               MFEM_VERIFY(min_size > 0.0,
                           "Non-positive size propagated in the target definition.");
               const real_t size = std::max(shape * par_vals, min_size);
               real_t dz_dsize = (1./dim)*pow(size, 1./dim - 1.);

               Mult(Jtrcomp_q, Jtrcomp_d, work1); // Q*D
               Mult(Jtrcomp_r, work1, work2);     // R*Q*D

               for (int d = 0; d < dim; d++)
               {
                  DenseMatrix &dJtr_i = dJtr(i + d*ir.GetNPoints());
                  work1 = Wideal;
                  work1.Set(dz_dsize, work1);    // dz/dsize
                  work1 *= grad_q(d);            // dz/dsize*dsize/dx
                  AddMult(work1, work2, dJtr_i); // dz/dx*R*Q*D
               }
            } // Done size

            if (target_type == IDEAL_SHAPE_GIVEN_SIZE) { continue; }

            if (aspectratioidx != -1) // Set aspect ratio
            {
               if (dim == 2)
               {
                  par_vals.SetDataAndSize(tspec_vals.GetData()+
                                          aspectratioidx*ndofs, ndofs);

                  grad_phys.Mult(par_vals, grad_ptr_c1);
                  Vector grad_q(dim);
                  tspec_fesv->GetFE(e_id)->CalcShape(ip, shape);
                  grad_e_c1.MultTranspose(shape, grad_q);

                  const real_t aspectratio = shape * par_vals;
                  dD_rho = 0.;
                  dD_rho(0,0) = -0.5*pow(aspectratio,-1.5);
                  dD_rho(1,1) = 0.5*pow(aspectratio,-0.5);

                  Mult(Jtrcomp_s, Jtrcomp_r, work1); // z*R
                  Mult(work1, Jtrcomp_q, work2);     // z*R*Q

                  for (int d = 0; d < dim; d++)
                  {
                     DenseMatrix &dJtr_i = dJtr(i + d*ir.GetNPoints());
                     work1 = dD_rho;
                     work1 *= grad_q(d); // work1 = dD/drho*drho/dx
                     AddMult(work2, work1, dJtr_i); // z*R*Q*dD/dx
                  }
               }
               else // 3D
               {
                  par_vals.SetDataAndSize(tspec_vals.GetData()+
                                          aspectratioidx*ndofs, ndofs*3);
                  par_vals_c1.SetData(par_vals.GetData());
                  par_vals_c2.SetData(par_vals.GetData()+ndofs);
                  par_vals_c3.SetData(par_vals.GetData()+2*ndofs);

                  grad_phys.Mult(par_vals_c1, grad_ptr_c1);
                  grad_phys.Mult(par_vals_c2, grad_ptr_c2);
                  grad_phys.Mult(par_vals_c3, grad_ptr_c3);
                  Vector grad_q1(dim), grad_q2(dim), grad_q3(dim);
                  tspec_fesv->GetFE(e_id)->CalcShape(ip, shape);
                  grad_e_c1.MultTranspose(shape, grad_q1);
                  grad_e_c2.MultTranspose(shape, grad_q2);
                  grad_e_c3.MultTranspose(shape, grad_q3);

                  const real_t rho1 = shape * par_vals_c1;
                  const real_t rho2 = shape * par_vals_c2;
                  const real_t rho3 = shape * par_vals_c3;
                  dD_rho = 0.;
                  dD_rho(0,0) = (2./3.)*pow(rho1,-1./3.);
                  dD_rho(1,1) = (2./3.)*pow(rho2,-1./3.);
                  dD_rho(2,2) = (2./3.)*pow(rho3,-1./3.);

                  Mult(Jtrcomp_s, Jtrcomp_r, work1); // z*R
                  Mult(work1, Jtrcomp_q, work2);     // z*R*Q


                  for (int d = 0; d < dim; d++)
                  {
                     DenseMatrix &dJtr_i = dJtr(i + d*ir.GetNPoints());
                     work1 = dD_rho;
                     work1(0,0) *= grad_q1(d);
                     work1(1,2) *= grad_q2(d);
                     work1(2,2) *= grad_q3(d);
                     // work1 = dD/dx = dD/drho1*drho1/dx + dD/drho2*drho2/dx
                     AddMult(work2, work1, dJtr_i); // z*R*Q*dD/dx
                  }
               }
            } // Done aspect ratio

            if (skewidx != -1) // Set skew
            {
               if (dim == 2)
               {
                  par_vals.SetDataAndSize(tspec_vals.GetData()+
                                          skewidx*ndofs, ndofs);

                  grad_phys.Mult(par_vals, grad_ptr_c1);
                  Vector grad_q(dim);
                  tspec_fesv->GetFE(e_id)->CalcShape(ip, shape);
                  grad_e_c1.MultTranspose(shape, grad_q);

                  const real_t skew = shape * par_vals;

                  dQ_phi = 0.;
                  dQ_phi(0,0) = 1.;
                  dQ_phi(0,1) = -sin(skew);
                  dQ_phi(1,1) = cos(skew);

                  Mult(Jtrcomp_s, Jtrcomp_r, work2); // z*R

                  for (int d = 0; d < dim; d++)
                  {
                     DenseMatrix &dJtr_i = dJtr(i + d*ir.GetNPoints());
                     work1 = dQ_phi;
                     work1 *= grad_q(d); // work1 = dQ/dphi*dphi/dx
                     Mult(work1, Jtrcomp_d, work3); // dQ/dx*D
                     AddMult(work2, work3, dJtr_i); // z*R*dQ/dx*D
                  }
               }
               else
               {
                  par_vals.SetDataAndSize(tspec_vals.GetData()+
                                          skewidx*ndofs, ndofs*3);
                  par_vals_c1.SetData(par_vals.GetData());
                  par_vals_c2.SetData(par_vals.GetData()+ndofs);
                  par_vals_c3.SetData(par_vals.GetData()+2*ndofs);

                  grad_phys.Mult(par_vals_c1, grad_ptr_c1);
                  grad_phys.Mult(par_vals_c2, grad_ptr_c2);
                  grad_phys.Mult(par_vals_c3, grad_ptr_c3);
                  Vector grad_q1(dim), grad_q2(dim), grad_q3(dim);
                  tspec_fesv->GetFE(e_id)->CalcShape(ip, shape);
                  grad_e_c1.MultTranspose(shape, grad_q1);
                  grad_e_c2.MultTranspose(shape, grad_q2);
                  grad_e_c3.MultTranspose(shape, grad_q3);

                  const real_t phi12  = shape * par_vals_c1;
                  const real_t phi13  = shape * par_vals_c2;
                  const real_t chi = shape * par_vals_c3;

                  dQ_phi = 0.;
                  dQ_phi(0,0) = 1.;
                  dQ_phi(0,1) = -sin(phi12);
                  dQ_phi(1,1) = cos(phi12);

                  dQ_phi13 = 0.;
                  dQ_phi13(0,2) = -sin(phi13);
                  dQ_phi13(1,2) = cos(phi13)*cos(chi);
                  dQ_phi13(2,2) = cos(phi13)*sin(chi);

                  dQ_phichi = 0.;
                  dQ_phichi(1,2) = -sin(phi13)*sin(chi);
                  dQ_phichi(2,2) =  sin(phi13)*cos(chi);

                  Mult(Jtrcomp_s, Jtrcomp_r, work2); // z*R

                  for (int d = 0; d < dim; d++)
                  {
                     DenseMatrix &dJtr_i = dJtr(i + d*ir.GetNPoints());
                     work1 = dQ_phi;
                     work1 *= grad_q1(d); // work1 = dQ/dphi12*dphi12/dx
                     work1.Add(grad_q2(d), dQ_phi13);  // + dQ/dphi13*dphi13/dx
                     work1.Add(grad_q3(d), dQ_phichi); // + dQ/dchi*dchi/dx
                     Mult(work1, Jtrcomp_d, work3); // dQ/dx*D
                     AddMult(work2, work3, dJtr_i); // z*R*dQ/dx*D
                  }
               }
            } // Done skew

            if (orientationidx != -1) // Set orientation
            {
               if (dim == 2)
               {
                  par_vals.SetDataAndSize(tspec_vals.GetData()+
                                          orientationidx*ndofs, ndofs);

                  grad_phys.Mult(par_vals, grad_ptr_c1);
                  Vector grad_q(dim);
                  tspec_fesv->GetFE(e_id)->CalcShape(ip, shape);
                  grad_e_c1.MultTranspose(shape, grad_q);

                  const real_t theta = shape * par_vals;
                  dR_theta(0,0) = -sin(theta);
                  dR_theta(0,1) = -cos(theta);
                  dR_theta(1,0) =  cos(theta);
                  dR_theta(1,1) = -sin(theta);

                  Mult(Jtrcomp_q, Jtrcomp_d, work1); // Q*D
                  Mult(Jtrcomp_s, work1, work2);     // z*Q*D
                  for (int d = 0; d < dim; d++)
                  {
                     DenseMatrix &dJtr_i = dJtr(i + d*ir.GetNPoints());
                     work1 = dR_theta;
                     work1 *= grad_q(d); // work1 = dR/dtheta*dtheta/dx
                     AddMult(work1, work2, dJtr_i);  // z*dR/dx*Q*D
                  }
               }
               else
               {
                  par_vals.SetDataAndSize(tspec_vals.GetData()+
                                          orientationidx*ndofs, ndofs*3);
                  par_vals_c1.SetData(par_vals.GetData());
                  par_vals_c2.SetData(par_vals.GetData()+ndofs);
                  par_vals_c3.SetData(par_vals.GetData()+2*ndofs);

                  grad_phys.Mult(par_vals_c1, grad_ptr_c1);
                  grad_phys.Mult(par_vals_c2, grad_ptr_c2);
                  grad_phys.Mult(par_vals_c3, grad_ptr_c3);
                  Vector grad_q1(dim), grad_q2(dim), grad_q3(dim);
                  tspec_fesv->GetFE(e_id)->CalcShape(ip, shape);
                  grad_e_c1.MultTranspose(shape, grad_q1);
                  grad_e_c2.MultTranspose(shape, grad_q2);
                  grad_e_c3.MultTranspose(shape, grad_q3);

                  const real_t theta = shape * par_vals_c1;
                  const real_t psi   = shape * par_vals_c2;
                  const real_t beta  = shape * par_vals_c3;

                  const real_t ct = cos(theta), st = sin(theta),
                               cp = cos(psi),   sp = sin(psi),
                               cb = cos(beta),  sb = sin(beta);

                  dR_theta = 0.;
                  dR_theta(0,0) = -st*sp;
                  dR_theta(1,0) = ct*sp;
                  dR_theta(2,0) = 0;

                  dR_theta(0,1) = -ct*cb - st*cp*sb;
                  dR_theta(1,1) = -st*cb + ct*cp*sb;
                  dR_theta(2,1) = 0.;

                  dR_theta(0,0) = -ct*sb + st*cp*cb;
                  dR_theta(1,0) = -st*sb - ct*cp*cb;
                  dR_theta(2,0) = 0.;

                  dR_beta = 0.;
                  dR_beta(0,0) = 0.;
                  dR_beta(1,0) = 0.;
                  dR_beta(2,0) = 0.;

                  dR_beta(0,1) = st*sb + ct*cp*cb;
                  dR_beta(1,1) = -ct*sb + st*cp*cb;
                  dR_beta(2,1) = -sp*cb;

                  dR_beta(0,0) = -st*cb + ct*cp*sb;
                  dR_beta(1,0) = ct*cb + st*cp*sb;
                  dR_beta(2,0) = 0.;

                  dR_psi = 0.;
                  dR_psi(0,0) = ct*cp;
                  dR_psi(1,0) = st*cp;
                  dR_psi(2,0) = -sp;

                  dR_psi(0,1) = 0. - ct*sp*sb;
                  dR_psi(1,1) = 0. + st*sp*sb;
                  dR_psi(2,1) = -cp*sb;

                  dR_psi(0,0) = 0. + ct*sp*cb;
                  dR_psi(1,0) = 0. + st*sp*cb;
                  dR_psi(2,0) = cp*cb;

                  Mult(Jtrcomp_q, Jtrcomp_d, work1); // Q*D
                  Mult(Jtrcomp_s, work1, work2);     // z*Q*D
                  for (int d = 0; d < dim; d++)
                  {
                     DenseMatrix &dJtr_i = dJtr(i + d*ir.GetNPoints());
                     work1 = dR_theta;
                     work1 *= grad_q1(d); // work1 = dR/dtheta*dtheta/dx
                     work1.Add(grad_q2(d), dR_psi);  // +dR/dpsi*dpsi/dx
                     work1.Add(grad_q3(d), dR_beta); // +dR/dbeta*dbeta/dx
                     AddMult(work1, work2, dJtr_i);  // z*dR/dx*Q*D
                  }
               }
            } // Done orientation
         }
         break;
      }
      default:
         MFEM_ABORT("Incompatible target type for discrete adaptation!");
   }
   Jtrcomp.Clear();
}

void DiscreteAdaptTC::
UpdateGradientTargetSpecification(const Vector &x, real_t dx,
                                  bool reuse_flag, int x_ordering)
{
   if (reuse_flag && good_tspec_grad) { return; }

   const int dim = tspec_fesv->GetTypicalFE()->GetDim(),
             cnt = x.Size()/dim;

   MFEM_VERIFY(tspec_fesv->GetVSize() / ncomp == cnt,
               "FD with discrete adaptivity assume mesh_order = field_order.");

   tspec_pert1h.SetSize(x.Size()*ncomp);

   Vector TSpecTemp;
   Vector xtemp = x;
   for (int j = 0; j < dim; j++)
   {
      for (int i = 0; i < cnt; i++)
      {
         int idx = x_ordering == Ordering::byNODES ? j*cnt + i : i*dim + j;
         xtemp(idx) += dx;
      }

      TSpecTemp.NewDataAndSize(tspec_pert1h.GetData() + j*cnt*ncomp, cnt*ncomp);
      UpdateTargetSpecification(xtemp, TSpecTemp, x_ordering);

      for (int i = 0; i < cnt; i++)
      {
         int idx = x_ordering == Ordering::byNODES ? j*cnt + i : i*dim + j;
         xtemp(idx) -= dx;
      }
   }

   good_tspec_grad = reuse_flag;
}

void DiscreteAdaptTC::
UpdateHessianTargetSpecification(const Vector &x, real_t dx,
                                 bool reuse_flag, int x_ordering)
{
   if (reuse_flag && good_tspec_hess) { return; }

   const int dim    = tspec_fesv->GetTypicalFE()->GetDim(),
             cnt    = x.Size()/dim,
             totmix = 1+2*(dim-2);

   MFEM_VERIFY(tspec_fesv->GetVSize() / ncomp == cnt,
               "FD with discrete adaptivity assume mesh_order = field_order.");

   tspec_pert2h.SetSize(cnt*dim*ncomp);
   tspec_pertmix.SetSize(cnt*totmix*ncomp);

   Vector TSpecTemp;
   Vector xtemp = x;

   // T(x+2h)
   for (int j = 0; j < dim; j++)
   {
      for (int i = 0; i < cnt; i++)
      {
         int idx = x_ordering == Ordering::byNODES ? j*cnt + i : i*dim + j;
         xtemp(idx) += 2*dx;
      }

      TSpecTemp.NewDataAndSize(tspec_pert2h.GetData() + j*cnt*ncomp, cnt*ncomp);
      UpdateTargetSpecification(xtemp, TSpecTemp, x_ordering);

      for (int i = 0; i < cnt; i++)
      {
         int idx = x_ordering == Ordering::byNODES ? j*cnt + i : i*dim + j;
         xtemp(idx) -= 2*dx;
      }
   }

   // T(x+h,y+h)
   int j = 0;
   for (int k1 = 0; k1 < dim; k1++)
   {
      for (int k2 = 0; (k1 != k2) && (k2 < dim); k2++)
      {
         for (int i = 0; i < cnt; i++)
         {
            int idx1 = x_ordering == Ordering::byNODES ? k1*cnt+i : i*dim + k1;
            int idx2 = x_ordering == Ordering::byNODES ? k2*cnt+i : i*dim + k2;
            xtemp(idx1) += dx;
            xtemp(idx2) += dx;
         }

         TSpecTemp.NewDataAndSize(tspec_pertmix.GetData() + j*cnt*ncomp, cnt*ncomp);
         UpdateTargetSpecification(xtemp, TSpecTemp, x_ordering);

         for (int i = 0; i < cnt; i++)
         {
            int idx1 = x_ordering == Ordering::byNODES ? k1*cnt+i : i*dim + k1;
            int idx2 = x_ordering == Ordering::byNODES ? k2*cnt+i : i*dim + k2;
            xtemp(idx1) -= dx;
            xtemp(idx2) -= dx;
         }
         j++;
      }
   }

   good_tspec_hess = reuse_flag;
}

DiscreteAdaptTC::~DiscreteAdaptTC()
{
   delete tspec_gf;
   delete adapt_eval;
   delete tspec_fesv;
#ifdef MFEM_USE_MPI
   delete ptspec_fesv;
#endif
}

void AdaptivityEvaluator::SetSerialMetaInfo(const Mesh &m,
                                            const FiniteElementSpace &f)
{
   delete fes;
   delete mesh;
   mesh = new Mesh(m, true);
   fes = new FiniteElementSpace(mesh, f.FEColl(),
                                f.GetVDim(), f.GetOrdering());
}

#ifdef MFEM_USE_MPI
void AdaptivityEvaluator::SetParMetaInfo(const ParMesh &m,
                                         const ParFiniteElementSpace &f)
{
   delete pfes;
   delete pmesh;
   pmesh = new ParMesh(m, true);
   pfes  = new ParFiniteElementSpace(pmesh, f.FEColl(),
                                     f.GetVDim(), f.GetOrdering());
}
#endif

void AdaptivityEvaluator::ClearGeometricFactors()
{
#ifdef MFEM_USE_MPI
   if (pmesh) { pmesh->DeleteGeometricFactors(); }
#else
   if (mesh) { mesh->DeleteGeometricFactors(); }
#endif
}

AdaptivityEvaluator::~AdaptivityEvaluator()
{
   delete fes;
   delete mesh;
#ifdef MFEM_USE_MPI
   delete pfes;
   delete pmesh;
#endif
}

void TMOP_Integrator::ReleasePADeviceMemory(bool copy_to_host)
{
   if (PA.enabled)
   {
      PA.H.GetMemory().DeleteDevice(copy_to_host);
      PA.H0.GetMemory().DeleteDevice(copy_to_host);
      if (!copy_to_host && !PA.Jtr.GetMemory().HostIsValid())
      {
         PA.Jtr_needs_update = true;
      }
      PA.Jtr.GetMemory().DeleteDevice(copy_to_host);
   }
}

void TMOP_Integrator::SetInitialMeshPos(const GridFunction *x0)
{
   x_0 = x0;

   periodic = (x_0 && x_0->FESpace()->IsDGSpace()) ? true : false;

   // Compute PA.X0 when we're setting x_0 to something.
   if (PA.enabled && x_0 != nullptr)
   {
      const ElementDofOrdering ord = ElementDofOrdering::LEXICOGRAPHIC;
      const Operator *n0_R = x0->FESpace()->GetElementRestriction(ord);
      PA.X0.UseDevice(true);
      PA.X0.SetSize(n0_R->Height(), Device::GetMemoryType());
      n0_R->Mult(*x_0, PA.X0);
   }
}

TMOP_Integrator::~TMOP_Integrator()
{
   delete lim_func;
   delete adapt_lim_gf;
   delete surf_fit_gf;
   delete surf_fit_limiter;
   delete surf_fit_grad;
   delete surf_fit_hess;
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
   lim_nodes0 = &n0;
   lim_coeff = &w0;
   lim_dist = &dist;
   MFEM_VERIFY(lim_dist->FESpace()->GetVDim() == 1,
               "'dist' must be a scalar GridFunction!");

   delete lim_func;
   lim_func = (lfunc) ? lfunc : new TMOP_QuadraticLimiter;
}

void TMOP_Integrator::EnableLimiting(const GridFunction &n0, Coefficient &w0,
                                     TMOP_LimiterFunction *lfunc)
{
   lim_nodes0 = &n0;
   lim_coeff = &w0;
   lim_dist = NULL;

   delete lim_func;
   lim_func = (lfunc) ? lfunc : new TMOP_QuadraticLimiter;
}

void TMOP_Integrator::EnableAdaptiveLimiting(const GridFunction &z0,
                                             Coefficient &coeff,
                                             AdaptivityEvaluator &ae)
{
   adapt_lim_gf0 = &z0;
   delete adapt_lim_gf;
   adapt_lim_gf   = new GridFunction(z0);
   adapt_lim_coeff = &coeff;
   adapt_lim_eval = &ae;

   adapt_lim_eval->SetSerialMetaInfo(*z0.FESpace()->GetMesh(),
                                     *z0.FESpace());
   adapt_lim_eval->SetInitialField
   (*adapt_lim_gf->FESpace()->GetMesh()->GetNodes(), *adapt_lim_gf);
}

#ifdef MFEM_USE_MPI
void TMOP_Integrator::EnableAdaptiveLimiting(const ParGridFunction &z0,
                                             Coefficient &coeff,
                                             AdaptivityEvaluator &ae)
{
   adapt_lim_gf0 = &z0;
   adapt_lim_pgf0 = &z0;
   delete adapt_lim_gf;
   adapt_lim_gf   = new GridFunction(z0);
   adapt_lim_coeff = &coeff;
   adapt_lim_eval = &ae;

   adapt_lim_eval->SetParMetaInfo(*z0.ParFESpace()->GetParMesh(),
                                  *z0.ParFESpace());
   adapt_lim_eval->SetInitialField
   (*adapt_lim_gf->FESpace()->GetMesh()->GetNodes(), *adapt_lim_gf);
}
#endif

void TMOP_Integrator::EnableSurfaceFitting(const GridFunction &s0,
                                           const Array<bool> &smarker,
                                           Coefficient &coeff,
                                           AdaptivityEvaluator &ae)
{
   // To have both we must duplicate the markers.
   MFEM_VERIFY(surf_fit_pos == NULL,
               "Using both fitting approaches is not supported.");

   const bool per = s0.FESpace()->IsDGSpace();
   MFEM_VERIFY(per == false, "Fitting is not supported for periodic meshes.");

   const int dim = s0.FESpace()->GetMesh()->Dimension();
   Mesh *mesh = s0.FESpace()->GetMesh();
   MFEM_VERIFY(mesh->GetNodes()->Size() == dim*s0.Size(),
               "Mesh and level-set polynomial order must be the same.");
   const H1_FECollection *fec = dynamic_cast<const H1_FECollection *>
                                (s0.FESpace()->FEColl());
   MFEM_VERIFY(fec, "Only H1_FECollection is supported for the surface fitting "
               "grid function.");

   delete surf_fit_gf;
   surf_fit_gf = new GridFunction(s0);
   surf_fit_gf->CountElementsPerVDof(surf_fit_dof_count);
   surf_fit_marker = &smarker;
   surf_fit_coeff = &coeff;
   surf_fit_eval = &ae;

   surf_fit_eval->SetSerialMetaInfo(*s0.FESpace()->GetMesh(),
                                    *s0.FESpace());
   surf_fit_eval->SetInitialField
   (*surf_fit_gf->FESpace()->GetMesh()->GetNodes(), *surf_fit_gf);
}

void TMOP_Integrator::EnableSurfaceFitting(const GridFunction &pos,
                                           const Array<bool> &smarker,
                                           Coefficient &coeff)
{
   // To have both we must duplicate the markers.
   MFEM_VERIFY(surf_fit_gf == NULL,
               "Using both fitting approaches is not supported.");
   MFEM_VERIFY(pos.FESpace()->GetMesh()->GetNodes(),
               "Positions on a mesh without Nodes is not supported.");
   MFEM_VERIFY(pos.FESpace()->GetOrdering() ==
               pos.FESpace()->GetMesh()->GetNodes()->FESpace()->GetOrdering(),
               "Incompatible ordering of spaces!");

   const bool per = pos.FESpace()->IsDGSpace();
   MFEM_VERIFY(per == false, "Fitting is not supported for periodic meshes.");

   surf_fit_pos     = &pos;
   pos.CountElementsPerVDof(surf_fit_dof_count);
   surf_fit_marker  = &smarker;
   surf_fit_coeff   = &coeff;
   delete surf_fit_limiter;
   surf_fit_limiter = new TMOP_QuadraticLimiter;
}

#ifdef MFEM_USE_MPI
void TMOP_Integrator::EnableSurfaceFitting(const ParGridFunction &s0,
                                           const Array<bool> &smarker,
                                           Coefficient &coeff,
                                           AdaptivityEvaluator &ae,
                                           AdaptivityEvaluator *aegrad,
                                           AdaptivityEvaluator *aehess)
{
   // To have both we must duplicate the markers.
   MFEM_VERIFY(surf_fit_pos == NULL,
               "Using both fitting approaches is not supported.");

   const bool per = s0.FESpace()->IsDGSpace();
   MFEM_VERIFY(per == false, "Fitting is not supported for periodic meshes.");

   const int dim = s0.FESpace()->GetMesh()->Dimension();
   ParMesh *pmesh = s0.ParFESpace()->GetParMesh();
   MFEM_VERIFY(pmesh->GetNodes()->Size() == dim*s0.Size(),
               "Mesh and level-set polynomial order must be the same.");
   const H1_FECollection *fec = dynamic_cast<const H1_FECollection *>
                                (s0.FESpace()->FEColl());
   MFEM_VERIFY(fec, "Only H1_FECollection is supported for the surface fitting "
               "grid function.");


   delete surf_fit_gf;
   surf_fit_gf = new GridFunction(s0);
   s0.CountElementsPerVDof(surf_fit_dof_count);
   surf_fit_marker = &smarker;
   surf_fit_coeff = &coeff;
   surf_fit_eval = &ae;

   surf_fit_eval->SetParMetaInfo(*pmesh, *s0.ParFESpace());
   surf_fit_eval->SetInitialField
   (*surf_fit_gf->FESpace()->GetMesh()->GetNodes(), *surf_fit_gf);

   if (!aegrad) { return; }

   MFEM_VERIFY(aehess, "AdaptivityEvaluator for Hessians must be provided too.");

   ParFiniteElementSpace *fes = s0.ParFESpace();

   // FE space for gradients.
   delete surf_fit_grad;
   H1_FECollection *fec_grad = new H1_FECollection(fec->GetOrder(), dim,
                                                   fec->GetBasisType());
   ParFiniteElementSpace *fes_grad = new ParFiniteElementSpace(pmesh, fec_grad,
                                                               dim);
   // Initial gradients.
   surf_fit_grad = new GridFunction(fes_grad);
   surf_fit_grad->MakeOwner(fec_grad);
   for (int d = 0; d < dim; d++)
   {
      ParGridFunction surf_fit_grad_comp(fes, surf_fit_grad->GetData()+d*s0.Size());
      s0.GetDerivative(1, d, surf_fit_grad_comp);
   }
   surf_fit_eval_grad = aegrad;
   surf_fit_eval_grad->SetParMetaInfo(*pmesh, *fes_grad);
   surf_fit_eval_grad->SetInitialField(*pmesh->GetNodes(), *surf_fit_grad);

   // FE space for Hessians.
   delete surf_fit_hess;
   H1_FECollection *fec_hess = new H1_FECollection(fec->GetOrder(), dim,
                                                   fec->GetBasisType());
   ParFiniteElementSpace *fes_hess = new ParFiniteElementSpace(pmesh, fec_hess,
                                                               dim*dim);
   // Initial Hessians.
   surf_fit_hess = new GridFunction(fes_hess);
   surf_fit_hess->MakeOwner(fec_hess);
   int id = 0;
   for (int d = 0; d < dim; d++)
   {
      for (int idir = 0; idir < dim; idir++)
      {
         ParGridFunction surf_fit_grad_comp(fes,
                                            surf_fit_grad->GetData()+d*s0.Size());
         ParGridFunction surf_fit_hess_comp(fes,
                                            surf_fit_hess->GetData()+id*s0.Size());
         surf_fit_grad_comp.GetDerivative(1, idir, surf_fit_hess_comp);
         id++;
      }
   }
   surf_fit_eval_hess = aehess;
   surf_fit_eval_hess->SetParMetaInfo(*pmesh, *fes_hess);
   surf_fit_eval_hess->SetInitialField(*pmesh->GetNodes(), *surf_fit_hess);

   // Store DOF indices that are marked for fitting. Used to reduce work for
   // transferring information between source/background and current mesh.
   surf_fit_marker_dof_index.SetSize(0);
#ifdef MFEM_USE_GSLIB
   if (dynamic_cast<InterpolatorFP *>(surf_fit_eval) &&
       dynamic_cast<InterpolatorFP *>(surf_fit_eval_grad) &&
       dynamic_cast<InterpolatorFP *>(surf_fit_eval_hess))
   {
      for (int i = 0; i < surf_fit_marker->Size(); i++)
      {
         if ((*surf_fit_marker)[i] == true)
         {
            surf_fit_marker_dof_index.Append(i);
         }
      }
   }
#endif

   *surf_fit_grad = 0.0;
   *surf_fit_hess = 0.0;
}

void TMOP_Integrator::EnableSurfaceFittingFromSource(
   const ParGridFunction &s_bg, ParGridFunction &s0,
   const Array<bool> &smarker, Coefficient &coeff, AdaptivityEvaluator &ae,
   const ParGridFunction &s_bg_grad,
   ParGridFunction &s0_grad, AdaptivityEvaluator &age,
   const ParGridFunction &s_bg_hess,
   ParGridFunction &s0_hess, AdaptivityEvaluator &ahe)
{
#ifndef MFEM_USE_GSLIB
   MFEM_ABORT("Surface fitting from source requires GSLIB!");
#endif

   const bool per = s0.FESpace()->IsDGSpace();
   MFEM_VERIFY(per == false, "Fitting is not supported for periodic meshes.");

   // Setup for level set function
   delete surf_fit_gf;
   surf_fit_gf = new GridFunction(s0);
   surf_fit_marker = &smarker;
   surf_fit_coeff = &coeff;
   surf_fit_eval = &ae;
   surf_fit_eval->SetParMetaInfo(*s_bg.ParFESpace()->GetParMesh(),
                                 *s_bg.ParFESpace());
   surf_fit_eval->SetInitialField
   (*s_bg.FESpace()->GetMesh()->GetNodes(), s_bg);
   surf_fit_eval->SetNewFieldFESpace(*surf_fit_gf->FESpace());
   GridFunction *nodes = s0.FESpace()->GetMesh()->GetNodes();
   surf_fit_eval->ComputeAtNewPosition(*nodes, *surf_fit_gf,
                                       nodes->FESpace()->GetOrdering());

   // Setup for gradient on background mesh
   MFEM_VERIFY(s_bg_grad.ParFESpace()->GetOrdering() ==
               s0_grad.ParFESpace()->GetOrdering(),
               "Nodal ordering for grid function on source mesh and current mesh"
               "should be the same.");
   delete surf_fit_grad;
   surf_fit_grad = new GridFunction(s0_grad);
   *surf_fit_grad = 0.0;
   surf_fit_eval_grad = &age;
   surf_fit_eval_grad->SetParMetaInfo(*s_bg_grad.ParFESpace()->GetParMesh(),
                                      *s_bg_grad.ParFESpace());
   surf_fit_eval_grad->SetInitialField
   (*s_bg_grad.FESpace()->GetMesh()->GetNodes(), s_bg_grad);
   surf_fit_eval_grad->SetNewFieldFESpace(*surf_fit_grad->FESpace());

   // Setup for Hessian on background mesh
   MFEM_VERIFY(s_bg_hess.ParFESpace()->GetOrdering() ==
               s0_hess.ParFESpace()->GetOrdering(),
               "Nodal ordering for grid function on source mesh and current mesh"
               "should be the same.");
   delete surf_fit_hess;
   surf_fit_hess = new GridFunction(s0_hess);
   *surf_fit_hess = 0.0;
   surf_fit_eval_hess = &ahe;
   surf_fit_eval_hess->SetParMetaInfo(*s_bg_hess.ParFESpace()->GetParMesh(),
                                      *s_bg_hess.ParFESpace());
   surf_fit_eval_hess->SetInitialField
   (*s_bg_hess.FESpace()->GetMesh()->GetNodes(), s_bg_hess);
   surf_fit_eval_hess->SetNewFieldFESpace(*surf_fit_hess->FESpace());

   // Count number of zones that share each of the DOFs
   s0.CountElementsPerVDof(surf_fit_dof_count);
   // Store DOF indices that are marked for fitting. Used to reduce work for
   // transferring information between source/background and current mesh.
   surf_fit_marker_dof_index.SetSize(0);
   for (int i = 0; i < surf_fit_marker->Size(); i++)
   {
      if ((*surf_fit_marker)[i] == true)
      {
         surf_fit_marker_dof_index.Append(i);
      }
   }
}
#endif

void TMOP_Integrator::GetSurfaceFittingErrors(const Vector &d_loc,
                                              real_t &err_avg, real_t &err_max)
{
   MFEM_VERIFY(periodic == false,
               "Fitting is not supported for periodic meshes.");

   Vector pos(d_loc.Size());
   if (x_0) { add(*x_0, d_loc, pos); }
   else     { pos = d_loc; }

   MFEM_VERIFY(surf_fit_marker, "Surface fitting has not been enabled.");

   const FiniteElementSpace *fes =
      (surf_fit_gf) ? surf_fit_gf->FESpace() : surf_fit_pos->FESpace();
#ifdef MFEM_USE_MPI
   auto pfes = dynamic_cast<const ParFiniteElementSpace *>(fes);
   bool parallel = (pfes) ? true : false;
#endif

   int dim = fes->GetMesh()->Dimension();
   const int node_cnt = surf_fit_marker->Size();
   err_max = 0.0;
   int dof_cnt = 0;
   real_t err_sum = 0.0;
   for (int i = 0; i < node_cnt; i++)
   {
      if ((*surf_fit_marker)[i] == false) { continue; }

#ifdef MFEM_USE_MPI
      // Don't count the overlapping DOFs in parallel.
      // The pfes might be ordered byVDIM, while the loop goes consecutively.
      const int dof_i = pfes->DofToVDof(i, 0);
      if (parallel && pfes->GetLocalTDofNumber(dof_i) < 0) { continue; }
#endif

      dof_cnt++;
      real_t sigma_s = 0.0;
      if (surf_fit_gf) { sigma_s = fabs((*surf_fit_gf)(i)); }
      if (surf_fit_pos)
      {
         Vector pos_s(dim), pos_s_target(dim);
         for (int d = 0; d < dim; d++)
         {
            pos_s(d) = (fes->GetOrdering() == Ordering::byNODES) ?
                       pos(d*node_cnt + i) : pos(i*dim + d);
            pos_s_target(d) = (fes->GetOrdering() == Ordering::byNODES)
                              ? (*surf_fit_pos)(d*node_cnt + i)
                              : (*surf_fit_pos)(i*dim + d);
         }
         sigma_s = pos_s.DistanceTo(pos_s_target);
      }

      err_max  = std::max(err_max, sigma_s);
      err_sum += sigma_s;
   }

#ifdef MFEM_USE_MPI
   if (parallel)
   {
      MPI_Comm comm = pfes->GetComm();
      MPI_Allreduce(MPI_IN_PLACE, &err_max, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX,
                    comm);
      MPI_Allreduce(MPI_IN_PLACE, &dof_cnt, 1, MPI_INT, MPI_SUM, comm);
      MPI_Allreduce(MPI_IN_PLACE, &err_sum, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                    comm);
   }
#endif

   err_avg = (dof_cnt > 0) ? err_sum / dof_cnt : 0.0;
}

void TMOP_Integrator::UpdateAfterMeshTopologyChange()
{
   if (adapt_lim_gf)
   {
      adapt_lim_gf->Update();
      adapt_lim_eval->SetSerialMetaInfo(*adapt_lim_gf->FESpace()->GetMesh(),
                                        *adapt_lim_gf->FESpace());
      adapt_lim_eval->SetInitialField
      (*adapt_lim_gf->FESpace()->GetMesh()->GetNodes(), *adapt_lim_gf);
   }
}

#ifdef MFEM_USE_MPI
void TMOP_Integrator::ParUpdateAfterMeshTopologyChange()
{
   if (adapt_lim_gf)
   {
      adapt_lim_gf->Update();
      adapt_lim_eval->SetParMetaInfo(*adapt_lim_pgf0->ParFESpace()->GetParMesh(),
                                     *adapt_lim_pgf0->ParFESpace());
      adapt_lim_eval->SetInitialField
      (*adapt_lim_gf->FESpace()->GetMesh()->GetNodes(), *adapt_lim_gf);
   }
}
#endif

real_t TMOP_Integrator::GetElementEnergy(const FiniteElement &el,
                                         ElementTransformation &T,
                                         const Vector &d_el)
{
   const int dof = el.GetDof(), dim = el.GetDim();
   const int el_id = T.ElementNo;

   // Form the Vector of node positions, depending on what's the input.
   Vector elfun;
   if (x_0)
   {
      // The input is the displacement.
      x_0->GetElementDofValues(el_id, elfun);
      if (periodic)
      {
         auto n_el = dynamic_cast<const NodalFiniteElement *>(&el);
         n_el->ReorderLexToNative(dim, elfun);
      }
      elfun += d_el;
   }
   else { elfun = d_el; }

   real_t energy;

   // No adaptive limiting / surface fitting terms if the function is called
   // as part of a FD derivative computation (because we include the exact
   // derivatives of these terms in FD computations).
   const bool adaptive_limiting = (adapt_lim_gf && fd_call_flag == false);
   const bool surface_fit = (surf_fit_marker && fd_call_flag == false);

   DSh.SetSize(dof, dim);
   Jrt.SetSize(dim);
   Jpr.SetSize(dim);
   Jpt.SetSize(dim);
   PMatI.UseExternalData(elfun.GetData(), dof, dim);

   const IntegrationRule &ir = EnergyIntegrationRule(el);

   energy = 0.0;
   DenseTensor Jtr(dim, dim, ir.GetNPoints());
   targetC->ComputeElementTargets(el_id, el, ir, elfun, Jtr);

   // Limited case.
   Vector shape, p, p0, d_vals;
   DenseMatrix pos0;
   if (lim_coeff)
   {
      shape.SetSize(dof);
      p.SetSize(dim);
      p0.SetSize(dim);
      pos0.SetSize(dof, dim);
      Vector pos0V(pos0.Data(), dof * dim);
      Array<int> pos_dofs;
      lim_nodes0->FESpace()->GetElementVDofs(el_id, pos_dofs);
      lim_nodes0->GetSubVector(pos_dofs, pos0V);
      if (periodic)
      {
         auto n_el = dynamic_cast<const NodalFiniteElement *>(&el);
         n_el->ReorderLexToNative(dim, pos0V);
      }
      if (lim_dist)
      {
         lim_dist->GetValues(el_id, ir, d_vals);
      }
      else
      {
         d_vals.SetSize(ir.GetNPoints()); d_vals = 1.0;
      }
   }

   // Define ref->physical transformation, when a Coefficient is specified.
   IsoparametricTransformation *Tpr = NULL;
   if (metric_coeff || lim_coeff || adaptive_limiting || surface_fit)
   {
      Tpr = new IsoparametricTransformation;
      Tpr->SetFE(&el);
      Tpr->ElementNo = el_id;
      Tpr->ElementType = ElementTransformation::ELEMENT;
      Tpr->Attribute = T.Attribute;
      Tpr->mesh = T.mesh;
      Tpr->GetPointMat().Transpose(PMatI); // PointMat = PMatI^T
   }
   // TODO: computing the coefficients 'metric_coeff' and 'lim_coeff' in physical
   //       coordinates means that, generally, the gradient and Hessian of the
   //       TMOP_Integrator will depend on the derivatives of the coefficients.
   //
   //       In some cases the coefficients are independent of any movement of
   //       the physical coordinates (i.e. changes in 'elfun'), e.g. when the
   //       coefficient is a ConstantCoefficient or a GridFunctionCoefficient.

   Vector adapt_lim_gf_q, adapt_lim_gf0_q;
   if (adaptive_limiting)
   {
      adapt_lim_gf->GetValues(el_id, ir, adapt_lim_gf_q);
      adapt_lim_gf0->GetValues(el_id, ir, adapt_lim_gf0_q);
   }

   for (int i = 0; i < ir.GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);

      metric->SetTargetJacobian(Jtr(i));
      CalcInverse(Jtr(i), Jrt);
      const real_t weight =
         (integ_over_target) ? ip.weight * Jtr(i).Det() : ip.weight;

      el.CalcDShape(ip, DSh);
      MultAtB(PMatI, DSh, Jpr);
      Mult(Jpr, Jrt, Jpt);

      real_t val = metric_normal * metric->EvalW(Jpt);
      if (metric_coeff) { val *= metric_coeff->Eval(*Tpr, ip); }

      if (lim_coeff)
      {
         el.CalcShape(ip, shape);
         PMatI.MultTranspose(shape, p);
         pos0.MultTranspose(shape, p0);
         val += lim_normal *
                lim_func->Eval(p, p0, d_vals(i)) *
                lim_coeff->Eval(*Tpr, ip);
      }

      // Contribution from the adaptive limiting term.
      if (adaptive_limiting)
      {
         const real_t diff = adapt_lim_gf_q(i) - adapt_lim_gf0_q(i);
         val += adapt_lim_coeff->Eval(*Tpr, ip) * lim_normal * diff * diff;
      }

      energy += weight * val;
   }

   // Contribution from the surface fitting term.
   if (surface_fit)
   {
      // Scalar for surf_fit_gf, vector for surf_fit_pos, but that's ok.
      const FiniteElementSpace *fes_fit =
         (surf_fit_gf) ? surf_fit_gf->FESpace() : surf_fit_pos->FESpace();
      const IntegrationRule *ir_s = &fes_fit->GetFE(el_id)->GetNodes();
      Array<int> vdofs;
      fes_fit->GetElementVDofs(el_id, vdofs);

      Vector sigma_e(dof);
      if (surf_fit_gf) { surf_fit_gf->GetSubVector(vdofs, sigma_e); }

      for (int s = 0; s < dof; s++)
      {
         // Because surf_fit_pos.fes might be ordered byVDIM.
         const int scalar_dof_id = fes_fit->VDofToDof(vdofs[s]);
         if ((*surf_fit_marker)[scalar_dof_id] == false) { continue; }

         const IntegrationPoint &ip_s = ir_s->IntPoint(s);
         Tpr->SetIntPoint(&ip_s);
         real_t w = surf_fit_coeff->Eval(*Tpr, ip_s) * surf_fit_normal *
                    1.0 / surf_fit_dof_count[scalar_dof_id];

         if (surf_fit_gf)
         {
            energy += w * sigma_e(s) * sigma_e(s);
         }
         if (surf_fit_pos)
         {
            // Fitting to exact positions.
            Vector pos(dim), pos_target(dim);
            for (int d = 0; d < dim; d++)
            {
               pos(d) = PMatI(s, d);
               pos_target(d) = (*surf_fit_pos)(vdofs[d*dof + s]);
            }
            energy += w * surf_fit_limiter->Eval(pos, pos_target, 1.0);
         }
      }
   }

   delete Tpr;
   return energy;
}

real_t TMOP_Integrator::GetRefinementElementEnergy(const FiniteElement &el,
                                                   ElementTransformation &T,
                                                   const Vector &elfun,
                                                   const IntegrationRule &irule)
{
   int dof = el.GetDof(), dim = el.GetDim(),
       NEsplit = elfun.Size() / (dof*dim), el_id = T.ElementNo;
   real_t energy = 0.;

   TargetConstructor *tc = const_cast<TargetConstructor *>(targetC);
   DiscreteAdaptTC *dtc = dynamic_cast<DiscreteAdaptTC *>(tc);
   // For DiscreteAdaptTC the GridFunctions used to set the targets must be
   // mapped onto the fine elements.
   if (dtc) { dtc->SetTspecFromIntRule(el_id, irule); }

   for (int e = 0; e < NEsplit; e++)
   {
      DSh.SetSize(dof, dim);
      Jrt.SetSize(dim);
      Jpr.SetSize(dim);
      Jpt.SetSize(dim);
      Vector elfun_child(dof*dim);
      for (int i = 0; i < dof; i++)
      {
         for (int d = 0; d < dim; d++)
         {
            // elfun is (xe1,xe2,...xen,ye1,ye2...yen) and has nodal coordinates
            // for all the children element of the parent element being considered.
            // So we must index and get (xek, yek) i.e. nodal coordinates for
            // the fine element being considered.
            elfun_child(i + d*dof) = elfun(i + e*dof + d*dof*NEsplit);
         }
      }
      PMatI.UseExternalData(elfun_child.GetData(), dof, dim);

      const IntegrationRule &ir = EnergyIntegrationRule(el);

      real_t el_energy = 0;
      DenseTensor Jtr(dim, dim, ir.GetNPoints());
      if (dtc)
      {
         // This is used to index into the tspec vector inside DiscreteAdaptTC.
         dtc->SetRefinementSubElement(e);
      }
      targetC->ComputeElementTargets(el_id, el, ir, elfun_child, Jtr);

      // Define ref->physical transformation, wn a Coefficient is specified.
      IsoparametricTransformation *Tpr = NULL;
      if (metric_coeff || lim_coeff)
      {
         Tpr = new IsoparametricTransformation;
         Tpr->SetFE(&el);
         Tpr->ElementNo = T.ElementNo;
         Tpr->ElementType = ElementTransformation::ELEMENT;
         Tpr->Attribute = T.Attribute;
         Tpr->mesh = T.mesh;
         Tpr->GetPointMat().Transpose(PMatI); // PointMat = PMatI^T
      }

      for (int i = 0; i < ir.GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir.IntPoint(i);
         h_metric->SetTargetJacobian(Jtr(i));
         CalcInverse(Jtr(i), Jrt);
         const real_t weight =
            (integ_over_target) ? ip.weight * Jtr(i).Det() : ip.weight;

         el.CalcDShape(ip, DSh);
         MultAtB(PMatI, DSh, Jpr);
         Mult(Jpr, Jrt, Jpt);

         real_t val = metric_normal * h_metric->EvalW(Jpt);
         if (metric_coeff) { val *= metric_coeff->Eval(*Tpr, ip); }

         el_energy += weight * val;
         delete Tpr;
      }
      energy += el_energy;
   }
   energy /= NEsplit;

   if (dtc) { dtc->ResetRefinementTspecData(); }

   return energy;
}

real_t TMOP_Integrator::GetDerefinementElementEnergy(const FiniteElement &el,
                                                     ElementTransformation &T,
                                                     const Vector &elfun)
{
   int dof = el.GetDof(), dim = el.GetDim();
   real_t energy = 0.;

   DSh.SetSize(dof, dim);
   Jrt.SetSize(dim);
   Jpr.SetSize(dim);
   Jpt.SetSize(dim);
   PMatI.UseExternalData(elfun.GetData(), dof, dim);

   const IntegrationRule &ir = EnergyIntegrationRule(el);

   energy = 0.0;
   DenseTensor Jtr(dim, dim, ir.GetNPoints());
   targetC->ComputeElementTargets(T.ElementNo, el, ir, elfun, Jtr);

   // Define ref->physical transformation, wn a Coefficient is specified.
   IsoparametricTransformation *Tpr = NULL;
   if (metric_coeff)
   {
      Tpr = new IsoparametricTransformation;
      Tpr->SetFE(&el);
      Tpr->ElementNo = T.ElementNo;
      Tpr->ElementType = ElementTransformation::ELEMENT;
      Tpr->Attribute = T.Attribute;
      Tpr->mesh = T.mesh;
      Tpr->GetPointMat().Transpose(PMatI); // PointMat = PMatI^T
   }

   for (int i = 0; i < ir.GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);
      h_metric->SetTargetJacobian(Jtr(i));
      CalcInverse(Jtr(i), Jrt);
      const real_t weight =
         (integ_over_target) ? ip.weight * Jtr(i).Det() : ip.weight;

      el.CalcDShape(ip, DSh);
      MultAtB(PMatI, DSh, Jpr);
      Mult(Jpr, Jrt, Jpt);

      real_t val = metric_normal * h_metric->EvalW(Jpt);
      if (metric_coeff) { val *= metric_coeff->Eval(*Tpr, ip); }

      energy += weight * val;
   }

   delete Tpr;
   return energy;
}

void TMOP_Integrator::AssembleElementVector(const FiniteElement &el,
                                            ElementTransformation &T,
                                            const Vector &d_el, Vector &elvect)
{
   if (!fdflag)
   {
      AssembleElementVectorExact(el, T, d_el, elvect);
   }
   else
   {
      AssembleElementVectorFD(el, T, d_el, elvect);
   }
}

void TMOP_Integrator::AssembleElementGrad(const FiniteElement &el,
                                          ElementTransformation &T,
                                          const Vector &d_el,
                                          DenseMatrix &elmat)
{
   if (!fdflag)
   {
      AssembleElementGradExact(el, T, d_el, elmat);
   }
   else
   {
      AssembleElementGradFD(el, T, d_el, elmat);
   }
}

void TMOP_Integrator::AssembleElementVectorExact(const FiniteElement &el,
                                                 ElementTransformation &T,
                                                 const Vector &d_el,
                                                 Vector &elvect)
{
   const int dof = el.GetDof(), dim = el.GetDim();
   const int el_id = T.ElementNo;

   // Form the Vector of node positions, depending on what's the input.
   Vector elfun;
   if (x_0)
   {
      // The input is the displacement.
      x_0->GetElementDofValues(el_id, elfun);
      if (periodic)
      {
         auto n_el = dynamic_cast<const NodalFiniteElement *>(&el);
         n_el->ReorderLexToNative(dim, elfun);
      }
      elfun += d_el;
   }
   else { elfun = d_el; }

   DenseMatrix Amat(dim), work1(dim), work2(dim);
   DSh.SetSize(dof, dim);
   DS.SetSize(dof, dim);
   Jrt.SetSize(dim);
   Jpt.SetSize(dim);
   P.SetSize(dim);
   PMatI.UseExternalData(elfun.GetData(), dof, dim);
   elvect.SetSize(dof*dim);
   PMatO.UseExternalData(elvect.GetData(), dof, dim);

   const IntegrationRule &ir = ActionIntegrationRule(el);
   const int nqp = ir.GetNPoints();

   elvect = 0.0;
   Vector weights(nqp);
   DenseTensor Jtr(dim, dim, nqp);
   DenseTensor dJtr(dim, dim, dim*nqp);
   targetC->ComputeElementTargets(el_id, el, ir, elfun, Jtr);

   // Limited case.
   DenseMatrix pos0;
   Vector shape, p, p0, d_vals, grad;
   shape.SetSize(dof);
   if (lim_coeff)
   {
      p.SetSize(dim);
      p0.SetSize(dim);
      pos0.SetSize(dof, dim);
      Vector pos0V(pos0.Data(), dof * dim);
      Array<int> pos_dofs;
      lim_nodes0->FESpace()->GetElementVDofs(el_id, pos_dofs);
      lim_nodes0->GetSubVector(pos_dofs, pos0V);
      if (periodic)
      {
         auto n_el = dynamic_cast<const NodalFiniteElement *>(&el);
         n_el->ReorderLexToNative(dim, pos0V);
      }
      if (lim_dist)
      {
         lim_dist->GetValues(el_id, ir, d_vals);
      }
      else
      {
         d_vals.SetSize(nqp); d_vals = 1.0;
      }
   }

   // Define ref->physical transformation, when a Coefficient is specified.
   IsoparametricTransformation *Tpr = NULL;
   if (metric_coeff || lim_coeff || adapt_lim_gf ||
       surf_fit_gf || surf_fit_pos || exact_action)
   {
      Tpr = new IsoparametricTransformation;
      Tpr->SetFE(&el);
      Tpr->ElementNo = el_id;
      Tpr->ElementType = ElementTransformation::ELEMENT;
      Tpr->Attribute = T.Attribute;
      Tpr->mesh = T.mesh;
      Tpr->GetPointMat().Transpose(PMatI); // PointMat = PMatI^T
      if (exact_action)
      {
         targetC->ComputeElementTargetsGradient(ir, elfun, *Tpr, dJtr);
      }
   }

   Vector d_detW_dx(dim);
   Vector d_Winv_dx(dim);

   for (int q = 0; q < nqp; q++)
   {
      const IntegrationPoint &ip = ir.IntPoint(q);
      metric->SetTargetJacobian(Jtr(q));
      CalcInverse(Jtr(q), Jrt);
      weights(q) = (integ_over_target) ? ip.weight * Jtr(q).Det() : ip.weight;
      real_t weight_m = weights(q) * metric_normal;

      el.CalcDShape(ip, DSh);
      Mult(DSh, Jrt, DS);
      MultAtB(PMatI, DS, Jpt);

      metric->EvalP(Jpt, P);

      if (metric_coeff) { weight_m *= metric_coeff->Eval(*Tpr, ip); }
      P *= weight_m;
      AddMultABt(DS, P, PMatO); // w_q det(W) dmu/dx : dA/dx Winv

      if (exact_action)
      {
         el.CalcShape(ip, shape);
         // Derivatives of adaptivity-based targets.
         // First term: w_q d*(Det W)/dx * mu(T)
         // d(Det W)/dx = det(W)*Tr[Winv*dW/dx]
         DenseMatrix dwdx(dim);
         for (int d = 0; d < dim; d++)
         {
            const DenseMatrix &dJtr_q = dJtr(q + d * nqp);
            Mult(Jrt, dJtr_q, dwdx);
            d_detW_dx(d) = dwdx.Trace();
         }
         d_detW_dx *= weight_m*metric->EvalW(Jpt); // *[w_q*det(W)]*mu(T)

         // Second term: w_q det(W) dmu/dx : AdWinv/dx
         // dWinv/dx = -Winv*dW/dx*Winv
         MultAtB(PMatI, DSh, Amat);
         for (int d = 0; d < dim; d++)
         {
            const DenseMatrix &dJtr_q = dJtr(q + d*nqp);
            Mult(Jrt, dJtr_q, work1); // Winv*dw/dx
            Mult(work1, Jrt, work2);  // Winv*dw/dx*Winv
            Mult(Amat, work2, work1); // A*Winv*dw/dx*Winv
            MultAtB(P, work1, work2); // dmu/dT^T*A*Winv*dw/dx*Winv
            d_Winv_dx(d) = work2.Trace(); // Tr[dmu/dT : AWinv*dw/dx*Winv]
         }
         d_Winv_dx *= -weight_m; // Include (-) factor as well
         d_detW_dx += d_Winv_dx;

         AddMultVWt(shape, d_detW_dx, PMatO);

         // For mu(T,W) we also need w_q dmu/dW:dW/dx det(W)
         // dmu/dW:dW/dx_i
         DenseMatrix PW(dim);
         Vector dmudxw(dim);
         metric->EvalPW(Jpt, PW);
         DenseMatrix Prod(dim);

         for (int d = 0; d < dim; d++)
         {
            const DenseMatrix &dJtr_q = dJtr(q + d*nqp);
            Prod = 0.0;
            MultAtB(PW, dJtr_q, Prod); // dmu/dW:dW/dx_i
            dmudxw(d) = Prod.Trace();
         }
         dmudxw *= weight_m;
         AddMultVWt(shape, dmudxw, PMatO);
      }

      if (lim_coeff)
      {
         if (!exact_action) { el.CalcShape(ip, shape); }
         PMatI.MultTranspose(shape, p);
         pos0.MultTranspose(shape, p0);
         lim_func->Eval_d1(p, p0, d_vals(q), grad);
         grad *= weights(q) * lim_normal * lim_coeff->Eval(*Tpr, ip);
         AddMultVWt(shape, grad, PMatO);
      }
   }

   if (adapt_lim_gf) { AssembleElemVecAdaptLim(el, *Tpr, ir, weights, PMatO); }
   if (surf_fit_gf || surf_fit_pos) { AssembleElemVecSurfFit(el, *Tpr, PMatO); }

   delete Tpr;
}

void TMOP_Integrator::AssembleElementGradExact(const FiniteElement &el,
                                               ElementTransformation &T,
                                               const Vector &d_el,
                                               DenseMatrix &elmat)
{
   const int dof = el.GetDof(), dim = el.GetDim();
   const int el_id = T.ElementNo;

   // Form the Vector of node positions, depending on what's the input.
   Vector elfun;
   if (x_0)
   {
      // The input is the displacement.
      x_0->GetElementDofValues(el_id, elfun);
      if (periodic)
      {
         auto n_el = dynamic_cast<const NodalFiniteElement *>(&el);
         n_el->ReorderLexToNative(dim, elfun);
      }
      elfun += d_el;
   }
   else { elfun = d_el; }

   DSh.SetSize(dof, dim);
   DS.SetSize(dof, dim);
   Jrt.SetSize(dim);
   Jpt.SetSize(dim);
   PMatI.UseExternalData(elfun.GetData(), dof, dim);
   elmat.SetSize(dof*dim);

   const IntegrationRule &ir = GradientIntegrationRule(el);
   const int nqp = ir.GetNPoints();

   elmat = 0.0;
   Vector weights(nqp);
   DenseTensor Jtr(dim, dim, nqp);
   targetC->ComputeElementTargets(el_id, el, ir, elfun, Jtr);

   // Limited case.
   DenseMatrix pos0, hess;
   Vector shape, p, p0, d_vals;
   if (lim_coeff)
   {
      shape.SetSize(dof);
      p.SetSize(dim);
      p0.SetSize(dim);
      pos0.SetSize(dof, dim);
      Vector pos0V(pos0.Data(), dof * dim);
      Array<int> pos_dofs;
      lim_nodes0->FESpace()->GetElementVDofs(el_id, pos_dofs);
      lim_nodes0->GetSubVector(pos_dofs, pos0V);
      if (periodic)
      {
         auto n_el = dynamic_cast<const NodalFiniteElement *>(&el);
         n_el->ReorderLexToNative(dim, pos0V);
      }
      if (lim_dist)
      {
         lim_dist->GetValues(el_id, ir, d_vals);
      }
      else
      {
         d_vals.SetSize(nqp); d_vals = 1.0;
      }
   }

   // Define ref->physical transformation, when a Coefficient is specified.
   IsoparametricTransformation *Tpr = NULL;
   if (metric_coeff || lim_coeff || adapt_lim_gf || surf_fit_gf || surf_fit_pos)
   {
      Tpr = new IsoparametricTransformation;
      Tpr->SetFE(&el);
      Tpr->ElementNo = T.ElementNo;
      Tpr->ElementType = ElementTransformation::ELEMENT;
      Tpr->Attribute = T.Attribute;
      Tpr->mesh = T.mesh;
      Tpr->GetPointMat().Transpose(PMatI);
   }

   for (int q = 0; q < nqp; q++)
   {
      const IntegrationPoint &ip = ir.IntPoint(q);
      const DenseMatrix &Jtr_q = Jtr(q);
      metric->SetTargetJacobian(Jtr_q);
      CalcInverse(Jtr_q, Jrt);
      weights(q) = (integ_over_target) ? ip.weight * Jtr_q.Det() : ip.weight;
      real_t weight_m = weights(q) * metric_normal;

      el.CalcDShape(ip, DSh);
      Mult(DSh, Jrt, DS);
      MultAtB(PMatI, DS, Jpt);

      if (metric_coeff) { weight_m *= metric_coeff->Eval(*Tpr, ip); }

      metric->AssembleH(Jpt, DS, weight_m, elmat);

      // TODO: derivatives of adaptivity-based targets.

      if (lim_coeff)
      {
         el.CalcShape(ip, shape);
         PMatI.MultTranspose(shape, p);
         pos0.MultTranspose(shape, p0);
         weight_m = weights(q) * lim_normal * lim_coeff->Eval(*Tpr, ip);
         lim_func->Eval_d2(p, p0, d_vals(q), hess);
         for (int i = 0; i < dof; i++)
         {
            const real_t w_shape_i = weight_m * shape(i);
            for (int j = 0; j < dof; j++)
            {
               const real_t w = w_shape_i * shape(j);
               for (int d1 = 0; d1 < dim; d1++)
               {
                  for (int d2 = 0; d2 < dim; d2++)
                  {
                     elmat(d1*dof + i, d2*dof + j) += w * hess(d1, d2);
                  }
               }
            }
         }
      }
   }

   if (adapt_lim_gf) { AssembleElemGradAdaptLim(el, *Tpr, ir, weights, elmat); }
   if (surf_fit_gf || surf_fit_pos) { AssembleElemGradSurfFit(el, *Tpr, elmat);}

   delete Tpr;
}

void TMOP_Integrator::AssembleElemVecAdaptLim(const FiniteElement &el,
                                              IsoparametricTransformation &Tpr,
                                              const IntegrationRule &ir,
                                              const Vector &weights,
                                              DenseMatrix &mat)
{
   const int dof = el.GetDof(), dim = el.GetDim(), nqp = weights.Size();
   Vector shape(dof), adapt_lim_gf_e, adapt_lim_gf_q, adapt_lim_gf0_q(nqp);

   Array<int> dofs;
   adapt_lim_gf->FESpace()->GetElementDofs(Tpr.ElementNo, dofs);
   adapt_lim_gf->GetSubVector(dofs, adapt_lim_gf_e);
   adapt_lim_gf->GetValues(Tpr.ElementNo, ir, adapt_lim_gf_q);
   adapt_lim_gf0->GetValues(Tpr.ElementNo, ir, adapt_lim_gf0_q);

   // Project the gradient of adapt_lim_gf in the same space.
   // The FE coefficients of the gradient go in adapt_lim_gf_grad_e.
   DenseMatrix adapt_lim_gf_grad_e(dof, dim);
   DenseMatrix grad_phys; // This will be (dof x dim, dof).
   el.ProjectGrad(el, Tpr, grad_phys);
   Vector grad_ptr(adapt_lim_gf_grad_e.GetData(), dof*dim);
   grad_phys.Mult(adapt_lim_gf_e, grad_ptr);

   Vector adapt_lim_gf_grad_q(dim);

   for (int q = 0; q < nqp; q++)
   {
      const IntegrationPoint &ip = ir.IntPoint(q);
      el.CalcShape(ip, shape);
      adapt_lim_gf_grad_e.MultTranspose(shape, adapt_lim_gf_grad_q);
      adapt_lim_gf_grad_q *= 2.0 * (adapt_lim_gf_q(q) - adapt_lim_gf0_q(q));
      adapt_lim_gf_grad_q *= weights(q) * lim_normal * adapt_lim_coeff->Eval(Tpr, ip);
      AddMultVWt(shape, adapt_lim_gf_grad_q, mat);
   }
}

void TMOP_Integrator::AssembleElemGradAdaptLim(const FiniteElement &el,
                                               IsoparametricTransformation &Tpr,
                                               const IntegrationRule &ir,
                                               const Vector &weights,
                                               DenseMatrix &mat)
{
   const int dof = el.GetDof(), dim = el.GetDim(), nqp = weights.Size();
   Vector shape(dof), adapt_lim_gf_e, adapt_lim_gf_q, adapt_lim_gf0_q(nqp);

   Array<int> dofs;
   adapt_lim_gf->FESpace()->GetElementDofs(Tpr.ElementNo, dofs);
   adapt_lim_gf->GetSubVector(dofs, adapt_lim_gf_e);
   adapt_lim_gf->GetValues(Tpr.ElementNo, ir, adapt_lim_gf_q);
   adapt_lim_gf0->GetValues(Tpr.ElementNo, ir, adapt_lim_gf0_q);

   // Project the gradient of adapt_lim_gf in the same space.
   // The FE coefficients of the gradient go in adapt_lim_gf_grad_e.
   DenseMatrix adapt_lim_gf_grad_e(dof, dim);
   DenseMatrix grad_phys; // This will be (dof x dim, dof).
   el.ProjectGrad(el, Tpr, grad_phys);
   Vector grad_ptr(adapt_lim_gf_grad_e.GetData(), dof*dim);
   grad_phys.Mult(adapt_lim_gf_e, grad_ptr);

   // Project the gradient of each gradient of adapt_lim_gf in the same space.
   // The FE coefficients of the second derivatives go in adapt_lim_gf_hess_e.
   DenseMatrix adapt_lim_gf_hess_e(dof*dim, dim);
   Mult(grad_phys, adapt_lim_gf_grad_e, adapt_lim_gf_hess_e);
   // Reshape to be more convenient later (no change in the data).
   adapt_lim_gf_hess_e.SetSize(dof, dim*dim);

   Vector adapt_lim_gf_grad_q(dim);
   DenseMatrix adapt_lim_gf_hess_q(dim, dim);

   for (int q = 0; q < nqp; q++)
   {
      const IntegrationPoint &ip = ir.IntPoint(q);
      el.CalcShape(ip, shape);

      adapt_lim_gf_grad_e.MultTranspose(shape, adapt_lim_gf_grad_q);
      Vector gg_ptr(adapt_lim_gf_hess_q.GetData(), dim*dim);
      adapt_lim_gf_hess_e.MultTranspose(shape, gg_ptr);

      const real_t w = weights(q) * lim_normal * adapt_lim_coeff->Eval(Tpr, ip);
      for (int i = 0; i < dof * dim; i++)
      {
         const int idof = i % dof, idim = i / dof;
         for (int j = 0; j <= i; j++)
         {
            const int jdof = j % dof, jdim = j / dof;
            const real_t entry =
               w * ( 2.0 * adapt_lim_gf_grad_q(idim) * shape(idof) *
                     /* */ adapt_lim_gf_grad_q(jdim) * shape(jdof) +
                     2.0 * (adapt_lim_gf_q(q) - adapt_lim_gf0_q(q)) *
                     adapt_lim_gf_hess_q(idim, jdim) * shape(idof) * shape(jdof));
            mat(i, j) += entry;
            if (i != j) { mat(j, i) += entry; }
         }
      }
   }
}

void TMOP_Integrator::AssembleElemVecSurfFit(const FiniteElement &el_x,
                                             IsoparametricTransformation &Tpr,
                                             DenseMatrix &mat)
{
   const int el_id = Tpr.ElementNo;

   // Scalar for surf_fit_gf, vector for surf_fit_pos, but that's ok.
   const FiniteElementSpace *fes_fit =
      (surf_fit_gf) ? surf_fit_gf->FESpace() : surf_fit_pos->FESpace();
   const FiniteElement &el_s = *fes_fit->GetFE(el_id);
   const int dof_s = el_s.GetDof(), dim = el_x.GetDim();

   // Check if the element has any DOFs marked for surface fitting.
   Array<int> dofs, vdofs;
   fes_fit->GetElementVDofs(el_id, vdofs);
   int count = 0;
   for (int s = 0; s < dof_s; s++)
   {
      // Because surf_fit_pos.fes might be ordered byVDIM.
      const int scalar_dof_id = fes_fit->VDofToDof(vdofs[s]);
      count += ((*surf_fit_marker)[scalar_dof_id]) ? 1 : 0;
   }
   if (count == 0) { return; }

   Vector sigma_e(dof_s);
   DenseMatrix surf_fit_grad_e(dof_s, dim);
   if (surf_fit_gf)
   {
      surf_fit_gf->GetSubVector(vdofs, sigma_e);

      // Project the gradient of sigma in the same space.
      // The FE coefficients of the gradient go in surf_fit_grad_e.
      Vector grad_ptr(surf_fit_grad_e.GetData(), dof_s * dim);
      DenseMatrix grad_phys; // This will be (dof x dim, dof).
      if (surf_fit_grad)
      {
         surf_fit_grad->FESpace()->GetElementVDofs(el_id, dofs);
         surf_fit_grad->GetSubVector(dofs, grad_ptr);
      }
      else
      {
         el_s.ProjectGrad(el_s, Tpr, grad_phys);
         grad_phys.Mult(sigma_e, grad_ptr);
      }
   }
   else { Tpr.GetPointMat().Transpose(PMatI); }

   const IntegrationRule &ir = el_s.GetNodes();

   for (int s = 0; s < dof_s; s++)
   {
      // Because surf_fit_pos.fes might be ordered byVDIM.
      const int scalar_dof_id = fes_fit->VDofToDof(vdofs[s]);
      if ((*surf_fit_marker)[scalar_dof_id] == false) { continue; }

      const IntegrationPoint &ip = ir.IntPoint(s);
      Tpr.SetIntPoint(&ip);
      real_t w = surf_fit_normal * surf_fit_coeff->Eval(Tpr, ip) *
                 1.0 / surf_fit_dof_count[vdofs[s]];

      if (surf_fit_gf) { w *= 2.0 * sigma_e(s); }
      if (surf_fit_pos)
      {
         Vector pos(dim), pos_target(dim);
         for (int d = 0; d < dim; d++)
         {
            pos(d) = PMatI(s, d);
            pos_target(d) = (*surf_fit_pos)(vdofs[d*dof_s + s]);
         }
         Vector grad_s(dim);
         surf_fit_limiter->Eval_d1(pos, pos_target, 1.0, grad_s);
         for (int d = 0; d < dim; d++) { surf_fit_grad_e(s, d) = grad_s(d); }
      }

      for (int d = 0; d < dim; d++)
      {
         mat(s, d) += w * surf_fit_grad_e(s, d);
      }
   }
}

void TMOP_Integrator::AssembleElemGradSurfFit(const FiniteElement &el_x,
                                              IsoparametricTransformation &Tpr,
                                              DenseMatrix &mat)
{
   const int el_id = Tpr.ElementNo;

   // Scalar for surf_fit_gf, vector for surf_fit_pos, but that's ok.
   const FiniteElementSpace *fes_fit =
      (surf_fit_gf) ? surf_fit_gf->FESpace() : surf_fit_pos->FESpace();
   const FiniteElement &el_s = *fes_fit->GetFE(el_id);
   const int dof_s = el_s.GetDof(), dim = el_x.GetDim();

   // Check if the element has any DOFs marked for surface fitting.
   Array<int> dofs, vdofs;
   fes_fit->GetElementVDofs(el_id, vdofs);
   int count = 0;
   for (int s = 0; s < dof_s; s++)
   {
      // Because surf_fit_pos.fes might be ordered byVDIM.
      const int scalar_dof_id = fes_fit->VDofToDof(vdofs[s]);
      count += ((*surf_fit_marker)[scalar_dof_id]) ? 1 : 0;
   }
   if (count == 0) { return; }

   Vector sigma_e(dof_s);
   DenseMatrix surf_fit_grad_e(dof_s, dim);
   DenseMatrix surf_fit_hess_e(dof_s, dim*dim);
   if (surf_fit_gf)
   {
      surf_fit_gf->GetSubVector(vdofs, sigma_e);

      // Project the gradient of sigma in the same space.
      // The FE coefficients of the gradient go in surf_fit_grad_e.
      Vector grad_ptr(surf_fit_grad_e.GetData(), dof_s * dim);
      DenseMatrix grad_phys; // This will be (dof x dim, dof).
      if (surf_fit_grad)
      {
         surf_fit_grad->FESpace()->GetElementVDofs(el_id, dofs);
         surf_fit_grad->GetSubVector(dofs, grad_ptr);
      }
      else
      {
         el_s.ProjectGrad(el_s, Tpr, grad_phys);
         grad_phys.Mult(sigma_e, grad_ptr);
      }

      // Project the Hessian of sigma in the same space.
      // The FE coefficients of the Hessian go in surf_fit_hess_e.
      Vector hess_ptr(surf_fit_hess_e.GetData(), dof_s*dim*dim);
      if (surf_fit_hess)
      {
         surf_fit_hess->FESpace()->GetElementVDofs(el_id, dofs);
         surf_fit_hess->GetSubVector(dofs, hess_ptr);
      }
      else
      {
         surf_fit_hess_e.SetSize(dof_s*dim, dim);
         Mult(grad_phys, surf_fit_grad_e, surf_fit_hess_e);
         surf_fit_hess_e.SetSize(dof_s, dim * dim);
      }
   }
   else { Tpr.GetPointMat().Transpose(PMatI); }

   const IntegrationRule &ir = el_s.GetNodes();

   DenseMatrix surf_fit_hess_s(dim, dim);
   for (int s = 0; s < dof_s; s++)
   {
      // Because surf_fit_pos.fes might be ordered byVDIM.
      const int scalar_dof_id = fes_fit->VDofToDof(vdofs[s]);
      if ((*surf_fit_marker)[scalar_dof_id] == false) { continue; }

      const IntegrationPoint &ip = ir.IntPoint(s);
      Tpr.SetIntPoint(&ip);
      real_t w = surf_fit_normal * surf_fit_coeff->Eval(Tpr, ip);

      if (surf_fit_gf)
      {
         Vector gg_ptr(surf_fit_hess_s.GetData(), dim * dim);
         surf_fit_hess_e.GetRow(s, gg_ptr);
         w *= 2.0;
      }
      if (surf_fit_pos)
      {
         Vector pos(dim), pos_target(dim);
         for (int d = 0; d < dim; d++)
         {
            pos(d) = PMatI(s, d);
            pos_target(d) = (*surf_fit_pos)(vdofs[d*dof_s + s]);
         }
         // Eval_d2 returns the full Hessian, but we still use the general
         // computation that's in the dim x dim loop below.
         sigma_e(s) = 1.0;
         for (int d = 0; d < dim; d++) { surf_fit_grad_e(s, d) = 0.0; }
         surf_fit_limiter->Eval_d2(pos, pos_target, 1.0, surf_fit_hess_s);
      }

      // Loops over the local matrix.
      for (int idim = 0; idim < dim; idim++)
      {
         for (int jdim = 0; jdim <= idim; jdim++)
         {
            real_t entry = w * ( surf_fit_grad_e(s, idim) *
                                 surf_fit_grad_e(s, jdim) +
                                 sigma_e(s) * surf_fit_hess_s(idim, jdim));
            entry *= 1.0 / surf_fit_dof_count[vdofs[s]];
            int idx = s + idim*dof_s;
            int jdx = s + jdim*dof_s;
            mat(idx, jdx) += entry;
            if (idx != jdx) { mat(jdx, idx) += entry; }
         }
      }
   }
}

real_t TMOP_Integrator::GetFDDerivative(const FiniteElement &el,
                                        ElementTransformation &T,
                                        Vector &d_el, const int dofidx,
                                        const int dir, const real_t e_fx,
                                        bool update_stored)
{
   int dof = el.GetDof();
   int idx = dir*dof+dofidx;
   d_el[idx]     += fd_h;
   real_t e_fxph = GetElementEnergy(el, T, d_el);
   d_el[idx]     -= fd_h;
   real_t dfdx   = (e_fxph - e_fx) / fd_h;

   if (update_stored)
   {
      (*(ElemPertEnergy[T.ElementNo]))(idx) = e_fxph;
      (*(ElemDer[T.ElementNo]))(idx) = dfdx;
   }

   return dfdx;
}

void TMOP_Integrator::AssembleElementVectorFD(const FiniteElement &el,
                                              ElementTransformation &T,
                                              const Vector &d_el,
                                              Vector &elvect)
{
   // Form the Vector of node positions, depending on what's the input.
   Vector elfun;
   if (x_0)
   {
      // The input is the displacement.
      x_0->GetElementDofValues(T.ElementNo, elfun);
      elfun += d_el;
   }
   else { elfun = d_el; }

   const int dof = el.GetDof(), dim = el.GetDim(), elnum = T.ElementNo;
   if (elnum >= ElemDer.Size())
   {
      ElemDer.Append(new Vector);
      ElemPertEnergy.Append(new Vector);
      ElemDer[elnum]->SetSize(dof*dim);
      ElemPertEnergy[elnum]->SetSize(dof*dim);
   }

   elvect.SetSize(dof*dim);

   // In GetElementEnergy(), skip terms that have exact derivative calculations.
   fd_call_flag = true;

   // Energy for unperturbed configuration.
   const real_t e_fx = GetElementEnergy(el, T, d_el);

   Vector d_el_mod(d_el);
   for (int j = 0; j < dim; j++)
   {
      for (int i = 0; i < dof; i++)
      {
         if (discr_tc)
         {
            discr_tc->UpdateTargetSpecificationAtNode(
               el, T, i, j, discr_tc->GetTspecPert1H());
         }
         elvect(j*dof+i) = GetFDDerivative(el, T, d_el_mod, i, j, e_fx, true);
         if (discr_tc) { discr_tc->RestoreTargetSpecificationAtNode(T, i); }
      }
   }
   fd_call_flag = false;

   // Contributions from adaptive limiting, surface fitting (exact derivatives).
   if (adapt_lim_gf || surf_fit_gf || surf_fit_pos)
   {
      const IntegrationRule &ir = ActionIntegrationRule(el);
      const int nqp = ir.GetNPoints();
      DenseTensor Jtr(dim, dim, nqp);
      targetC->ComputeElementTargets(T.ElementNo, el, ir, elfun, Jtr);

      IsoparametricTransformation Tpr;
      Tpr.SetFE(&el);
      Tpr.ElementNo = T.ElementNo;
      Tpr.Attribute = T.Attribute;
      Tpr.mesh = T.mesh;
      PMatI.UseExternalData(elfun.GetData(), dof, dim);
      Tpr.GetPointMat().Transpose(PMatI); // PointMat = PMatI^T

      Vector weights(nqp);
      for (int q = 0; q < nqp; q++)
      {
         weights(q) = (integ_over_target) ?
                      ir.IntPoint(q).weight * Jtr(q).Det() :
                      ir.IntPoint(q).weight;
      }

      PMatO.UseExternalData(elvect.GetData(), dof, dim);
      if (adapt_lim_gf) { AssembleElemVecAdaptLim(el, Tpr, ir, weights, PMatO); }
      if (surf_fit_gf || surf_fit_pos) { AssembleElemVecSurfFit(el, Tpr, PMatO); }
   }
}

void TMOP_Integrator::AssembleElementGradFD(const FiniteElement &el,
                                            ElementTransformation &T,
                                            const Vector &d_el,
                                            DenseMatrix &elmat)
{
   // Form the Vector of node positions, depending on what's the input.
   Vector elfun;
   if (x_0)
   {
      // The input is the displacement.
      x_0->GetElementDofValues(T.ElementNo, elfun);
      elfun += d_el;
   }
   else { elfun = d_el; }

   const int dof = el.GetDof(), dim = el.GetDim();

   elmat.SetSize(dof*dim);

   const Vector &ElemDerLoc = *(ElemDer[T.ElementNo]);
   const Vector &ElemPertLoc = *(ElemPertEnergy[T.ElementNo]);

   // In GetElementEnergy(), skip terms that have exact derivative calculations.
   Vector d_el_mod(d_el);
   fd_call_flag = true;
   for (int i = 0; i < dof; i++)
   {
      for (int j = 0; j < i+1; j++)
      {
         for (int k1 = 0; k1 < dim; k1++)
         {
            for (int k2 = 0; k2 < dim; k2++)
            {
               d_el_mod(k2 * dof + j) += fd_h;

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

               real_t e_fx    = ElemPertLoc(k2 * dof + j);
               real_t e_fpxph = GetFDDerivative(el, T, d_el_mod, i, k1, e_fx,
                                                false);
               d_el_mod(k2 * dof + j) -= fd_h;
               real_t e_fpx = ElemDerLoc(k1*dof+i);

               elmat(k1*dof+i, k2*dof+j) = (e_fpxph - e_fpx) / fd_h;
               elmat(k2*dof+j, k1*dof+i) = (e_fpxph - e_fpx) / fd_h;

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
   if (adapt_lim_gf || surf_fit_gf || surf_fit_pos)
   {
      const IntegrationRule &ir = GradientIntegrationRule(el);
      const int nqp = ir.GetNPoints();
      DenseTensor Jtr(dim, dim, nqp);
      targetC->ComputeElementTargets(T.ElementNo, el, ir, elfun, Jtr);

      IsoparametricTransformation Tpr;
      Tpr.SetFE(&el);
      Tpr.ElementNo = T.ElementNo;
      Tpr.Attribute = T.Attribute;
      Tpr.mesh = T.mesh;
      PMatI.UseExternalData(elfun.GetData(), dof, dim);
      Tpr.GetPointMat().Transpose(PMatI); // PointMat = PMatI^T

      Vector weights(nqp);
      for (int q = 0; q < nqp; q++)
      {
         weights(q) = (integ_over_target) ?
                      ir.IntPoint(q).weight * Jtr(q).Det() :
                      ir.IntPoint(q).weight;
      }

      if (adapt_lim_gf) { AssembleElemGradAdaptLim(el, Tpr, ir, weights, elmat); }
      if (surf_fit_gf || surf_fit_pos) { AssembleElemGradSurfFit(el, Tpr, elmat); }
   }
}

void TMOP_Integrator::UpdateSurfaceFittingWeight(real_t factor)
{
   if (!surf_fit_coeff) { return; }

   if (surf_fit_coeff)
   {
      auto cf = dynamic_cast<ConstantCoefficient *>(surf_fit_coeff);
      MFEM_VERIFY(cf, "Dynamic weight works only with a ConstantCoefficient.");
      cf->constant *= factor;
   }
}

real_t TMOP_Integrator::GetSurfaceFittingWeight()
{
   if (surf_fit_coeff)
   {
      auto cf = dynamic_cast<ConstantCoefficient *>(surf_fit_coeff);
      MFEM_VERIFY(cf, "Dynamic weight works only with a ConstantCoefficient.");
      return cf->constant;
   }
   return 0.0;
}

void TMOP_Integrator::EnableNormalization(const GridFunction &x)
{
   ComputeNormalizationEnergies(x, metric_normal, lim_normal);
   metric_normal = 1.0 / metric_normal;
   lim_normal = 1.0 / lim_normal;
   if (surf_fit_gf || surf_fit_pos) { surf_fit_normal = lim_normal; }
}

#ifdef MFEM_USE_MPI
void TMOP_Integrator::ParEnableNormalization(const ParGridFunction &x)
{
   real_t loc[2];
   ComputeNormalizationEnergies(x, loc[0], loc[1]);
   real_t rdc[2];
   MPI_Allreduce(loc, rdc, 2, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                 x.ParFESpace()->GetComm());
   metric_normal = 1.0 / rdc[0];
   lim_normal    = 1.0 / rdc[1];
   if (surf_fit_gf || surf_fit_pos) { surf_fit_normal = lim_normal; }
}
#endif

void TMOP_Integrator::ComputeNormalizationEnergies(const GridFunction &x,
                                                   real_t &metric_energy,
                                                   real_t &lim_energy)
{
   metric_energy = 0.0;
   lim_energy = 0.0;
   if (PA.enabled)
   {
      MFEM_VERIFY(PA.E.Size() > 0, "Must be called after AssemblePA!");
      MFEM_VERIFY(surf_fit_gf == nullptr,
                  "Normalization + PA + Fitting is not implemented!");

      const ElementDofOrdering ord = ElementDofOrdering::LEXICOGRAPHIC;
      auto R = x.FESpace()->GetElementRestriction(ord);
      Vector xe(R->Height());
      R->Mult(x, xe);

      // Force update of the target Jacobian.
      ComputeAllElementTargets(xe);

      if (PA.dim == 2)
      {
         GetLocalNormalizationEnergiesPA_2D(xe, metric_energy, lim_energy);
      }
      else
      {
         GetLocalNormalizationEnergiesPA_3D(xe, metric_energy, lim_energy);
      }

      // Cases when integration is not over the target element, or when the
      // targets don't contain volumetric information.
      if (integ_over_target == false || targetC->ContainsVolumeInfo() == false)
      {
         lim_energy = x.FESpace()->GetNE();
      }

      return;
   }

   Array<int> vdofs;
   Vector x_vals;
   const FiniteElementSpace* const fes = x.FESpace();

   const int dim = fes->GetMesh()->Dimension();
   Jrt.SetSize(dim);
   Jpr.SetSize(dim);
   Jpt.SetSize(dim);

   for (int i = 0; i < fes->GetNE(); i++)
   {
      const FiniteElement *fe = fes->GetFE(i);
      const IntegrationRule &ir = EnergyIntegrationRule(*fe);
      const int nqp = ir.GetNPoints();
      DenseTensor Jtr(dim, dim, nqp);
      const int dof = fe->GetDof();
      DSh.SetSize(dof, dim);

      fes->GetElementVDofs(i, vdofs);
      x.GetSubVector(vdofs, x_vals);
      PMatI.UseExternalData(x_vals.GetData(), dof, dim);

      targetC->ComputeElementTargets(i, *fe, ir, x_vals, Jtr);

      for (int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         metric->SetTargetJacobian(Jtr(q));
         CalcInverse(Jtr(q), Jrt);
         const real_t weight =
            (integ_over_target) ? ip.weight * Jtr(q).Det() : ip.weight;

         fe->CalcDShape(ip, DSh);
         MultAtB(PMatI, DSh, Jpr);
         Mult(Jpr, Jrt, Jpt);

         metric_energy += weight * metric->EvalW(Jpt);
         lim_energy += weight;
      }

      // TODO: Normalization of the surface fitting term.
   }

   // Cases when integration is not over the target element, or when the
   // targets don't contain volumetric information.
   if (integ_over_target == false || targetC->ContainsVolumeInfo() == false)
   {
      lim_energy = fes->GetNE();
   }
}

void TMOP_Integrator::ComputeMinJac(const Vector &x,
                                    const FiniteElementSpace &fes)
{
   const FiniteElement *fe = fes.GetTypicalFE();
   const IntegrationRule &ir = EnergyIntegrationRule(*fe);
   const int NE = fes.GetMesh()->GetNE(), dim = fe->GetDim(),
             dof = fe->GetDof(), nsp = ir.GetNPoints();

   Array<int> xdofs(dof * dim);
   DenseMatrix dshape(dof, dim), pos(dof, dim);
   Vector posV(pos.Data(), dof * dim);
   Jpr.SetSize(dim);

   fd_h = std::numeric_limits<float>::max();

   real_t detv_sum;
   real_t detv_avg_min = std::numeric_limits<float>::max();
   for (int i = 0; i < NE; i++)
   {
      fes.GetElementVDofs(i, xdofs);
      x.GetSubVector(xdofs, posV);
      detv_sum = 0.;
      for (int j = 0; j < nsp; j++)
      {
         fes.GetFE(i)->CalcDShape(ir.IntPoint(j), dshape);
         MultAtB(pos, dshape, Jpr);
         detv_sum += std::fabs(Jpr.Det());
      }
      real_t detv_avg = pow(detv_sum/nsp, 1./dim);
      detv_avg_min = std::min(detv_avg, detv_avg_min);
   }
   fd_h = detv_avg_min / fd_h_scale;
}

void TMOP_Integrator::RemapSurfaceFittingLevelSetAtNodes(const Vector &new_x,
                                                         int new_x_ordering)
{
   MFEM_VERIFY(periodic == false, "Periodic not implemented yet.");

   if (!surf_fit_gf) { return; }

   if (surf_fit_marker_dof_index.Size())
   {
      // Interpolate information only at DOFs marked for fitting.
      const int dim = surf_fit_gf->FESpace()->GetMesh()->Dimension();
      const int cnt = surf_fit_marker_dof_index.Size();
      const int total_cnt = new_x.Size()/dim;
      Vector new_x_sorted(cnt*dim);
      if (new_x_ordering == 0)
      {
         for (int d = 0; d < dim; d++)
         {
            for (int i = 0; i < cnt; i++)
            {
               int dof_index = surf_fit_marker_dof_index[i];
               new_x_sorted(i + d*cnt) = new_x(dof_index + d*total_cnt);
            }
         }
      }
      else
      {
         for (int i = 0; i < cnt; i++)
         {
            int dof_index = surf_fit_marker_dof_index[i];
            for (int d = 0; d < dim; d++)
            {
               new_x_sorted(d + i*dim) = new_x(d + dof_index*dim);
            }
         }
      }

      // Interpolate values of the LS.
      Vector surf_fit_gf_int, surf_fit_grad_int, surf_fit_hess_int;
      surf_fit_eval->ComputeAtGivenPositions(new_x_sorted, surf_fit_gf_int,
                                             new_x_ordering);
      for (int i = 0; i < cnt; i++)
      {
         int dof_index = surf_fit_marker_dof_index[i];
         (*surf_fit_gf)[dof_index] = surf_fit_gf_int(i);
      }

      // Interpolate gradients of the LS.
      surf_fit_eval_grad->ComputeAtGivenPositions(new_x_sorted,
                                                  surf_fit_grad_int,
                                                  new_x_ordering);
      // Assumes surf_fit_grad and surf_fit_gf share the same space
      const int grad_dim = surf_fit_grad->VectorDim();
      const int grad_cnt = surf_fit_grad->Size()/grad_dim;
      if (surf_fit_grad->FESpace()->GetOrdering() == Ordering::byNODES)
      {
         for (int d = 0; d < grad_dim; d++)
         {
            for (int i = 0; i < cnt; i++)
            {
               int dof_index = surf_fit_marker_dof_index[i];
               (*surf_fit_grad)[dof_index + d*grad_cnt] =
                  surf_fit_grad_int(i + d*cnt);
            }
         }
      }
      else
      {
         for (int i = 0; i < cnt; i++)
         {
            int dof_index = surf_fit_marker_dof_index[i];
            for (int d = 0; d < grad_dim; d++)
            {
               (*surf_fit_grad)[dof_index*grad_dim + d] =
                  surf_fit_grad_int(i*grad_dim + d);
            }
         }
      }

      // Interpolate Hessians of the LS.
      surf_fit_eval_hess->ComputeAtGivenPositions(new_x_sorted,
                                                  surf_fit_hess_int,
                                                  new_x_ordering);
      // Assumes surf_fit_hess and surf_fit_gf share the same space
      const int hess_dim = surf_fit_hess->VectorDim();
      const int hess_cnt = surf_fit_hess->Size()/hess_dim;
      if (surf_fit_hess->FESpace()->GetOrdering() == Ordering::byNODES)
      {
         for (int d = 0; d < hess_dim; d++)
         {
            for (int i = 0; i < cnt; i++)
            {
               int dof_index = surf_fit_marker_dof_index[i];
               (*surf_fit_hess)[dof_index + d*hess_cnt] =
                  surf_fit_hess_int(i + d*cnt);
            }
         }
      }
      else
      {
         for (int i = 0; i < cnt; i++)
         {
            int dof_index = surf_fit_marker_dof_index[i];
            for (int d = 0; d < hess_dim; d++)
            {
               (*surf_fit_hess)[dof_index*hess_dim + d] =
                  surf_fit_hess_int(i*hess_dim + d);
            }
         }
      }

   }
   else
   {
      surf_fit_eval->ComputeAtNewPosition(new_x, *surf_fit_gf, new_x_ordering);
      if (surf_fit_eval_grad)
      {
         surf_fit_eval_grad->ComputeAtNewPosition(new_x, *surf_fit_grad,
                                                  new_x_ordering);
      }
      if (surf_fit_eval_hess)
      {
         surf_fit_eval_hess->ComputeAtNewPosition(new_x, *surf_fit_hess,
                                                  new_x_ordering);
      }
   }
}

void TMOP_Integrator::
UpdateAfterMeshPositionChange(const Vector &d, const FiniteElementSpace &d_fes)
{
   if (discr_tc) { PA.Jtr_needs_update = true; }

   if (PA.enabled) { UpdateCoefficientsPA(d); }

   Ordering::Type ordering = d_fes.GetOrdering();

   // Update the finite difference delta if FD are used.
   if (fdflag) { ComputeFDh(d, d_fes); }

   Vector x_loc;
   if (periodic)
   {
      GetPeriodicPositions(*x_0, d, *x_0->FESpace(), d_fes, x_loc);
   }
   else
   {
      x_loc.SetSize(x_0->Size());
      add(*x_0, d, x_loc);
   }

   // Update the target constructor if it's a discrete one.
   if (discr_tc)
   {
      discr_tc->UpdateTargetSpecification(x_loc, true, ordering);
      if (fdflag)
      {
         if (periodic) { MFEM_ABORT("Periodic not implemented yet."); }
         discr_tc->UpdateGradientTargetSpecification(x_loc, fd_h, true, ordering);
         discr_tc->UpdateHessianTargetSpecification(x_loc, fd_h, true, ordering);
      }
   }

   // Update adapt_lim_gf if adaptive limiting is enabled.
   if (adapt_lim_gf)
   {
      adapt_lim_eval->ComputeAtNewPosition(x_loc, *adapt_lim_gf, ordering);
   }

   // Update surf_fit_gf (and optionally its gradients) if surface
   // fitting is enabled.
   if (surf_fit_gf)
   {
      RemapSurfaceFittingLevelSetAtNodes(x_loc, ordering);
   }
}

void TMOP_Integrator::ComputeFDh(const Vector &d, const FiniteElementSpace &fes)
{
   if (periodic) { MFEM_ABORT("Periodic not implemented yet."); }

   Vector x_loc(*x_0);
   x_loc += d;

   if (!fdflag) { return; }
   ComputeMinJac(x_loc, fes);
#ifdef MFEM_USE_MPI
   const ParFiniteElementSpace *pfes =
      dynamic_cast<const ParFiniteElementSpace *>(&fes);
   if (pfes)
   {
      real_t min_jac_all;
      MPI_Allreduce(&fd_h, &min_jac_all, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_MIN, pfes->GetComm());
      fd_h = min_jac_all;
   }
#endif
}

void TMOP_Integrator::EnableFiniteDifferences(const GridFunction &x)
{
   fdflag = true;
   const FiniteElementSpace *fes = x.FESpace();

   const bool per = fes->IsDGSpace();
   MFEM_VERIFY(per == false, "FD is not supported for periodic meshes.");

   if (discr_tc)
   {
#ifdef MFEM_USE_GSLIB
      const AdaptivityEvaluator *ae = discr_tc->GetAdaptivityEvaluator();
      if (dynamic_cast<const InterpolatorFP *>(ae))
      {
         MFEM_ABORT("Using GSLIB-based interpolation with finite differences"
                    "requires careful consideration. Contact TMOP team.");
      }
#endif
      discr_tc->UpdateTargetSpecification(x, false, fes->GetOrdering());
      discr_tc->UpdateGradientTargetSpecification(x, fd_h, false,
                                                  fes->GetOrdering());
      discr_tc->UpdateHessianTargetSpecification(x, fd_h, false,
                                                 fes->GetOrdering());
   }
}

#ifdef MFEM_USE_MPI
void TMOP_Integrator::EnableFiniteDifferences(const ParGridFunction &x)
{
   fdflag = true;
   const ParFiniteElementSpace *pfes = x.ParFESpace();

   const bool per = pfes->IsDGSpace();
   MFEM_VERIFY(per == false, "FD is not supported for periodic meshes.");

   if (discr_tc)
   {
#ifdef MFEM_USE_GSLIB
      const AdaptivityEvaluator *ae = discr_tc->GetAdaptivityEvaluator();
      if (dynamic_cast<const InterpolatorFP *>(ae))
      {
         MFEM_ABORT("Using GSLIB-based interpolation with finite differences"
                    "requires careful consideration. Contact TMOP team.");
      }
#endif
      discr_tc->UpdateTargetSpecification(x, false, pfes->GetOrdering());
      discr_tc->UpdateGradientTargetSpecification(x, fd_h, false,
                                                  pfes->GetOrdering());
      discr_tc->UpdateHessianTargetSpecification(x, fd_h, false,
                                                 pfes->GetOrdering());
   }
}
#endif

real_t TMOP_Integrator::ComputeMinDetT(const Vector &x,
                                       const FiniteElementSpace &fes)
{
   real_t min_detT = std::numeric_limits<real_t>::infinity();
   const int NE = fes.GetMesh()->GetNE();
   const int dim = fes.GetMesh()->Dimension();
   Array<int> xdofs;
   Jpr.SetSize(dim);
   Jpt.SetSize(dim);
   Jrt.SetSize(dim);

   for (int i = 0; i < NE; i++)
   {
      const FiniteElement *fe = fes.GetFE(i);
      const IntegrationRule &ir = EnergyIntegrationRule(*fe);
      const int dof = fe->GetDof(), nsp = ir.GetNPoints();

      DSh.SetSize(dof, dim);
      Vector posV(dof * dim);
      PMatI.UseExternalData(posV.GetData(), dof, dim);

      fes.GetElementVDofs(i, xdofs);
      x.GetSubVector(xdofs, posV);

      DenseTensor Jtr(dim, dim, ir.GetNPoints());
      targetC->ComputeElementTargets(i, *fe, ir, posV, Jtr);

      for (int q = 0; q < nsp; q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         const DenseMatrix &Jtr_q = Jtr(q);
         CalcInverse(Jtr_q, Jrt);
         fe->CalcDShape(ip, DSh);
         MultAtB(PMatI, DSh, Jpr);
         Mult(Jpr, Jrt, Jpt);
         real_t detT = Jpt.Det();
         min_detT = std::min(min_detT, detT);
      }
   }
   return min_detT;
}

real_t TMOP_Integrator::
ComputeUntanglerMaxMuBarrier(const Vector &x, const FiniteElementSpace &fes)
{
   real_t max_muT = -std::numeric_limits<real_t>::infinity();
   const int NE = fes.GetMesh()->GetNE();
   const int dim = fes.GetMesh()->Dimension();
   Array<int> xdofs;
   Jpr.SetSize(dim);
   Jpt.SetSize(dim);
   Jrt.SetSize(dim);

   TMOP_WorstCaseUntangleOptimizer_Metric *wcuo =
      dynamic_cast<TMOP_WorstCaseUntangleOptimizer_Metric *>(metric);

   if (!wcuo || wcuo->GetWorstCaseType() !=
       TMOP_WorstCaseUntangleOptimizer_Metric::WorstCaseType::Beta)
   {
      return 0.0;
   }

   for (int i = 0; i < NE; i++)
   {
      const FiniteElement *fe = fes.GetFE(i);
      const IntegrationRule &ir = EnergyIntegrationRule(*fe);
      const int dof = fe->GetDof(), nsp = ir.GetNPoints();
      Jpr.SetSize(dim);
      Jrt.SetSize(dim);
      Jpt.SetSize(dim);

      DSh.SetSize(dof, dim);
      Vector posV(dof * dim);
      PMatI.UseExternalData(posV.GetData(), dof, dim);

      fes.GetElementVDofs(i, xdofs);
      x.GetSubVector(xdofs, posV);

      DenseTensor Jtr(dim, dim, ir.GetNPoints());
      targetC->ComputeElementTargets(i, *fe, ir, posV, Jtr);

      for (int q = 0; q < nsp; q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         const DenseMatrix &Jtr_q = Jtr(q);
         CalcInverse(Jtr_q, Jrt);

         fe->CalcDShape(ip, DSh);
         MultAtB(PMatI, DSh, Jpr);
         Mult(Jpr, Jrt, Jpt);

         real_t metric_val = 0.0;
         if (wcuo)
         {
            wcuo->SetTargetJacobian(Jtr_q);
            metric_val = wcuo->EvalWBarrier(Jpt);
         }

         max_muT = std::max(max_muT, metric_val);
      }
   }
   return max_muT;
}

void TMOP_Integrator::
ComputeUntangleMetricQuantiles(const Vector &d, const FiniteElementSpace &fes)
{
   TMOP_WorstCaseUntangleOptimizer_Metric *wcuo =
      dynamic_cast<TMOP_WorstCaseUntangleOptimizer_Metric *>(metric);

   if (!wcuo) { return; }

   if (periodic) { MFEM_ABORT("Periodic not implemented yet."); }

   Vector x_loc(d.Size());
   if (x_0)
   {
      add(*x_0, d, x_loc);
   }
   else { x_loc = d; }

#ifdef MFEM_USE_MPI
   const ParFiniteElementSpace *pfes =
      dynamic_cast<const ParFiniteElementSpace *>(&fes);
#endif

   if (wcuo && wcuo->GetBarrierType() ==
       TMOP_WorstCaseUntangleOptimizer_Metric::BarrierType::Shifted)
   {
      real_t min_detT = ComputeMinDetT(x_loc, fes);
      real_t min_detT_all = min_detT;
#ifdef MFEM_USE_MPI
      if (pfes)
      {
         MPI_Allreduce(&min_detT, &min_detT_all, 1,
                       MPITypeMap<real_t>::mpi_type, MPI_MIN, pfes->GetComm());
      }
#endif
      if (wcuo) { wcuo->SetMinDetT(min_detT_all); }
   }

   real_t max_muT = ComputeUntanglerMaxMuBarrier(x_loc, fes);
   real_t max_muT_all = max_muT;
#ifdef MFEM_USE_MPI
   if (pfes)
   {
      MPI_Allreduce(&max_muT, &max_muT_all, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_MAX, pfes->GetComm());
   }
#endif
   wcuo->SetMaxMuT(max_muT_all);
}

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

real_t TMOPComboIntegrator::GetElementEnergy(const FiniteElement &el,
                                             ElementTransformation &T,
                                             const Vector &elfun)
{
   real_t energy= 0.0;
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

real_t TMOPComboIntegrator::GetRefinementElementEnergy(const FiniteElement &el,
                                                       ElementTransformation &T,
                                                       const Vector &elfun,
                                                       const IntegrationRule &irule)
{
   real_t energy= 0.0;
   for (int i = 0; i < tmopi.Size(); i++)
   {
      energy += tmopi[i]->GetRefinementElementEnergy(el, T, elfun, irule);
   }
   return energy;
}

real_t TMOPComboIntegrator::GetDerefinementElementEnergy(
   const FiniteElement &el,
   ElementTransformation &T,
   const Vector &elfun)
{
   real_t energy= 0.0;
   for (int i = 0; i < tmopi.Size(); i++)
   {
      energy += tmopi[i]->GetDerefinementElementEnergy(el, T, elfun);
   }
   return energy;
}

void TMOPComboIntegrator::EnableNormalization(const GridFunction &x)
{
   const int cnt = tmopi.Size();
   real_t total_integral = 0.0;
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
   real_t total_integral = 0.0;
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

void TMOPComboIntegrator::AssemblePA(const FiniteElementSpace &fes)
{
   for (int i = 0; i < tmopi.Size(); i++)
   {
      tmopi[i]->AssemblePA(fes);
   }
}

void TMOPComboIntegrator::AssembleGradPA(const Vector &xe,
                                         const FiniteElementSpace &fes)
{
   for (int i = 0; i < tmopi.Size(); i++)
   {
      tmopi[i]->AssembleGradPA(xe,fes);
   }
}

void TMOPComboIntegrator::AssembleGradDiagonalPA(Vector &de) const
{
   for (int i = 0; i < tmopi.Size(); i++)
   {
      tmopi[i]->AssembleGradDiagonalPA(de);
   }
}

void TMOPComboIntegrator::AddMultPA(const Vector &xe, Vector &ye) const
{
   for (int i = 0; i < tmopi.Size(); i++)
   {
      tmopi[i]->AddMultPA(xe, ye);
   }
}

void TMOPComboIntegrator::AddMultGradPA(const Vector &re, Vector &ce) const
{
   for (int i = 0; i < tmopi.Size(); i++)
   {
      tmopi[i]->AddMultGradPA(re, ce);
   }
}

real_t TMOPComboIntegrator::GetLocalStateEnergyPA(const Vector &xe) const
{
   real_t energy = 0.0;
   for (int i = 0; i < tmopi.Size(); i++)
   {
      energy += tmopi[i]->GetLocalStateEnergyPA(xe);
   }
   return energy;
}

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
