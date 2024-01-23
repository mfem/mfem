////////////////////////////////////////////////////////////////////////////////
// Copyright © 2018 Jérémie Dumas
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
////////////////////////////////////////////////////////////////////////////////
//
// BETA VERSION  0.99
//
// MMA solver using a dual interior point method
//
// Original code by Niels Aage, February 2013
// Modified to use OpenMP by Jun Wu, April 2017
// Various modifications by Jérémie Dumas, June 2017
//
// The class solves a general non-linear programming problem
// on standard from, i.e. non-linear objective f, m non-linear
// inequality constraints g and box constraints on the n
// design variables xmin, xmax.
//
//        min_x^n f(x)
//        s.t. g_j(x) < 0,   j = 1,m
//        xmin < x_i < xmax, i = 1,n
//
// Each call to Update() sets up and solve the following
// convex subproblem:
//
//   min_x     sum(p0j./(U-x)+q0j./(x-L)) + a0*z + sum(c.*y + 0.5*d.*y.^2)
//
//   s.t.      sum(pij./(U-x)+qij./(x-L)) - ai*z - yi <= bi, i = 1,m
//             Lj < alphaj <=  xj <= betaj < Uj,  j = 1,n
//             yi >= 0, i = 1,m
//             z >= 0.
//
// NOTE: a0 == 1 in this implementation !!!!
//
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include "mfem.hpp"

class MMASolver
{

public:
   MMASolver(int n, int m, double a = 0.0, double c = 1000.0, double d = 0.0);

   void SetAsymptotes(double init, double decrease, double increase);

   void ConstraintModification(bool conMod) {}

   void Update(double *xval, const double *dfdx, const double *gx,
               const double *dgdx, const double *xmin,
               const double *xmax);

   void Reset() { iter = 0; };

private:
   int n, m, iter;

   const double xmamieps;
   const double epsimin;

   const double raa0;
   const double move, albefa;
   double asyminit, asymdec, asyminc;

   std::vector<double> a, c, d;
   std::vector<double> y;
   double z;

   std::vector<double> lam, mu, s;
   std::vector<double> low, upp, alpha, beta, p0, q0, pij, qij, b, grad, hess;

   std::vector<double> xold1, xold2;

   void GenSub(const double *xval, const double *dfdx, const double *gx,
               const double *dgdx, const double *xmin,
               const double *xmax);

   void SolveDSA(double *x);
   void SolveDIP(double *x);

   void XYZofLAMBDA(double *x);

   void DualGrad(double *x);
   void DualHess(double *x);
   void DualLineSearch();
   double DualResidual(double *x, double epsi);

   static void Factorize(double *K, int n);
   static void Solve(double *K, double *x, int n);
};


////////////////////////////////////////////////////////////////////////////////
// PUBLIC
////////////////////////////////////////////////////////////////////////////////

MMASolver::MMASolver(int n, int m, double ai, double ci, double di)
   : n(n)
   , m(m)
   , iter(0)
   , xmamieps(1.0e-5)
     //, epsimin(1e-7)
   , epsimin(std::sqrt(n + m) * 1e-9)
   , raa0(0.00001)
   , move(0.5)
   , albefa(0.1)
   , asyminit(0.5) // 0.2;
   , asymdec(0.7) // 0.65;
   , asyminc(1.2) // 1.08;
   , a(m, ai)
   , c(m, ci)
   , d(m, di)
   , y(m)
   , lam(m)
   , mu(m), s(2 * m)
   , low(n)
   , upp(n)
   , alpha(n)
   , beta(n)
   , p0(n)
   , q0(n)
   , pij(n * m)
   , qij(n * m), b(m)
   , grad(m)
   , hess(m * m)
   , xold1(n)
   , xold2(n)
{ }

void MMASolver::SetAsymptotes(double init, double decrease, double increase)
{

   // asymptotes initialization and increase/decrease
   asyminit = init;
   asymdec = decrease;
   asyminc = increase;
}

void MMASolver::Update(double *xval, const double *dfdx, const double *gx,
                       const double *dgdx,
                       const double *xmin, const double *xmax)
{
   // Generate the subproblem
   GenSub(xval, dfdx, gx, dgdx, xmin, xmax);

   // Update xolds
   xold2 = xold1;
   std::copy_n(xval, n, xold1.data());

   // Solve the dual with an interior point method
   SolveDIP(xval);

   // Solve the dual with a steepest ascent method
   // SolveDSA(xval);
}

////////////////////////////////////////////////////////////////////////////////
// PRIVATE
////////////////////////////////////////////////////////////////////////////////

void MMASolver::SolveDIP(double *x)
{

   for (int j = 0; j < m; j++)
   {
      lam[j] = c[j] / 2.0;
      mu[j] = 1.0;
   }

   const double tol = epsimin; // 1.0e-9*sqrt(m+n);
   double epsi = 1.0;
   double err = 1.0;
   int loop;

   while (epsi > tol)
   {

      loop = 0;
      while (err > 0.9 * epsi && loop < 100)
      {
         loop++;

         // Set up Newton system
         XYZofLAMBDA(x);
         DualGrad(x);
         for (int j = 0; j < m; j++)
         {
            grad[j] = -1.0 * grad[j] - epsi / lam[j];
         }
         DualHess(x);

         // Solve Newton system
         if (m > 1)
         {
            Factorize(hess.data(), m);
            Solve(hess.data(), grad.data(), m);
            for (int j = 0; j < m; j++)
            {
               s[j] = grad[j];
            }
         }
         else if (m > 0)
         {
            s[0] = grad[0] / hess[0];
         }

         // Get the full search direction
         for (int i = 0; i < m; i++)
         {
            s[m + i] = -mu[i] + epsi / lam[i] - s[i] * mu[i] / lam[i];
         }

         // Perform linesearch and update lam and mu
         DualLineSearch();

         XYZofLAMBDA(x);

         // Compute KKT res
         err = DualResidual(x, epsi);
      }
      epsi = epsi * 0.1;
   }
}

void MMASolver::SolveDSA(double *x)
{

   for (int j = 0; j < m; j++)
   {
      lam[j] = 1.0;
   }

   const double tol = epsimin; // 1.0e-9*sqrt(m+n);
   double err = 1.0;
   int loop = 0;

   while (err > tol && loop < 500)
   {
      loop++;
      XYZofLAMBDA(x);
      DualGrad(x);
      double theta = 1.0;
      err = 0.0;
      for (int j = 0; j < m; j++)
      {
         lam[j] = std::max(0.0, lam[j] + theta * grad[j]);
         err += grad[j] * grad[j];
      }
      err = std::sqrt(err);
   }
}

double MMASolver::DualResidual(double *x, double epsi)
{

   double *res = new double[2 * m];

   for (int j = 0; j < m; j++)
   {
      res[j] = -b[j] - a[j] * z - y[j] + mu[j];
      res[j + m] = mu[j] * lam[j] - epsi;
      for (int i = 0; i < n; i++)
      {
         res[j] += pij[i * m + j] / (upp[i] - x[i]) + qij[i * m + j] / (x[i] - low[i]);
      }
   }

   double nrI = 0.0;
   for (int i = 0; i < 2 * m; i++)
   {
      if (nrI < std::abs(res[i]))
      {
         nrI = std::abs(res[i]);
      }
   }

   delete[] res;

   return nrI;
}

void MMASolver::DualLineSearch()
{

   double theta = 1.005;
   for (int i = 0; i < m; i++)
   {
      if (theta < -1.01 * s[i] / lam[i])
      {
         theta = -1.01 * s[i] / lam[i];
      }
      if (theta < -1.01 * s[i + m] / mu[i])
      {
         theta = -1.01 * s[i + m] / mu[i];
      }
   }
   theta = 1.0 / theta;

   for (int i = 0; i < m; i++)
   {
      lam[i] = lam[i] + theta * s[i];
      mu[i] = mu[i] + theta * s[i + m];
   }
}

void MMASolver::DualHess(double *x)
{

   double *df2 = new double[n];
   double *PQ = new double[n * m];
#ifdef MMA_WITH_OPENMP
   #pragma omp parallel for
#endif
   for (int i = 0; i < n; i++)
   {
      double pjlam = p0[i];
      double qjlam = q0[i];
      for (int j = 0; j < m; j++)
      {
         pjlam += pij[i * m + j] * lam[j];
         qjlam += qij[i * m + j] * lam[j];
         PQ[i * m + j] = pij[i * m + j] / pow(upp[i] - x[i],
                                              2.0) - qij[i * m + j] / pow(x[i] - low[i], 2.0);
      }
      df2[i] = -1.0 / (2.0 * pjlam / pow(upp[i] - x[i],
                                         3.0) + 2.0 * qjlam / pow(x[i] - low[i], 3.0));
      double xp = (sqrt(pjlam) * low[i] + sqrt(qjlam) * upp[i]) / (sqrt(pjlam) + sqrt(
                                                                      qjlam));
      if (xp < alpha[i])
      {
         df2[i] = 0.0;
      }
      if (xp > beta[i])
      {
         df2[i] = 0.0;
      }
   }

   // Create the matrix/matrix/matrix product: PQ^T * diag(df2) * PQ
   double *tmp = new double[n * m];
   for (int j = 0; j < m; j++)
   {
#ifdef MMA_WITH_OPENMP
      #pragma omp parallel for
#endif
      for (int i = 0; i < n; i++)
      {
         tmp[j * n + i] = 0.0;
         tmp[j * n + i] += PQ[i * m + j] * df2[i];
      }
   }

   for (int i = 0; i < m; i++)
   {
      for (int j = 0; j < m; j++)
      {
         hess[i * m + j] = 0.0;
         for (int k = 0; k < n; k++)
         {
            hess[i * m + j] += tmp[i * n + k] * PQ[k * m + j];
         }
      }
   }

   double lamai = 0.0;
   for (int j = 0; j < m; j++)
   {
      if (lam[j] < 0.0)
      {
         lam[j] = 0.0;
      }
      lamai += lam[j] * a[j];
      if (lam[j] > c[j])
      {
         hess[j * m + j] += -1.0;
      }
      hess[j * m + j] += -mu[j] / lam[j];
   }

   if (lamai > 0.0)
   {
      for (int j = 0; j < m; j++)
      {
         for (int k = 0; k < m; k++)
         {
            hess[j * m + k] += -10.0 * a[j] * a[k];
         }
      }
   }

   // pos def check
   double HessTrace = 0.0;
   for (int i = 0; i < m; i++)
   {
      HessTrace += hess[i * m + i];
   }
   double HessCorr = 1e-4 * HessTrace / m;

   if (-1.0 * HessCorr < 1.0e-7)
   {
      HessCorr = -1.0e-7;
   }

   for (int i = 0; i < m; i++)
   {
      hess[i * m + i] += HessCorr;
   }

   delete[] df2;
   delete[] PQ;
   delete[] tmp;
}

void MMASolver::DualGrad(double *x)
{
   for (int j = 0; j < m; j++)
   {
      grad[j] = -b[j] - a[j] * z - y[j];
      for (int i = 0; i < n; i++)
      {
         grad[j] += pij[i * m + j] / (upp[i] - x[i]) + qij[i * m + j] / (x[i] - low[i]);
      }
   }
}

void MMASolver::XYZofLAMBDA(double *x)
{

   double lamai = 0.0;
   for (int i = 0; i < m; i++)
   {
      if (lam[i] < 0.0)
      {
         lam[i] = 0;
      }
      y[i] = std::max(0.0, lam[i] -
                      c[i]); // Note y=(lam-c)/d - however d is fixed at one !!
      lamai += lam[i] * a[i];
   }
   z = std::max(0.0, 10.0 * (lamai - 1.0)); // SINCE a0 = 1.0

#ifdef MMA_WITH_OPENMP
   #pragma omp parallel for
#endif
   for (int i = 0; i < n; i++)
   {
      double pjlam = p0[i];
      double qjlam = q0[i];
      for (int j = 0; j < m; j++)
      {
         pjlam += pij[i * m + j] * lam[j];
         qjlam += qij[i * m + j] * lam[j];
      }
      x[i] = (sqrt(pjlam) * low[i] + sqrt(qjlam) * upp[i]) / (sqrt(pjlam) + sqrt(
                                                                 qjlam));
      if (x[i] < alpha[i])
      {
         x[i] = alpha[i];
      }
      if (x[i] > beta[i])
      {
         x[i] = beta[i];
      }
   }
}

void MMASolver::GenSub(const double *xval, const double *dfdx, const double *gx,
                       const double *dgdx, const double *xmin,
                       const double *xmax)
{
   // Forward the iterator
   iter++;

   // Set asymptotes
   if (iter < 3)
   {
#ifdef MMA_WITH_OPENMP
      #pragma omp parallel for
#endif
      for (int i = 0; i < n; i++)
      {
         low[i] = xval[i] - asyminit * (xmax[i] - xmin[i]);
         upp[i] = xval[i] + asyminit * (xmax[i] - xmin[i]);
      }
   }
   else
   {
#ifdef MMA_WITH_OPENMP
      #pragma omp parallel for
#endif
      for (int i = 0; i < n; i++)
      {
         double zzz = (xval[i] - xold1[i]) * (xold1[i] - xold2[i]);
         double gamma;
         if (zzz < 0.0)
         {
            gamma = asymdec;
         }
         else if (zzz > 0.0)
         {
            gamma = asyminc;
         }
         else
         {
            gamma = 1.0;
         }
         low[i] = xval[i] - gamma * (xold1[i] - low[i]);
         upp[i] = xval[i] + gamma * (upp[i] - xold1[i]);

         double xmami = std::max(xmamieps, xmax[i] - xmin[i]);
         // double xmami = xmax[i] - xmin[i];
         low[i] = std::max(low[i], xval[i] - 100.0 * xmami);
         low[i] = std::min(low[i], xval[i] - 1.0e-5 * xmami);
         upp[i] = std::max(upp[i], xval[i] + 1.0e-5 * xmami);
         upp[i] = std::min(upp[i], xval[i] + 100.0 * xmami);

         double xmi = xmin[i] - 1.0e-6;
         double xma = xmax[i] + 1.0e-6;
         if (xval[i] < xmi)
         {
            low[i] = xval[i] - (xma - xval[i]) / 0.9;
            upp[i] = xval[i] + (xma - xval[i]) / 0.9;
         }
         if (xval[i] > xma)
         {
            low[i] = xval[i] - (xval[i] - xmi) / 0.9;
            upp[i] = xval[i] + (xval[i] - xmi) / 0.9;
         }
      }
   }

   // Set bounds and the coefficients for the approximation
   // double raa0 = 0.5*1e-6;
#ifdef MMA_WITH_OPENMP
   #pragma omp parallel for
#endif
   for (int i = 0; i < n; ++i)
   {
      // Compute bounds alpha and beta
      alpha[i] = std::max(xmin[i], low[i] + albefa * (xval[i] - low[i]));
      alpha[i] = std::max(alpha[i], xval[i] - move * (xmax[i] - xmin[i]));
      alpha[i] = std::min(alpha[i], xmax[i]);
      beta[i] = std::min(xmax[i], upp[i] - albefa * (upp[i] - xval[i]));
      beta[i] = std::min(beta[i], xval[i] + move * (xmax[i] - xmin[i]));
      beta[i] = std::max(beta[i], xmin[i]);

      // Objective function
      {
         double dfdxp = std::max(0.0, dfdx[i]);
         double dfdxm = std::max(0.0, -1.0 * dfdx[i]);
         double xmamiinv = 1.0 / std::max(xmamieps, xmax[i] - xmin[i]);
         double pq = 0.001 * std::abs(dfdx[i]) + raa0 * xmamiinv;
         p0[i] = std::pow(upp[i] - xval[i], 2.0) * (dfdxp + pq);
         q0[i] = std::pow(xval[i] - low[i], 2.0) * (dfdxm + pq);
      }

      // Constraints
      for (int j = 0; j < m; j++)
      {
         double dgdxp = std::max(0.0, dgdx[i * m + j]);
         double dgdxm = std::max(0.0, -1.0 * dgdx[i * m + j]);
         double xmamiinv = 1.0 / std::max(xmamieps, xmax[i] - xmin[i]);
         double pq = 0.001 * std::abs(dgdx[i * m + j]) + raa0 * xmamiinv;
         pij[i * m + j] = std::pow(upp[i] - xval[i], 2.0) * (dgdxp + pq);
         qij[i * m + j] = std::pow(xval[i] - low[i], 2.0) * (dgdxm + pq);
      }
   }

   // The constant for the constraints
   for (int j = 0; j < m; j++)
   {
      b[j] = -gx[j];
      for (int i = 0; i < n; i++)
      {
         b[j] += pij[i * m + j] / (upp[i] - xval[i]) + qij[i * m + j] /
                 (xval[i] - low[i]);
      }
   }
}

void MMASolver::Factorize(double *K, int n)
{

   for (int s = 0; s < n - 1; s++)
   {
      for (int i = s + 1; i < n; i++)
      {
         K[i * n + s] = K[i * n + s] / K[s * n + s];
         for (int j = s + 1; j < n; j++)
         {
            K[i * n + j] = K[i * n + j] - K[i * n + s] * K[s * n + j];
         }
      }
   }
}

void MMASolver::Solve(double *K, double *x, int n)
{

   for (int i = 1; i < n; i++)
   {
      double a = 0.0;
      for (int j = 0; j < i; j++)
      {
         a = a - K[i * n + j] * x[j];
      }
      x[i] = x[i] + a;
   }

   x[n - 1] = x[n - 1] / K[(n - 1) * n + (n - 1)];
   for (int i = n - 2; i >= 0; i--)
   {
      double a = x[i];
      for (int j = i + 1; j < n; j++)
      {
         a = a - K[i * n + j] * x[j];
      }
      x[i] = a / K[i * n + i];
   }
}

// Wrapper class for MMASolver to mfem
namespace mfem
{
class MMA
{
public:
   MMA(int n, int m, double a = 0.0, double c = 1000.0,
       double d = 0.0):mma(new MMASolver(n, m, a, c, d)) {}

   void SetAsymptotes(double init, double decrease, double increase)
   {
      mma->SetAsymptotes(init, decrease, increase);
   }

   void ConstraintModification(bool conMod)
   {
      mma->ConstraintModification(conMod);
   }

   void Update(Vector &x, const Vector &dfdx, const double &gx,
               const Vector &dgdx, const Vector &xmin,
               const Vector &xmax)
   {
      mma->Update(x.GetData(), dfdx.GetData(), &gx, dgdx.GetData(),
                  xmin.GetData(), xmax.GetData());
   }

   void Update(Vector &x, const Vector &dfdx, const Vector &gx,
               const DenseMatrix &dgdx, const Vector &xmin,
               const Vector &xmax)
   {
      mma->Update(x.GetData(), dfdx.GetData(), gx.GetData(), dgdx.GetData(),
                  xmin.GetData(), xmax.GetData());
   }

   void Reset() { mma->Reset(); };
private:
   std::unique_ptr<MMASolver> mma;
};
}
