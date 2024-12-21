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

// Implementation of IntegrationRule(s) classes

// Acknowledgment: Some of the high-precision triangular and tetrahedral
// quadrature rules below were obtained from the Encyclopaedia of Cubature
// Formulas at http://nines.cs.kuleuven.be/research/ecf/ecf.html

#include "fem.hpp"
#include "../mesh/nurbs.hpp"
#include <cmath>

#ifdef MFEM_USE_MPFR
#include <mpfr.h>
#endif

using namespace std;

namespace mfem
{

IntegrationRule::IntegrationRule(IntegrationRule &irx, IntegrationRule &iry)
{
   int i, j, nx, ny;

   nx = irx.GetNPoints();
   ny = iry.GetNPoints();
   SetSize(nx * ny);
   SetPointIndices();
   Order = std::min(irx.GetOrder(), iry.GetOrder());

   for (j = 0; j < ny; j++)
   {
      IntegrationPoint &ipy = iry.IntPoint(j);
      for (i = 0; i < nx; i++)
      {
         IntegrationPoint &ipx = irx.IntPoint(i);
         IntegrationPoint &ip  = IntPoint(j*nx+i);

         ip.x = ipx.x;
         ip.y = ipy.x;
         ip.weight = ipx.weight * ipy.weight;
      }
   }
}

IntegrationRule::IntegrationRule(IntegrationRule &irx, IntegrationRule &iry,
                                 IntegrationRule &irz)
{
   const int nx = irx.GetNPoints();
   const int ny = iry.GetNPoints();
   const int nz = irz.GetNPoints();
   SetSize(nx*ny*nz);
   SetPointIndices();
   Order = std::min({irx.GetOrder(), iry.GetOrder(), irz.GetOrder()});

   for (int iz = 0; iz < nz; ++iz)
   {
      IntegrationPoint &ipz = irz.IntPoint(iz);
      for (int iy = 0; iy < ny; ++iy)
      {
         IntegrationPoint &ipy = iry.IntPoint(iy);
         for (int ix = 0; ix < nx; ++ix)
         {
            IntegrationPoint &ipx = irx.IntPoint(ix);
            IntegrationPoint &ip  = IntPoint(iz*nx*ny + iy*nx + ix);

            ip.x = ipx.x;
            ip.y = ipy.x;
            ip.z = ipz.x;
            ip.weight = ipx.weight*ipy.weight*ipz.weight;
         }
      }
   }
}

const Array<real_t> &IntegrationRule::GetWeights() const
{
   if (weights.Size() != GetNPoints())
   {
      weights.SetSize(GetNPoints());
      for (int i = 0; i < GetNPoints(); i++)
      {
         weights[i] = IntPoint(i).weight;
      }
   }
   return weights;
}

void IntegrationRule::SetPointIndices()
{
   for (int i = 0; i < Size(); i++)
   {
      IntPoint(i).index = i;
   }
}

void IntegrationRule::GrundmannMollerSimplexRule(int s, int n)
{
   // for pow on older compilers
   using std::pow;
   const int d = 2*s + 1;
   Vector fact(d + n + 1);
   Array<int> beta(n), sums(n);

   fact(0) = 1.;
   for (int i = 1; i < fact.Size(); i++)
   {
      fact(i) = fact(i - 1)*i;
   }

   // number of points is \binom{n + s + 1}{n + 1}
   int np = 1, f = 1;
   for (int i = 0; i <= n; i++)
   {
      np *= (s + i + 1), f *= (i + 1);
   }
   np /= f;
   SetSize(np);
   SetPointIndices();
   Order = 2*s + 1;

   int pt = 0;
   for (int i = 0; i <= s; i++)
   {
      real_t weight;

      weight = pow(2., -2*s)*pow(static_cast<real_t>(d + n - 2*i),
                                 d)/fact(i)/fact(d + n - i);
      if (i%2)
      {
         weight = -weight;
      }

      // loop over all beta : beta_0 + ... + beta_{n-1} <= s - i
      int k = s - i;
      beta = 0;
      sums = 0;
      while (true)
      {
         IntegrationPoint &ip = IntPoint(pt++);
         ip.weight = weight;
         ip.x = real_t(2*beta[0] + 1)/(d + n - 2*i);
         ip.y = real_t(2*beta[1] + 1)/(d + n - 2*i);
         if (n == 3)
         {
            ip.z = real_t(2*beta[2] + 1)/(d + n - 2*i);
         }

         int j = 0;
         while (sums[j] == k)
         {
            beta[j++] = 0;
            if (j == n)
            {
               goto done_beta;
            }
         }
         beta[j]++;
         sums[j]++;
         for (j--; j >= 0; j--)
         {
            sums[j] = sums[j+1];
         }
      }
   done_beta:
      ;
   }
}

IntegrationRule*
IntegrationRule::ApplyToKnotIntervals(KnotVector const& kv) const
{
   const int np = this->GetNPoints();
   const int ne = kv.GetNE();

   IntegrationRule *kvir = new IntegrationRule(ne * np);
   kvir->SetOrder(GetOrder());

   real_t x0 = kv[0];
   real_t x1 = x0;

   int id = 0;
   for (int e=0; e<ne; ++e)
   {
      x0 = x1;

      if (e == ne-1)
      {
         x1 = kv[kv.Size() - 1];
      }
      else
      {
         // Find the next unique knot
         while (id < kv.Size() - 1)
         {
            id++;
            if (kv[id] != x0)
            {
               x1 = kv[id];
               break;
            }
         }
      }

      const real_t s = x1 - x0;

      for (int j=0; j<this->GetNPoints(); ++j)
      {
         const real_t x = x0 + (s * (*this)[j].x);
         (*kvir)[(e * np) + j].Set1w(x, (*this)[j].weight);
      }
   }

   return kvir;
}

#ifdef MFEM_USE_MPFR

// Class for computing hi-precision (HP) quadrature in 1D
class HP_Quadrature1D
{
protected:
   mpfr_t pi, z, pp, p1, p2, p3, dz, w, rtol;

public:
   static const mpfr_rnd_t rnd = GMP_RNDN;
   static const int default_prec = 128;

   // prec = MPFR precision in bits
   HP_Quadrature1D(const int prec = default_prec)
   {
      mpfr_inits2(prec, pi, z, pp, p1, p2, p3, dz, w, rtol, (mpfr_ptr) 0);
      mpfr_const_pi(pi, rnd);
      mpfr_set_si_2exp(rtol, 1, -32, rnd); // 2^(-32) < 2.33e-10
   }

   // set rtol = 2^exponent
   // this is a tolerance for the last correction of x_i in Newton's algorithm;
   // this gives roughly rtol^2 accuracy for the final x_i.
   void SetRelTol(const int exponent = -32)
   {
      mpfr_set_si_2exp(rtol, 1, exponent, rnd);
   }

   // n - number of quadrature points
   // k - index of the point to compute, 0 <= k < n
   // see also: QuadratureFunctions1D::GaussLegendre
   void ComputeGaussLegendrePoint(const int n, const int k)
   {
      MFEM_ASSERT(n > 0 && 0 <= k && k < n, "invalid n = " << n
                  << " and/or k = " << k);

      int i = (k < (n+1)/2) ? k+1 : n-k;

      // Initial guess for the x-coordinate:
      // set z = cos(pi * (i - 0.25) / (n + 0.5)) =
      //       = sin(pi * ((n+1-2*i) / (2*n+1)))
      mpfr_set_si(z, n+1-2*i, rnd);
      mpfr_div_si(z, z, 2*n+1, rnd);
      mpfr_mul(z, z, pi, rnd);
      mpfr_sin(z, z, rnd);

      bool done = false;
      while (1)
      {
         mpfr_set_si(p2, 1, rnd);
         mpfr_set(p1, z, rnd);
         for (int j = 2; j <= n; j++)
         {
            mpfr_set(p3, p2, rnd);
            mpfr_set(p2, p1, rnd);
            // p1 = ((2 * j - 1) * z * p2 - (j - 1) * p3) / j;
            mpfr_mul_si(p1, z, 2*j-1, rnd);
            mpfr_mul_si(p3, p3, j-1, rnd);
            mpfr_fms(p1, p1, p2, p3, rnd);
            mpfr_div_si(p1, p1, j, rnd);
         }
         // p1 is Legendre polynomial

         // derivative of the Legendre polynomial:
         // pp = n * (z*p1-p2) / (z*z - 1);
         mpfr_fms(pp, z, p1, p2, rnd);
         mpfr_mul_si(pp, pp, n, rnd);
         mpfr_sqr(p2, z, rnd);
         mpfr_sub_si(p2, p2, 1, rnd);
         mpfr_div(pp, pp, p2, rnd);

         if (done) { break; }

         // set delta_z: dz = p1/pp;
         mpfr_div(dz, p1, pp, rnd);
         // compute absolute tolerance: atol = rtol*(1-z)
         mpfr_t &atol = w;
         mpfr_si_sub(atol, 1, z, rnd);
         mpfr_mul(atol, atol, rtol, rnd);
         if (mpfr_cmpabs(dz, atol) <= 0)
         {
            done = true;
            // continue the computation: get pp at the new point, then exit
         }
         // update z = z - dz
         mpfr_sub(z, z, dz, rnd);
      }

      // map z to (0,1): z = (1 - z)/2
      mpfr_si_sub(z, 1, z, rnd);
      mpfr_div_2si(z, z, 1, rnd);

      // weight: w = 1/(4*z*(1 - z)*pp*pp)
      mpfr_sqr(w, pp, rnd);
      mpfr_mul_2si(w, w, 2, rnd);
      mpfr_mul(w, w, z, rnd);
      mpfr_si_sub(p1, 1, z, rnd); // p1 = 1-z
      mpfr_mul(w, w, p1, rnd);
      mpfr_si_div(w, 1, w, rnd);

      if (k >= (n+1)/2) { mpfr_swap(z, p1); }
   }

   // n - number of quadrature points
   // k - index of the point to compute, 0 <= k < n
   // see also: QuadratureFunctions1D::GaussLobatto
   void ComputeGaussLobattoPoint(const int n, const int k)
   {
      MFEM_ASSERT(n > 1 && 0 <= k && k < n, "invalid n = " << n
                  << " and/or k = " << k);

      int i = (k < (n+1)/2) ? k : n-1-k;

      if (i == 0)
      {
         mpfr_set_si(z, 0, rnd);
         mpfr_set_si(p1, 1, rnd);
         mpfr_set_si(w, n*(n-1), rnd);
         mpfr_si_div(w, 1, w, rnd); // weight = 1/(n*(n-1))
         return;
      }
      // initial guess is the corresponding Chebyshev point, z:
      //    z = -cos(pi * i/(n-1)) = sin(pi * (2*i-n+1)/(2*n-2))
      mpfr_set_si(z, 2*i-n+1, rnd);
      mpfr_div_si(z, z, 2*(n-1), rnd);
      mpfr_mul(z, pi, z, rnd);
      mpfr_sin(z, z, rnd);
      bool done = false;
      for (int iter = 0 ; true ; ++iter)
      {
         // build Legendre polynomials, up to P_{n}(z)
         mpfr_set_si(p1, 1, rnd);
         mpfr_set(p2, z, rnd);

         for (int l = 1 ; l < (n-1) ; ++l)
         {
            // P_{l+1}(x) = [ (2*l+1)*x*P_l(x) - l*P_{l-1}(x) ]/(l+1)
            mpfr_mul_si(p1, p1, l, rnd);
            mpfr_mul_si(p3, z, 2*l+1, rnd);
            mpfr_fms(p3, p3, p2, p1, rnd);
            mpfr_div_si(p3, p3, l+1, rnd);

            mpfr_set(p1, p2, rnd);
            mpfr_set(p2, p3, rnd);
         }
         if (done) { break; }
         // compute dz = resid/deriv = (z*p2 - p1) / (n*p2);
         mpfr_fms(dz, z, p2, p1, rnd);
         mpfr_mul_si(p3, p2, n, rnd);
         mpfr_div(dz, dz, p3, rnd);
         // update: z = z - dz
         mpfr_sub(z, z, dz, rnd);
         // compute absolute tolerance: atol = rtol*(1 + z)
         mpfr_t &atol = w;
         mpfr_add_si(atol, z, 1, rnd);
         mpfr_mul(atol, atol, rtol, rnd);
         // check for convergence
         if (mpfr_cmpabs(dz, atol) <= 0)
         {
            done = true;
            // continue the computation: get p2 at the new point, then exit
         }
         // If the iteration does not converge fast, something is wrong.
         MFEM_VERIFY(iter < 8, "n = " << n << ", i = " << i
                     << ", dz = " << mpfr_get_d(dz, rnd));
      }
      // Map to the interval [0,1] and scale the weights
      mpfr_add_si(z, z, 1, rnd);
      mpfr_div_2si(z, z, 1, rnd);
      // set the symmetric point
      mpfr_si_sub(p1, 1, z, rnd);
      // w = 1/[ n*(n-1)*[P_{n-1}(z)]^2 ]
      mpfr_sqr(w, p2, rnd);
      mpfr_mul_si(w, w, n*(n-1), rnd);
      mpfr_si_div(w, 1, w, rnd);

      if (k >= (n+1)/2) { mpfr_swap(z, p1); }
   }

   real_t GetPoint() const { return mpfr_get_d(z, rnd); }
   real_t GetSymmPoint() const { return mpfr_get_d(p1, rnd); }
   real_t GetWeight() const { return mpfr_get_d(w, rnd); }

   const mpfr_t &GetHPPoint() const { return z; }
   const mpfr_t &GetHPSymmPoint() const { return p1; }
   const mpfr_t &GetHPWeight() const { return w; }

   ~HP_Quadrature1D()
   {
      mpfr_clears(pi, z, pp, p1, p2, p3, dz, w, rtol, (mpfr_ptr) 0);
      mpfr_free_cache();
   }
};

#endif // MFEM_USE_MPFR


void QuadratureFunctions1D::GaussLegendre(const int np, IntegrationRule* ir)
{
   ir->SetSize(np);
   ir->SetPointIndices();
   ir->SetOrder(2*np - 1);

   switch (np)
   {
      case 1:
         ir->IntPoint(0).Set1w(0.5, 1.0);
         return;
      case 2:
         ir->IntPoint(0).Set1w(0.21132486540518711775, 0.5);
         ir->IntPoint(1).Set1w(0.78867513459481288225, 0.5);
         return;
      case 3:
         ir->IntPoint(0).Set1w(0.11270166537925831148, 5./18.);
         ir->IntPoint(1).Set1w(0.5, 4./9.);
         ir->IntPoint(2).Set1w(0.88729833462074168852, 5./18.);
         return;
   }

   const int n = np;
   const int m = (n+1)/2;

#ifndef MFEM_USE_MPFR

   for (int i = 1; i <= m; i++)
   {
      real_t z = cos(M_PI * (i - 0.25) / (n + 0.5));
      real_t pp, p1, dz, xi = 0.;
      bool done = false;
      while (1)
      {
         real_t p2 = 1;
         p1 = z;
         for (int j = 2; j <= n; j++)
         {
            real_t p3 = p2;
            p2 = p1;
            p1 = ((2 * j - 1) * z * p2 - (j - 1) * p3) / j;
         }
         // p1 is Legendre polynomial

         pp = n * (z*p1-p2) / (z*z - 1);
         if (done) { break; }

         dz = p1/pp;
#ifdef MFEM_USE_SINGLE
         if (std::abs(dz) < 1e-7)
#elif defined MFEM_USE_DOUBLE
         if (std::abs(dz) < 1e-16)
#else
         MFEM_ABORT("Floating point type undefined");
         if (std::abs(dz) < 1e-16)
#endif
         {
            done = true;
            // map the new point (z-dz) to (0,1):
            xi = ((1 - z) + dz)/2; // (1 - (z - dz))/2 has bad round-off
            // continue the computation: get pp at the new point, then exit
         }
         // update: z = z - dz
         z -= dz;
      }

      ir->IntPoint(i-1).x = xi;
      ir->IntPoint(n-i).x = 1 - xi;
      ir->IntPoint(i-1).weight =
         ir->IntPoint(n-i).weight = 1./(4*xi*(1 - xi)*pp*pp);
   }

#else // MFEM_USE_MPFR is defined

   HP_Quadrature1D hp_quad;
   for (int i = 1; i <= m; i++)
   {
      hp_quad.ComputeGaussLegendrePoint(n, i-1);

      ir->IntPoint(i-1).x = hp_quad.GetPoint();
      ir->IntPoint(n-i).x = hp_quad.GetSymmPoint();
      ir->IntPoint(i-1).weight = ir->IntPoint(n-i).weight = hp_quad.GetWeight();
   }

#endif // MFEM_USE_MPFR

}

void QuadratureFunctions1D::GaussLobatto(const int np, IntegrationRule* ir)
{
   /* An np point Gauss-Lobatto quadrature has (np - 2) free abscissa the other
      (2) abscissa are the interval endpoints.

      The interior x_i are the zeros of P'_{np-1}(x). The weights of the
      interior points on the interval [-1,1] are:

      w_i = 2/(np*(np-1)*[P_{np-1}(x_i)]^2)

      The end point weights (on [-1,1]) are: w_{end} = 2/(np*(np-1)).

      The interior abscissa are found via a nonlinear solve, the initial guess
      for each point is the corresponding Chebyshev point.

      After we find all points on the interval [-1,1], we will map and scale the
      points and weights to the MFEM natural interval [0,1].

      References:
      [1] E. E. Lewis and W. F. Millier, "Computational Methods of Neutron
          Transport", Appendix A
      [2] the QUADRULE software by John Burkardt,
          https://people.sc.fsu.edu/~jburkardt/cpp_src/quadrule/quadrule.cpp
   */

   ir->SetSize(np);
   ir->SetPointIndices();
   if ( np == 1 )
   {
      ir->IntPoint(0).Set1w(0.5, 1.0);
      ir->SetOrder(1);
   }
   else
   {
      ir->SetOrder(2*np - 3);

#ifndef MFEM_USE_MPFR

      // endpoints and respective weights
      ir->IntPoint(0).x = 0.0;
      ir->IntPoint(np-1).x = 1.0;
      ir->IntPoint(0).weight = ir->IntPoint(np-1).weight = 1.0/(np*(np-1));

      // interior points and weights
      // use symmetry and compute just half of the points
      for (int i = 1 ; i <= (np-1)/2 ; ++i)
      {
         // initial guess is the corresponding Chebyshev point, x_i:
         //    x_i = -cos(\pi * (i / (np-1)))
         real_t x_i = std::sin(M_PI * ((real_t)(i)/(np-1) - 0.5));
         real_t z_i = 0., p_l;
         bool done = false;
         for (int iter = 0 ; true ; ++iter)
         {
            // build Legendre polynomials, up to P_{np}(x_i)
            real_t p_lm1 = 1.0;
            p_l = x_i;

            for (int l = 1 ; l < (np-1) ; ++l)
            {
               // The Legendre polynomials can be built by recursion:
               // x * P_l(x) = 1/(2*l+1)*[ (l+1)*P_{l+1}(x) + l*P_{l-1} ], i.e.
               // P_{l+1}(x) = [ (2*l+1)*x*P_l(x) - l*P_{l-1} ]/(l+1)
               real_t p_lp1 = ( (2*l + 1)*x_i*p_l - l*p_lm1)/(l + 1);

               p_lm1 = p_l;
               p_l = p_lp1;
            }
            if (done) { break; }
            // after this loop, p_l holds P_{np-1}(x_i)
            // resid = (x^2-1)*P'_{np-1}(x_i)
            // but use the recurrence relationship
            // (x^2 -1)P'_l(x) = l*[ x*P_l(x) - P_{l-1}(x) ]
            // thus, resid = (np-1) * (x_i*p_l - p_lm1)

            // The derivative of the residual is:
            // \frac{d}{d x} \left[ (x^2 -1)P'_l(x) ] \right] =
            // l * (l+1) * P_l(x), with l = np-1,
            // therefore, deriv = np * (np-1) * p_l;

            // compute dx = resid/deriv
            real_t dx = (x_i*p_l - p_lm1) / (np*p_l);
#ifdef MFEM_USE_SINGLE
            if (std::abs(dx) < 1e-7)
#elif defined MFEM_USE_DOUBLE
            if (std::abs(dx) < 1e-16)
#else
            MFEM_ABORT("Floating point type undefined");
            if (std::abs(dx) < 1e-16)
#endif
            {
               done = true;
               // Map the point to the interval [0,1]
               z_i = ((1.0 + x_i) - dx)/2;
               // continue the computation: get p_l at the new point, then exit
            }
            // If the iteration does not converge fast, something is wrong.
            MFEM_VERIFY(iter < 8, "np = " << np << ", i = " << i
                        << ", dx = " << dx);
            // update x_i:
            x_i -= dx;
         }
         // Map to the interval [0,1] and scale the weights
         IntegrationPoint &ip = ir->IntPoint(i);
         ip.x = z_i;
         // w_i = (2/[ n*(n-1)*[P_{n-1}(x_i)]^2 ]) / 2
         ip.weight = (real_t)(1.0 / (np*(np-1)*p_l*p_l));

         // set the symmetric point
         IntegrationPoint &symm_ip = ir->IntPoint(np-1-i);
         symm_ip.x = 1.0 - z_i;
         symm_ip.weight = ip.weight;
      }

#else // MFEM_USE_MPFR is defined

      HP_Quadrature1D hp_quad;
      // use symmetry and compute just half of the points
      for (int i = 0 ; i <= (np-1)/2 ; ++i)
      {
         hp_quad.ComputeGaussLobattoPoint(np, i);
         ir->IntPoint(i).x = hp_quad.GetPoint();
         ir->IntPoint(np-1-i).x = hp_quad.GetSymmPoint();
         ir->IntPoint(i).weight =
            ir->IntPoint(np-1-i).weight = hp_quad.GetWeight();
      }

#endif // MFEM_USE_MPFR

   }
}

void QuadratureFunctions1D::OpenUniform(const int np, IntegrationRule* ir)
{
   ir->SetSize(np);
   ir->SetPointIndices();
   ir->SetOrder(np - 1 + np%2);

   // The Newton-Cotes quadrature is based on weights that integrate exactly the
   // interpolatory polynomial through the equally spaced quadrature points.
   for (int i = 0; i < np ; ++i)
   {
      ir->IntPoint(i).x = real_t(i+1) / real_t(np + 1);
   }

   CalculateUniformWeights(ir, Quadrature1D::OpenUniform);
}

void QuadratureFunctions1D::ClosedUniform(const int np,
                                          IntegrationRule* ir)
{
   ir->SetSize(np);
   ir->SetPointIndices();
   ir->SetOrder(np - 1 + np%2);
   if ( np == 1 ) // allow this case as "closed"
   {
      ir->IntPoint(0).Set1w(0.5, 1.0);
      return;
   }

   for (int i = 0; i < np ; ++i)
   {
      ir->IntPoint(i).x = real_t(i) / (np-1);
   }

   CalculateUniformWeights(ir, Quadrature1D::ClosedUniform);
}

void QuadratureFunctions1D::OpenHalfUniform(const int np, IntegrationRule* ir)
{
   ir->SetSize(np);
   ir->SetPointIndices();
   ir->SetOrder(np - 1 + np%2);

   // Open half points: the centers of np uniform intervals
   for (int i = 0; i < np ; ++i)
   {
      ir->IntPoint(i).x = real_t(2*i+1) / (2*np);
   }

   CalculateUniformWeights(ir, Quadrature1D::OpenHalfUniform);
}

void QuadratureFunctions1D::ClosedGL(const int np, IntegrationRule* ir)
{
   ir->SetSize(np);
   ir->SetPointIndices();
   ir->IntPoint(0).x = 0.0;
   ir->IntPoint(np-1).x = 1.0;
   ir->SetOrder(np - 1 + np%2); // Is this the correct order?

   if ( np > 2 )
   {
      IntegrationRule gl_ir;
      GaussLegendre(np-1, &gl_ir);

      for (int i = 1; i < np-1; ++i)
      {
         ir->IntPoint(i).x = (gl_ir.IntPoint(i-1).x + gl_ir.IntPoint(i).x)/2;
      }
   }

   CalculateUniformWeights(ir, Quadrature1D::ClosedGL);
}

void QuadratureFunctions1D::GivePolyPoints(const int np, real_t *pts,
                                           const int type)
{
   IntegrationRule ir(np);

   switch (type)
   {
      case Quadrature1D::GaussLegendre:
      {
         GaussLegendre(np,&ir);
         break;
      }
      case Quadrature1D::GaussLobatto:
      {
         GaussLobatto(np, &ir);
         break;
      }
      case Quadrature1D::OpenUniform:
      {
         OpenUniform(np,&ir);
         break;
      }
      case Quadrature1D::ClosedUniform:
      {
         ClosedUniform(np,&ir);
         break;
      }
      case Quadrature1D::OpenHalfUniform:
      {
         OpenHalfUniform(np, &ir);
         break;
      }
      case Quadrature1D::ClosedGL:
      {
         ClosedGL(np, &ir);
         break;
      }
      case Quadrature1D::Invalid:
      {
         MFEM_ABORT("Asking for an unknown type of 1D Quadrature points, "
                    "type = " << type);
      }
   }

   for (int i = 0 ; i < np ; ++i)
   {
      pts[i] = ir.IntPoint(i).x;
   }
}

void QuadratureFunctions1D::CalculateUniformWeights(IntegrationRule *ir,
                                                    const int type)
{
   /* The Lagrange polynomials are:
           p_i = \prod_{j \neq i} {\frac{x - x_j }{x_i - x_j}}

      The weight associated with each abscissa is the integral of p_i over
      [0,1]. To calculate the integral of p_i, we use a Gauss-Legendre
      quadrature rule. This approach does not suffer from bad round-off/
      cancellation errors for large number of points.
   */
   const int n = ir->Size();
   switch (n)
   {
      case 1:
         ir->IntPoint(0).weight = 1.;
         return;
      case 2:
         ir->IntPoint(0).weight = .5;
         ir->IntPoint(1).weight = .5;
         return;
   }

#ifndef MFEM_USE_MPFR

   // This algorithm should work for any set of points, not just uniform
   const IntegrationRule &glob_ir = IntRules.Get(Geometry::SEGMENT, n-1);
   const int m = glob_ir.GetNPoints();
   Vector xv(n);
   for (int j = 0; j < n; j++)
   {
      xv(j) = ir->IntPoint(j).x;
   }
   Poly_1D::Basis basis(n-1, xv.GetData()); // nodal basis, with nodes at 'xv'
   Vector w(n);
   // Integrate all nodal basis functions using 'glob_ir':
   w = 0.0;
   for (int i = 0; i < m; i++)
   {
      const IntegrationPoint &ip = glob_ir.IntPoint(i);
      basis.Eval(ip.x, xv);
      w.Add(ip.weight, xv); // w += ip.weight * xv
   }
   for (int j = 0; j < n; j++)
   {
      ir->IntPoint(j).weight = w(j);
   }

#else // MFEM_USE_MPFR is defined

   static const mpfr_rnd_t rnd = HP_Quadrature1D::rnd;
   HP_Quadrature1D hp_quad;
   mpfr_t l, lk, w0, wi, tmp, *weights;
   mpfr_inits2(hp_quad.default_prec, l, lk, w0, wi, tmp, (mpfr_ptr) 0);
   weights = new mpfr_t[n];
   for (int i = 0; i < n; i++)
   {
      mpfr_init2(weights[i], hp_quad.default_prec);
      mpfr_set_si(weights[i], 0, rnd);
   }
   hp_quad.SetRelTol(-48); // rtol = 2^(-48) ~ 3.5e-15
   const int p = n-1;
   const int m = p/2+1; // number of points for Gauss-Legendre quadrature
   int hinv = 0, ihoffset = 0; // x_i = (i+ihoffset/2)/hinv
   switch (type)
   {
      case Quadrature1D::ClosedUniform:
         // x_i = i/p, i=0,...,p
         hinv = p;
         ihoffset = 0;
         break;
      case Quadrature1D::OpenUniform:
         // x_i = (i+1)/(p+2), i=0,...,p
         hinv = p+2;
         ihoffset = 2;
         break;
      case Quadrature1D::OpenHalfUniform:
         // x_i = (i+1/2)/(p+1), i=0,...,p
         hinv = p+1;
         ihoffset = 1;
         break;
      case Quadrature1D::GaussLegendre:
      case Quadrature1D::GaussLobatto:
      case Quadrature1D::ClosedGL:
      case Quadrature1D::Invalid:
         MFEM_ABORT("invalid Quadrature1D type: " << type);
   }
   // set w0 = (-1)^p*(p!)/(hinv^p)
   mpfr_fac_ui(w0, p, rnd);
   mpfr_ui_pow_ui(tmp, hinv, p, rnd);
   mpfr_div(w0, w0, tmp, rnd);
   if (p%2) { mpfr_neg(w0, w0, rnd); }

   for (int j = 0; j < m; j++)
   {
      hp_quad.ComputeGaussLegendrePoint(m, j);

      // Compute l = \prod_{i=0}^p (x-x_i) and lk = l/(x-x_k), where
      // x = hp_quad.GetHPPoint(), x_i = (i+ihoffset/2)/hinv, and x_k is the
      // node closest to x, i.e. k = min(max(round(x*hinv-ihoffset/2),0),p)
      mpfr_mul_si(tmp, hp_quad.GetHPPoint(), hinv, rnd);
      mpfr_sub_d(tmp, tmp, 0.5*ihoffset, rnd);
      mpfr_round(tmp, tmp);
      int k = min(max((int)mpfr_get_si(tmp, rnd), 0), p);
      mpfr_set_si(lk, 1, rnd);
      for (int i = 0; i <= p; i++)
      {
         mpfr_set_si(tmp, 2*i+ihoffset, rnd);
         mpfr_div_si(tmp, tmp, 2*hinv, rnd);
         mpfr_sub(tmp, hp_quad.GetHPPoint(), tmp, rnd);
         if (i != k)
         {
            mpfr_mul(lk, lk, tmp, rnd);
         }
         else
         {
            mpfr_set(l, tmp, rnd);
         }
      }
      mpfr_mul(l, l, lk, rnd);
      mpfr_set(wi, w0, rnd);
      for (int i = 0; true; i++)
      {
         if (i != k)
         {
            // tmp = l/(wi*(x - x_i))
            mpfr_set_si(tmp, 2*i+ihoffset, rnd);
            mpfr_div_si(tmp, tmp, 2*hinv, rnd);
            mpfr_sub(tmp, hp_quad.GetHPPoint(), tmp, rnd);
            mpfr_mul(tmp, tmp, wi, rnd);
            mpfr_div(tmp, l, tmp, rnd);
         }
         else
         {
            // tmp = lk/wi
            mpfr_div(tmp, lk, wi, rnd);
         }
         // weights[i] += hp_quad.weight*tmp
         mpfr_mul(tmp, tmp, hp_quad.GetHPWeight(), rnd);
         mpfr_add(weights[i], weights[i], tmp, rnd);

         if (i == p) { break; }

         // update wi *= (i+1)/(i-p)
         mpfr_mul_si(wi, wi, i+1, rnd);
         mpfr_div_si(wi, wi, i-p, rnd);
      }
   }
   for (int i = 0; i < n; i++)
   {
      ir->IntPoint(i).weight = mpfr_get_d(weights[i], rnd);
      mpfr_clear(weights[i]);
   }
   delete [] weights;
   mpfr_clears(l, lk, w0, wi, tmp, (mpfr_ptr) 0);

#endif // MFEM_USE_MPFR

}


int Quadrature1D::CheckClosed(int type)
{
   switch (type)
   {
      case GaussLobatto:
      case ClosedUniform:
      case ClosedGL:
         return type;
      default:
         return Invalid;
   }
}

int Quadrature1D::CheckOpen(int type)
{
   switch (type)
   {
      case GaussLegendre:
      case GaussLobatto:
      case OpenUniform:
      case ClosedUniform:
      case OpenHalfUniform:
         return type; // all types can work as open
      default:
         return Invalid;
   }
}


IntegrationRules IntRules(0, Quadrature1D::GaussLegendre);

IntegrationRules RefinedIntRules(1, Quadrature1D::GaussLegendre);

IntegrationRules::IntegrationRules(int ref, int type)
   : quad_type(type)
{
   refined = ref;

   if (refined < 0) { own_rules = 0; return; }

   own_rules = 1;

   const MemoryType h_mt = MemoryType::HOST;
   PointIntRules.SetSize(2, h_mt);
   PointIntRules = NULL;

   SegmentIntRules.SetSize(32, h_mt);
   SegmentIntRules = NULL;

   // TriangleIntegrationRule() assumes that this size is >= 26
   TriangleIntRules.SetSize(32, h_mt);
   TriangleIntRules = NULL;

   SquareIntRules.SetSize(32, h_mt);
   SquareIntRules = NULL;

   // TetrahedronIntegrationRule() assumes that this size is >= 10
   TetrahedronIntRules.SetSize(32, h_mt);
   TetrahedronIntRules = NULL;

   PyramidIntRules.SetSize(32, h_mt);
   PyramidIntRules = NULL;

   PrismIntRules.SetSize(32, h_mt);
   PrismIntRules = NULL;

   CubeIntRules.SetSize(32, h_mt);
   CubeIntRules = NULL;

#if defined(MFEM_THREAD_SAFE) && defined(MFEM_USE_OPENMP)
   IntRuleLocks.SetSize(Geometry::NUM_GEOMETRIES, h_mt);
   for (int i = 0; i < Geometry::NUM_GEOMETRIES; i++)
   {
      omp_init_lock(&IntRuleLocks[i]);
   }
#endif
}

const IntegrationRule &IntegrationRules::Get(int GeomType, int Order)
{
   Array<IntegrationRule *> *ir_array = NULL;

   switch (GeomType)
   {
      case Geometry::POINT:       ir_array = &PointIntRules; Order = 0; break;
      case Geometry::SEGMENT:     ir_array = &SegmentIntRules; break;
      case Geometry::TRIANGLE:    ir_array = &TriangleIntRules; break;
      case Geometry::SQUARE:      ir_array = &SquareIntRules; break;
      case Geometry::TETRAHEDRON: ir_array = &TetrahedronIntRules; break;
      case Geometry::CUBE:        ir_array = &CubeIntRules; break;
      case Geometry::PRISM:       ir_array = &PrismIntRules; break;
      case Geometry::PYRAMID:     ir_array = &PyramidIntRules; break;
      case Geometry::INVALID:
      case Geometry::NUM_GEOMETRIES:
         MFEM_ABORT("Unknown type of reference element!");
   }

   if (Order < 0)
   {
      Order = 0;
   }

#if defined(MFEM_THREAD_SAFE) && defined(MFEM_USE_OPENMP)
   omp_set_lock(&IntRuleLocks[GeomType]);
#endif

   if (!HaveIntRule(*ir_array, Order))
   {
      IntegrationRule *ir = GenerateIntegrationRule(GeomType, Order);
#ifdef MFEM_DEBUG
      int RealOrder = Order;
      while (RealOrder+1 < ir_array->Size() && (*ir_array)[RealOrder+1] == ir)
      {
         RealOrder++;
      }
      MFEM_VERIFY(RealOrder == ir->GetOrder(), "internal error");
#else
      MFEM_CONTRACT_VAR(ir);
#endif
   }

#if defined(MFEM_THREAD_SAFE) && defined(MFEM_USE_OPENMP)
   omp_unset_lock(&IntRuleLocks[GeomType]);
#endif

   return *(*ir_array)[Order];
}

void IntegrationRules::Set(int GeomType, int Order, IntegrationRule &IntRule)
{
   Array<IntegrationRule *> *ir_array = NULL;

   switch (GeomType)
   {
      case Geometry::POINT:       ir_array = &PointIntRules; break;
      case Geometry::SEGMENT:     ir_array = &SegmentIntRules; break;
      case Geometry::TRIANGLE:    ir_array = &TriangleIntRules; break;
      case Geometry::SQUARE:      ir_array = &SquareIntRules; break;
      case Geometry::TETRAHEDRON: ir_array = &TetrahedronIntRules; break;
      case Geometry::CUBE:        ir_array = &CubeIntRules; break;
      case Geometry::PRISM:       ir_array = &PrismIntRules; break;
      case Geometry::PYRAMID:     ir_array = &PyramidIntRules; break;
      case Geometry::INVALID:
      case Geometry::NUM_GEOMETRIES:
         MFEM_ABORT("Unknown type of reference element!");
   }

#if defined(MFEM_THREAD_SAFE) && defined(MFEM_USE_OPENMP)
   omp_set_lock(&IntRuleLocks[GeomType]);
#endif

   if (HaveIntRule(*ir_array, Order))
   {
      MFEM_ABORT("Overwriting set rules is not supported!");
   }

   AllocIntRule(*ir_array, Order);

   (*ir_array)[Order] = &IntRule;

#if defined(MFEM_THREAD_SAFE) && defined(MFEM_USE_OPENMP)
   omp_unset_lock(&IntRuleLocks[GeomType]);
#endif
}

void IntegrationRules::DeleteIntRuleArray(
   Array<IntegrationRule *> &ir_array) const
{
   // Many of the intrules have multiple contiguous copies in the ir_array
   // so we have to be careful to not delete them twice.
   IntegrationRule *ir = NULL;
   for (int i = 0; i < ir_array.Size(); i++)
   {
      if (ir_array[i] != NULL && ir_array[i] != ir)
      {
         ir = ir_array[i];
         delete ir;
      }
   }
}

IntegrationRules::~IntegrationRules()
{
#if defined(MFEM_THREAD_SAFE) && defined(MFEM_USE_OPENMP)
   for (int i = 0; i < Geometry::NUM_GEOMETRIES; i++)
   {
      omp_destroy_lock(&IntRuleLocks[i]);
   }
#endif

   if (!own_rules) { return; }

   DeleteIntRuleArray(PointIntRules);
   DeleteIntRuleArray(SegmentIntRules);
   DeleteIntRuleArray(TriangleIntRules);
   DeleteIntRuleArray(SquareIntRules);
   DeleteIntRuleArray(TetrahedronIntRules);
   DeleteIntRuleArray(CubeIntRules);
   DeleteIntRuleArray(PrismIntRules);
   DeleteIntRuleArray(PyramidIntRules);
}


IntegrationRule *IntegrationRules::GenerateIntegrationRule(int GeomType,
                                                           int Order)
{
   switch (GeomType)
   {
      case Geometry::POINT:
         return PointIntegrationRule(Order);
      case Geometry::SEGMENT:
         return SegmentIntegrationRule(Order);
      case Geometry::TRIANGLE:
         return TriangleIntegrationRule(Order);
      case Geometry::SQUARE:
         return SquareIntegrationRule(Order);
      case Geometry::TETRAHEDRON:
         return TetrahedronIntegrationRule(Order);
      case Geometry::CUBE:
         return CubeIntegrationRule(Order);
      case Geometry::PRISM:
         return PrismIntegrationRule(Order);
      case Geometry::PYRAMID:
         return PyramidIntegrationRule(Order);
      case Geometry::INVALID:
      case Geometry::NUM_GEOMETRIES:
         MFEM_ABORT("Unknown type of reference element!");
   }
   return NULL;
}


// Integration rules for a point
IntegrationRule *IntegrationRules::PointIntegrationRule(int Order)
{
   if (Order > 1)
   {
      MFEM_ABORT("Point Integration Rule of Order > 1 not defined");
      return NULL;
   }

   IntegrationRule *ir = new IntegrationRule(1);
   ir->IntPoint(0).x = .0;
   ir->IntPoint(0).weight = 1.;
   ir->SetOrder(1);

   PointIntRules[1] = PointIntRules[0] = ir;

   return ir;
}

// Integration rules for line segment [0,1]
IntegrationRule *IntegrationRules::SegmentIntegrationRule(int Order)
{
   int RealOrder = GetSegmentRealOrder(Order); // RealOrder >= Order
   // Order is one of {RealOrder-1,RealOrder}
   AllocIntRule(SegmentIntRules, RealOrder);

   IntegrationRule *ir = new IntegrationRule;

   int n = 0;
   // n is the number of points to achieve the exact integral of a
   // degree Order polynomial
   switch (quad_type)
   {
      case Quadrature1D::GaussLegendre:
      {
         // Gauss-Legendre is exact for 2*n-1
         n = Order/2 + 1;
         QuadratureFunctions1D::GaussLegendre(n, ir);
         break;
      }
      case Quadrature1D::GaussLobatto:
      {
         // Gauss-Lobatto is exact for 2*n-3
         n = Order/2 + 2;
         QuadratureFunctions1D::GaussLobatto(n, ir);
         break;
      }
      case Quadrature1D::OpenUniform:
      {
         // Open Newton Cotes is exact for n-(n+1)%2 = n-1+n%2
         n = Order | 1; // n is always odd
         QuadratureFunctions1D::OpenUniform(n, ir);
         break;
      }
      case Quadrature1D::ClosedUniform:
      {
         // Closed Newton Cotes is exact for n-(n+1)%2 = n-1+n%2
         n = Order | 1; // n is always odd
         QuadratureFunctions1D::ClosedUniform(n, ir);
         break;
      }
      case Quadrature1D::OpenHalfUniform:
      {
         // Open half Newton Cotes is exact for n-(n+1)%2 = n-1+n%2
         n = Order | 1; // n is always odd
         QuadratureFunctions1D::OpenHalfUniform(n, ir);
         break;
      }
      case Quadrature1D::Invalid:
      {
         MFEM_ABORT("unknown Quadrature1D type: " << quad_type);
      }
   }
   if (refined)
   {
      // Effectively passing memory management to SegmentIntegrationRules
      IntegrationRule *refined_ir = new IntegrationRule(2*n);
      refined_ir->SetOrder(ir->GetOrder());
      for (int j = 0; j < n; j++)
      {
         refined_ir->IntPoint(j).x = ir->IntPoint(j).x/2.0;
         refined_ir->IntPoint(j).weight = ir->IntPoint(j).weight/2.0;
         refined_ir->IntPoint(j+n).x = 0.5 + ir->IntPoint(j).x/2.0;
         refined_ir->IntPoint(j+n).weight = ir->IntPoint(j).weight/2.0;
      }
      delete ir;
      ir = refined_ir;
   }
   SegmentIntRules[RealOrder-1] = SegmentIntRules[RealOrder] = ir;
   return ir;
}

// Integration rules for reference triangle {[0,0],[1,0],[0,1]}
IntegrationRule *IntegrationRules::TriangleIntegrationRule(int Order)
{
   IntegrationRule *ir = NULL;
   // Note: Set TriangleIntRules[*] to ir only *after* ir is fully constructed.
   // This is needed in multithreaded environment.

   // assuming that orders <= 25 are pre-allocated
   switch (Order)
   {
      case 0:  // 1 point - degree 1
      case 1:
         ir = new IntegrationRule(1);
         ir->AddTriMidPoint(0, 0.5);
         ir->SetOrder(1);
         TriangleIntRules[0] = TriangleIntRules[1] = ir;
         return ir;

      case 2:  // 3 point - 2 degree
         ir = new IntegrationRule(3);
         ir->AddTriPoints3(0, 1./6., 1./6.);
         ir->SetOrder(2);
         TriangleIntRules[2] = ir;
         // interior points
         return ir;

      case 3:  // 4 point - 3 degree (has one negative weight)
         ir = new IntegrationRule(4);
         ir->AddTriMidPoint(0, -0.28125); // -9./32.
         ir->AddTriPoints3(1, 0.2, 25./96.);
         ir->SetOrder(3);
         TriangleIntRules[3] = ir;
         return ir;

      case 4:  // 6 point - 4 degree
         ir = new IntegrationRule(6);
         ir->AddTriPoints3(0, 0.091576213509770743460, 0.054975871827660933819);
         ir->AddTriPoints3(3, 0.44594849091596488632, 0.11169079483900573285);
         ir->SetOrder(4);
         TriangleIntRules[4] = ir;
         return ir;

      case 5:  // 7 point - 5 degree
         ir = new IntegrationRule(7);
         ir->AddTriMidPoint(0, 0.1125);
         ir->AddTriPoints3(1, 0.10128650732345633880, 0.062969590272413576298);
         ir->AddTriPoints3(4, 0.47014206410511508977, 0.066197076394253090369);
         ir->SetOrder(5);
         TriangleIntRules[5] = ir;
         return ir;

      case 6:  // 12 point - 6 degree
         ir = new IntegrationRule(12);
         ir->AddTriPoints3(0, 0.063089014491502228340, 0.025422453185103408460);
         ir->AddTriPoints3(3, 0.24928674517091042129, 0.058393137863189683013);
         ir->AddTriPoints6(6, 0.053145049844816947353, 0.31035245103378440542,
                           0.041425537809186787597);
         ir->SetOrder(6);
         TriangleIntRules[6] = ir;
         return ir;

      case 7:  // 12 point - degree 7
         ir = new IntegrationRule(12);
         ir->AddTriPoints3R(0, 0.062382265094402118174, 0.067517867073916085443,
                            0.026517028157436251429);
         ir->AddTriPoints3R(3, 0.055225456656926611737, 0.32150249385198182267,
                            0.043881408714446055037);
         // slightly better with explicit 3rd area coordinate
         ir->AddTriPoints3R(6, 0.034324302945097146470, 0.66094919618673565761,
                            0.30472650086816719592, 0.028775042784981585738);
         ir->AddTriPoints3R(9, 0.51584233435359177926, 0.27771616697639178257,
                            0.20644149867001643817, 0.067493187009802774463);
         ir->SetOrder(7);
         TriangleIntRules[7] = ir;
         return ir;

      case 8:  // 16 point - 8 degree
         ir = new IntegrationRule(16);
         ir->AddTriMidPoint(0, 0.0721578038388935841255455552445323);
         ir->AddTriPoints3(1, 0.170569307751760206622293501491464,
                           0.0516086852673591251408957751460645);
         ir->AddTriPoints3(4, 0.0505472283170309754584235505965989,
                           0.0162292488115990401554629641708902);
         ir->AddTriPoints3(7, 0.459292588292723156028815514494169,
                           0.0475458171336423123969480521942921);
         ir->AddTriPoints6(10, 0.008394777409957605337213834539296,
                           0.263112829634638113421785786284643,
                           0.0136151570872174971324223450369544);
         ir->SetOrder(8);
         TriangleIntRules[8] = ir;
         return ir;

      case 9:  // 19 point - 9 degree
         ir = new IntegrationRule(19);
         ir->AddTriMidPoint(0, 0.0485678981413994169096209912536443);
         ir->AddTriPoints3b(1, 0.020634961602524744433,
                            0.0156673501135695352684274156436046);
         ir->AddTriPoints3b(4, 0.12582081701412672546,
                            0.0389137705023871396583696781497019);
         ir->AddTriPoints3(7, 0.188203535619032730240961280467335,
                           0.0398238694636051265164458871320226);
         ir->AddTriPoints3(10, 0.0447295133944527098651065899662763,
                           0.0127888378293490156308393992794999);
         ir->AddTriPoints6(13, 0.0368384120547362836348175987833851,
                           0.2219629891607656956751025276931919,
                           0.0216417696886446886446886446886446);
         ir->SetOrder(9);
         TriangleIntRules[9] = ir;
         return ir;

      case 10:  // 25 point - 10 degree
         ir = new IntegrationRule(25);
         ir->AddTriMidPoint(0, 0.0454089951913767900476432975500142);
         ir->AddTriPoints3b(1, 0.028844733232685245264984935583748,
                            0.0183629788782333523585030359456832);
         ir->AddTriPoints3(4, 0.109481575485037054795458631340522,
                           0.0226605297177639673913028223692986);
         ir->AddTriPoints6(7, 0.141707219414879954756683250476361,
                           0.307939838764120950165155022930631,
                           0.0363789584227100543021575883096803);
         ir->AddTriPoints6(13, 0.025003534762686386073988481007746,
                           0.246672560639902693917276465411176,
                           0.0141636212655287424183685307910495);
         ir->AddTriPoints6(19, 0.0095408154002994575801528096228873,
                           0.0668032510122002657735402127620247,
                           4.71083348186641172996373548344341E-03);
         ir->SetOrder(10);
         TriangleIntRules[10] = ir;
         return ir;

      case 11: // 28 point -- 11 degree
         ir = new IntegrationRule(28);
         ir->AddTriPoints6(0, 0.0,
                           0.141129718717363295960826061941652,
                           3.68119189165027713212944752369032E-03);
         ir->AddTriMidPoint(6, 0.0439886505811161193990465846607278);
         ir->AddTriPoints3(7, 0.0259891409282873952600324854988407,
                           4.37215577686801152475821439991262E-03);
         ir->AddTriPoints3(10, 0.0942875026479224956305697762754049,
                           0.0190407859969674687575121697178070);
         ir->AddTriPoints3b(13, 0.010726449965572372516734795387128,
                            9.42772402806564602923839129555767E-03);
         ir->AddTriPoints3(16, 0.207343382614511333452934024112966,
                           0.0360798487723697630620149942932315);
         ir->AddTriPoints3b(19, 0.122184388599015809877869236727746,
                            0.0346645693527679499208828254519072);
         ir->AddTriPoints6(22, 0.0448416775891304433090523914688007,
                           0.2772206675282791551488214673424523,
                           0.0205281577146442833208261574536469);
         ir->SetOrder(11);
         TriangleIntRules[11] = ir;
         return ir;

      case 12: // 33 point - 12 degree
         ir = new IntegrationRule(33);
         ir->AddTriPoints3b(0, 2.35652204523900E-02, 1.28655332202275E-02);
         ir->AddTriPoints3b(3, 1.20551215411079E-01, 2.18462722690190E-02);
         ir->AddTriPoints3(6, 2.71210385012116E-01, 3.14291121089425E-02);
         ir->AddTriPoints3(9, 1.27576145541586E-01, 1.73980564653545E-02);
         ir->AddTriPoints3(12, 2.13173504532100E-02, 3.08313052577950E-03);
         ir->AddTriPoints6(15, 1.15343494534698E-01, 2.75713269685514E-01,
                           2.01857788831905E-02);
         ir->AddTriPoints6(21, 2.28383322222570E-02, 2.81325580989940E-01,
                           1.11783866011515E-02);
         ir->AddTriPoints6(27, 2.57340505483300E-02, 1.16251915907597E-01,
                           8.65811555432950E-03);
         ir->SetOrder(12);
         TriangleIntRules[12] = ir;
         return ir;

      case 13: // 37 point - 13 degree
         ir = new IntegrationRule(37);
         ir->AddTriPoints3b(0, 0.0,
                            2.67845189554543044455908674650066E-03);
         ir->AddTriMidPoint(3, 0.0293480398063595158995969648597808);
         ir->AddTriPoints3(4, 0.0246071886432302181878499494124643,
                           3.92538414805004016372590903990464E-03);
         ir->AddTriPoints3b(7, 0.159382493797610632566158925635800,
                            0.0253344765879434817105476355306468);
         ir->AddTriPoints3(10, 0.227900255506160619646298948153592,
                           0.0250401630452545330803738542916538);
         ir->AddTriPoints3(13, 0.116213058883517905247155321839271,
                           0.0158235572961491595176634480481793);
         ir->AddTriPoints3b(16, 0.046794039901841694097491569577008,
                            0.0157462815379843978450278590138683);
         ir->AddTriPoints6(19, 0.0227978945382486125477207592747430,
                           0.1254265183163409177176192369310890,
                           7.90126610763037567956187298486575E-03);
         ir->AddTriPoints6(25, 0.0162757709910885409437036075960413,
                           0.2909269114422506044621801030055257,
                           7.99081889046420266145965132482933E-03);
         ir->AddTriPoints6(31, 0.0897330604516053590796290561145196,
                           0.2723110556841851025078181617634414,
                           0.0182757511120486476280967518782978);
         ir->SetOrder(13);
         TriangleIntRules[13] = ir;
         return ir;

      case 14: // 42 point - 14 degree
         ir = new IntegrationRule(42);
         ir->AddTriPoints3b(0, 2.20721792756430E-02, 1.09417906847145E-02);
         ir->AddTriPoints3b(3, 1.64710561319092E-01, 1.63941767720625E-02);
         ir->AddTriPoints3(6, 2.73477528308839E-01, 2.58870522536460E-02);
         ir->AddTriPoints3(9, 1.77205532412543E-01, 2.10812943684965E-02);
         ir->AddTriPoints3(12, 6.17998830908730E-02, 7.21684983488850E-03);
         ir->AddTriPoints3(15, 1.93909612487010E-02, 2.46170180120000E-03);
         ir->AddTriPoints6(18, 5.71247574036480E-02, 1.72266687821356E-01,
                           1.23328766062820E-02);
         ir->AddTriPoints6(24, 9.29162493569720E-02, 3.36861459796345E-01,
                           1.92857553935305E-02);
         ir->AddTriPoints6(30, 1.46469500556540E-02, 2.98372882136258E-01,
                           7.21815405676700E-03);
         ir->AddTriPoints6(36, 1.26833093287200E-03, 1.18974497696957E-01,
                           2.50511441925050E-03);
         ir->SetOrder(14);
         TriangleIntRules[14] = ir;
         return ir;

      case 15: // 54 point - 15 degree
         ir = new IntegrationRule(54);
         ir->AddTriPoints3b(0, 0.0834384072617499333, 0.016330909424402645);
         ir->AddTriPoints3b(3, 0.192779070841738867, 0.01370640901568218);
         ir->AddTriPoints3(6, 0.293197167913025367, 0.01325501829935165);
         ir->AddTriPoints3(9, 0.146467786942772933, 0.014607981068243055);
         ir->AddTriPoints3(12, 0.0563628676656034333, 0.005292304033121995);
         ir->AddTriPoints3(15, 0.0165751268583703333, 0.0018073215320460175);
         ir->AddTriPoints6(18, 0.0099122033092248, 0.239534554154794445,
                           0.004263874050854718);
         ir->AddTriPoints6(24, 0.015803770630228, 0.404878807318339958,
                           0.006958088258345965);
         ir->AddTriPoints6(30, 0.00514360881697066667, 0.0950021131130448885,
                           0.0021459664703674175);
         ir->AddTriPoints6(36, 0.0489223257529888, 0.149753107322273969,
                           0.008117664640887445);
         ir->AddTriPoints6(42, 0.0687687486325192, 0.286919612441334979,
                           0.012803670460631195);
         ir->AddTriPoints6(48, 0.1684044181246992, 0.281835668099084562,
                           0.016544097765822835);
         ir->SetOrder(15);
         TriangleIntRules[15] = ir;
         return ir;

      case 16:  // 61 point - 17 degree (used for 16 as well)
      case 17:
         ir = new IntegrationRule(61);
         ir->AddTriMidPoint(0,  1.67185996454015E-02);
         ir->AddTriPoints3b(1,  5.65891888645200E-03, 2.54670772025350E-03);
         ir->AddTriPoints3b(4,  3.56473547507510E-02, 7.33543226381900E-03);
         ir->AddTriPoints3b(7,  9.95200619584370E-02, 1.21754391768360E-02);
         ir->AddTriPoints3b(10, 1.99467521245206E-01, 1.55537754344845E-02);
         ir->AddTriPoints3 (13, 2.52141267970953E-01, 1.56285556093100E-02);
         ir->AddTriPoints3 (16, 1.62047004658461E-01, 1.24078271698325E-02);
         ir->AddTriPoints3 (19, 7.58758822607460E-02, 7.02803653527850E-03);
         ir->AddTriPoints3 (22, 1.56547269678220E-02, 1.59733808688950E-03);
         ir->AddTriPoints6 (25, 1.01869288269190E-02, 3.34319867363658E-01,
                            4.05982765949650E-03);
         ir->AddTriPoints6 (31, 1.35440871671036E-01, 2.92221537796944E-01,
                            1.34028711415815E-02);
         ir->AddTriPoints6 (37, 5.44239242905830E-02, 3.19574885423190E-01,
                            9.22999660541100E-03);
         ir->AddTriPoints6 (43, 1.28685608336370E-02, 1.90704224192292E-01,
                            4.23843426716400E-03);
         ir->AddTriPoints6 (49, 6.71657824135240E-02, 1.80483211648746E-01,
                            9.14639838501250E-03);
         ir->AddTriPoints6 (55, 1.46631822248280E-02, 8.07113136795640E-02,
                            3.33281600208250E-03);
         ir->SetOrder(17);
         TriangleIntRules[16] = TriangleIntRules[17] = ir;
         return ir;

      case 18: // 73 point - 19 degree (used for 18 as well)
      case 19:
         ir = new IntegrationRule(73);
         ir->AddTriMidPoint(0,  0.0164531656944595);
         ir->AddTriPoints3b(1,  0.020780025853987, 0.005165365945636);
         ir->AddTriPoints3b(4,  0.090926214604215, 0.011193623631508);
         ir->AddTriPoints3b(7,  0.197166638701138, 0.015133062934734);
         ir->AddTriPoints3 (10, 0.255551654403098, 0.015245483901099);
         ir->AddTriPoints3 (13, 0.17707794215213,  0.0120796063708205);
         ir->AddTriPoints3 (16, 0.110061053227952, 0.0080254017934005);
         ir->AddTriPoints3 (19, 0.05552862425184,  0.004042290130892);
         ir->AddTriPoints3 (22, 0.012621863777229, 0.0010396810137425);
         ir->AddTriPoints6 (25, 0.003611417848412, 0.395754787356943,
                            0.0019424384524905);
         ir->AddTriPoints6 (31, 0.13446675453078, 0.307929983880436,
                            0.012787080306011);
         ir->AddTriPoints6 (37, 0.014446025776115, 0.26456694840652,
                            0.004440451786669);
         ir->AddTriPoints6 (43, 0.046933578838178, 0.358539352205951,
                            0.0080622733808655);
         ir->AddTriPoints6 (49, 0.002861120350567, 0.157807405968595,
                            0.0012459709087455);
         ir->AddTriPoints6 (55, 0.075050596975911, 0.223861424097916,
                            0.0091214200594755);
         ir->AddTriPoints6 (61, 0.03464707481676, 0.142421601113383,
                            0.0051292818680995);
         ir->AddTriPoints6 (67, 0.065494628082938, 0.010161119296278,
                            0.001899964427651);
         ir->SetOrder(19);
         TriangleIntRules[18] = TriangleIntRules[19] = ir;
         return ir;

      case 20: // 85 point - 20 degree
         ir = new IntegrationRule(85);
         ir->AddTriMidPoint(0, 0.01380521349884976);
         ir->AddTriPoints3b(1, 0.001500649324429,     0.00088951477366337);
         ir->AddTriPoints3b(4, 0.0941397519389508667, 0.010056199056980585);
         ir->AddTriPoints3b(7, 0.2044721240895264,    0.013408923629665785);
         ir->AddTriPoints3(10, 0.264500202532787333,  0.012261566900751005);
         ir->AddTriPoints3(13, 0.211018964092076767,  0.008197289205347695);
         ir->AddTriPoints3(16, 0.107735607171271333,  0.0073979536993248);
         ir->AddTriPoints3(19, 0.0390690878378026667, 0.0022896411388521255);
         ir->AddTriPoints3(22, 0.0111743797293296333, 0.0008259132577881085);
         ir->AddTriPoints6(25, 0.00534961818733726667, 0.0635496659083522206,
                           0.001174585454287792);
         ir->AddTriPoints6(31, 0.00795481706619893333, 0.157106918940706982,
                           0.0022329628770908965);
         ir->AddTriPoints6(37, 0.0104223982812638,     0.395642114364374018,
                           0.003049783403953986);
         ir->AddTriPoints6(43, 0.0109644147961233333,  0.273167570712910522,
                           0.0034455406635941015);
         ir->AddTriPoints6(49, 0.0385667120854623333,  0.101785382485017108,
                           0.0039987375362390815);
         ir->AddTriPoints6(55, 0.0355805078172182,     0.446658549176413815,
                           0.003693067142668012);
         ir->AddTriPoints6(61, 0.0496708163627641333,  0.199010794149503095,
                           0.00639966593932413);
         ir->AddTriPoints6(67, 0.0585197250843317333,  0.3242611836922827,
                           0.008629035587848275);
         ir->AddTriPoints6(73, 0.121497787004394267,   0.208531363210132855,
                           0.009336472951467735);
         ir->AddTriPoints6(79, 0.140710844943938733,   0.323170566536257485,
                           0.01140911202919763);
         ir->SetOrder(20);
         TriangleIntRules[20] = ir;
         return ir;

      case 21: // 126 point - 25 degree (used also for degrees from 21 to 24)
      case 22:
      case 23:
      case 24:
      case 25:
         ir = new IntegrationRule(126);
         ir->AddTriPoints3b(0, 0.0279464830731742,   0.0040027909400102085);
         ir->AddTriPoints3b(3, 0.131178601327651467, 0.00797353841619525);
         ir->AddTriPoints3b(6, 0.220221729512072267, 0.006554570615397765);
         ir->AddTriPoints3 (9, 0.298443234019804467,   0.00979150048281781);
         ir->AddTriPoints3(12, 0.2340441723373718,     0.008235442720768635);
         ir->AddTriPoints3(15, 0.151468334609017567,   0.00427363953704605);
         ir->AddTriPoints3(18, 0.112733893545993667,   0.004080942928613246);
         ir->AddTriPoints3(21, 0.0777156920915263,     0.0030605732699918895);
         ir->AddTriPoints3(24, 0.034893093614297,      0.0014542491324683325);
         ir->AddTriPoints3(27, 0.00725818462093236667, 0.00034613762283099815);
         ir->AddTriPoints6(30,  0.0012923527044422,     0.227214452153364077,
                           0.0006241445996386985);
         ir->AddTriPoints6(36,  0.0053997012721162,     0.435010554853571706,
                           0.001702376454401511);
         ir->AddTriPoints6(42,  0.006384003033975,      0.320309599272204437,
                           0.0016798271630320255);
         ir->AddTriPoints6(48,  0.00502821150199306667, 0.0917503222800051889,
                           0.000858078269748377);
         ir->AddTriPoints6(54,  0.00682675862178186667, 0.0380108358587243835,
                           0.000740428158357803);
         ir->AddTriPoints6(60,  0.0100161996399295333,  0.157425218485311668,
                           0.0017556563053643425);
         ir->AddTriPoints6(66,  0.02575781317339,       0.239889659778533193,
                           0.003696775074853242);
         ir->AddTriPoints6(72,  0.0302278981199158,     0.361943118126060531,
                           0.003991543738688279);
         ir->AddTriPoints6(78,  0.0305049901071620667,  0.0835519609548285602,
                           0.0021779813065790205);
         ir->AddTriPoints6(84,  0.0459565473625693333,  0.148443220732418205,
                           0.003682528350708916);
         ir->AddTriPoints6(90,  0.0674428005402775333,  0.283739708727534955,
                           0.005481786423209775);
         ir->AddTriPoints6(96,  0.0700450914159106,     0.406899375118787573,
                           0.00587498087177056);
         ir->AddTriPoints6(102, 0.0839115246401166,     0.194113987024892542,
                           0.005007800356899285);
         ir->AddTriPoints6(108, 0.120375535677152667,   0.32413434700070316,
                           0.00665482039381434);
         ir->AddTriPoints6(114, 0.148066899157366667,   0.229277483555980969,
                           0.00707722325261307);
         ir->AddTriPoints6(120, 0.191771865867325067,   0.325618122595983752,
                           0.007440689780584005);
         ir->SetOrder(25);
         TriangleIntRules[21] =
            TriangleIntRules[22] =
               TriangleIntRules[23] =
                  TriangleIntRules[24] =
                     TriangleIntRules[25] = ir;
         return ir;

      default:
         // Grundmann-Moller rules
         int i = (Order / 2) * 2 + 1;   // Get closest odd # >= Order
         AllocIntRule(TriangleIntRules, i);
         ir = new IntegrationRule;
         ir->GrundmannMollerSimplexRule(i/2,2);
         TriangleIntRules[i-1] = TriangleIntRules[i] = ir;
         return ir;
   }
}

// Integration rules for unit square
IntegrationRule *IntegrationRules::SquareIntegrationRule(int Order)
{
   int RealOrder = GetSegmentRealOrder(Order);
   // Order is one of {RealOrder-1,RealOrder}
   if (!HaveIntRule(SegmentIntRules, RealOrder))
   {
      SegmentIntegrationRule(RealOrder);
   }
   AllocIntRule(SquareIntRules, RealOrder); // RealOrder >= Order
   SquareIntRules[RealOrder-1] =
      SquareIntRules[RealOrder] =
         new IntegrationRule(*SegmentIntRules[RealOrder],
                             *SegmentIntRules[RealOrder]);
   return SquareIntRules[Order];
}

/** Integration rules for reference tetrahedron
    {[0,0,0],[1,0,0],[0,1,0],[0,0,1]}          */
IntegrationRule *IntegrationRules::TetrahedronIntegrationRule(int Order)
{
   IntegrationRule *ir;
   // Note: Set TetrahedronIntRules[*] to ir only *after* ir is fully
   // constructed. This is needed in multithreaded environment.

   // assuming that orders <= 9 are pre-allocated
   switch (Order)
   {
      case 0:  // 1 point - degree 1
      case 1:
         ir = new IntegrationRule(1);
         ir->AddTetMidPoint(0, 1./6.);
         ir->SetOrder(1);
         TetrahedronIntRules[0] = TetrahedronIntRules[1] = ir;
         return ir;

      case 2:  // 4 points - degree 2
         ir = new IntegrationRule(4);
         // ir->AddTetPoints4(0, 0.13819660112501051518, 1./24.);
         ir->AddTetPoints4b(0, 0.58541019662496845446, 1./24.);
         ir->SetOrder(2);
         TetrahedronIntRules[2] = ir;
         return ir;

      case 3:  // 5 points - degree 3 (negative weight)
         ir = new IntegrationRule(5);
         ir->AddTetMidPoint(0, -2./15.);
         ir->AddTetPoints4b(1, 0.5, 0.075);
         ir->SetOrder(3);
         TetrahedronIntRules[3] = ir;
         return ir;

      case 4:  // 11 points - degree 4 (negative weight)
         ir = new IntegrationRule(11);
         ir->AddTetPoints4(0, 1./14., 343./45000.);
         ir->AddTetMidPoint(4, -74./5625.);
         ir->AddTetPoints6(5, 0.10059642383320079500, 28./1125.);
         ir->SetOrder(4);
         TetrahedronIntRules[4] = ir;
         return ir;

      case 5:  // 14 points - degree 5
         ir = new IntegrationRule(14);
         ir->AddTetPoints6(0, 0.045503704125649649492,
                           7.0910034628469110730E-03);
         ir->AddTetPoints4(6, 0.092735250310891226402, 0.012248840519393658257);
         ir->AddTetPoints4b(10, 0.067342242210098170608,
                            0.018781320953002641800);
         ir->SetOrder(5);
         TetrahedronIntRules[5] = ir;
         return ir;

      case 6:  // 24 points - degree 6
         ir = new IntegrationRule(24);
         ir->AddTetPoints4(0, 0.21460287125915202929,
                           6.6537917096945820166E-03);
         ir->AddTetPoints4(4, 0.040673958534611353116,
                           1.6795351758867738247E-03);
         ir->AddTetPoints4b(8, 0.032986329573173468968,
                            9.2261969239424536825E-03);
         ir->AddTetPoints12(12, 0.063661001875017525299, 0.26967233145831580803,
                            8.0357142857142857143E-03);
         ir->SetOrder(6);
         TetrahedronIntRules[6] = ir;
         return ir;

      case 7:  // 31 points - degree 7 (negative weight)
         ir = new IntegrationRule(31);
         ir->AddTetPoints6(0, 0.0, 9.7001763668430335097E-04);
         ir->AddTetMidPoint(6, 0.018264223466108820291);
         ir->AddTetPoints4(7, 0.078213192330318064374, 0.010599941524413686916);
         ir->AddTetPoints4(11, 0.12184321666390517465,
                           -0.062517740114331851691);
         ir->AddTetPoints4b(15, 2.3825066607381275412E-03,
                            4.8914252630734993858E-03);
         ir->AddTetPoints12(19, 0.1, 0.2, 0.027557319223985890653);
         ir->SetOrder(7);
         TetrahedronIntRules[7] = ir;
         return ir;

      case 8:  // 43 points - degree 8 (negative weight)
         ir = new IntegrationRule(43);
         ir->AddTetPoints4(0, 5.7819505051979972532E-03,
                           1.6983410909288737984E-04);
         ir->AddTetPoints4(4, 0.082103588310546723091,
                           1.9670333131339009876E-03);
         ir->AddTetPoints12(8, 0.036607749553197423679, 0.19048604193463345570,
                            2.1405191411620925965E-03);
         ir->AddTetPoints6(20, 0.050532740018894224426,
                           4.5796838244672818007E-03);
         ir->AddTetPoints12(26, 0.22906653611681113960, 0.035639582788534043717,
                            5.7044858086819185068E-03);
         ir->AddTetPoints4(38, 0.20682993161067320408, 0.014250305822866901248);
         ir->AddTetMidPoint(42, -0.020500188658639915841);
         ir->SetOrder(8);
         TetrahedronIntRules[8] = ir;
         return ir;

      case 9: // orders 9 and higher -- Grundmann-Moller rules
         ir = new IntegrationRule;
         ir->GrundmannMollerSimplexRule(4,3);
         TetrahedronIntRules[9] = ir;
         return ir;

      default: // Grundmann-Moller rules
         int i = (Order / 2) * 2 + 1;   // Get closest odd # >= Order
         AllocIntRule(TetrahedronIntRules, i);
         ir = new IntegrationRule;
         ir->GrundmannMollerSimplexRule(i/2,3);
         TetrahedronIntRules[i-1] = TetrahedronIntRules[i] = ir;
         return ir;
   }
}

// Integration rules for reference pyramid
IntegrationRule *IntegrationRules::PyramidIntegrationRule(int Order)
{
   // This is a simple integration rule adapted from an integration
   // rule for a cube which seems to be adequate for now. When we
   // implement high order finite elements for pyramids we should
   // revisit this and see if we can improve upon it.
   const IntegrationRule &irc = Get(Geometry::CUBE, Order);
   int npts = irc.GetNPoints();
   AllocIntRule(PyramidIntRules, Order);
   PyramidIntRules[Order] = new IntegrationRule(npts);
   PyramidIntRules[Order]->SetOrder(Order); // FIXME: see comment above

   for (int k=0; k<npts; k++)
   {
      const IntegrationPoint &ipc = irc.IntPoint(k);
      IntegrationPoint &ipp = PyramidIntRules[Order]->IntPoint(k);
      ipp.x = ipc.x * (1.0 - ipc.z);
      ipp.y = ipc.y * (1.0 - ipc.z);
      ipp.z = ipc.z;
      ipp.weight = ipc.weight / 3.0;
   }
   return PyramidIntRules[Order];
}

// Integration rules for reference prism
IntegrationRule *IntegrationRules::PrismIntegrationRule(int Order)
{
   const IntegrationRule &irt = Get(Geometry::TRIANGLE, Order);
   const IntegrationRule &irs = Get(Geometry::SEGMENT, Order);
   int nt = irt.GetNPoints();
   int ns = irs.GetNPoints();
   AllocIntRule(PrismIntRules, Order);
   PrismIntRules[Order] = new IntegrationRule(nt * ns);
   PrismIntRules[Order]->SetOrder(std::min(irt.GetOrder(), irs.GetOrder()));
   while (Order < std::min(irt.GetOrder(), irs.GetOrder()))
   {
      AllocIntRule(PrismIntRules, ++Order);
      PrismIntRules[Order] = PrismIntRules[Order-1];
   }

   for (int ks=0; ks<ns; ks++)
   {
      const IntegrationPoint &ips = irs.IntPoint(ks);
      for (int kt=0; kt<nt; kt++)
      {
         int kp = ks * nt + kt;
         const IntegrationPoint &ipt = irt.IntPoint(kt);
         IntegrationPoint &ipp = PrismIntRules[Order]->IntPoint(kp);
         ipp.x = ipt.x;
         ipp.y = ipt.y;
         ipp.z = ips.x;
         ipp.weight = ipt.weight * ips.weight;
      }
   }
   return PrismIntRules[Order];
}

// Integration rules for reference cube
IntegrationRule *IntegrationRules::CubeIntegrationRule(int Order)
{
   int RealOrder = GetSegmentRealOrder(Order);
   if (!HaveIntRule(SegmentIntRules, RealOrder))
   {
      SegmentIntegrationRule(RealOrder);
   }
   AllocIntRule(CubeIntRules, RealOrder);
   CubeIntRules[RealOrder-1] =
      CubeIntRules[RealOrder] =
         new IntegrationRule(*SegmentIntRules[RealOrder],
                             *SegmentIntRules[RealOrder],
                             *SegmentIntRules[RealOrder]);
   return CubeIntRules[Order];
}

IntegrationRule& NURBSMeshRules::GetElementRule(const int elem,
                                                const int patch, const int *ijk,
                                                Array<const KnotVector*> const& kv,
                                                bool & deleteRule) const
{
   deleteRule = false;

   // First check whether a rule has been assigned to element index elem.
   auto search = elementToRule.find(elem);
   if (search != elementToRule.end())
   {
      return *elementRule[search->second];
   }

   MFEM_VERIFY(patchRules1D.NumRows(),
               "Undefined rule in NURBSMeshRules::GetElementRule");

   // Use a tensor product of rules on the patch.
   MFEM_VERIFY(kv.Size() == dim, "");

   int np = 1;
   std::vector<std::vector<real_t>> el(dim);

   std::vector<int> npd;
   npd.assign(3, 0);

   for (int d=0; d<dim; ++d)
   {
      const int order = kv[d]->GetOrder();

      const real_t kv0 = (*kv[d])[order + ijk[d]];
      const real_t kv1 = (*kv[d])[order + ijk[d] + 1];

      const bool rightEnd = (order + ijk[d] + 1) == (kv[d]->Size() - 1);

      for (int i=0; i<patchRules1D(patch,d)->Size(); ++i)
      {
         const IntegrationPoint& ip = (*patchRules1D(patch,d))[i];
         if (kv0 <= ip.x && (ip.x < kv1 || rightEnd))
         {
            const real_t x = (ip.x - kv0) / (kv1 - kv0);
            el[d].push_back(x);
            el[d].push_back(ip.weight);
         }
      }

      npd[d] = static_cast<int>(el[d].size() / 2);
      np *= npd[d];
   }

   IntegrationRule *irp = new IntegrationRule(np);
   deleteRule = true;

   // Set (*irp)[i + j*npd[0] + k*npd[0]*npd[1]] =
   //     (el[0][2*i], el[1][2*j], el[2][2*k])

   MFEM_VERIFY(npd[0] > 0 && npd[1] > 0, "Assuming 2D or 3D");

   for (int i = 0; i < npd[0]; ++i)
   {
      for (int j = 0; j < npd[1]; ++j)
      {
         for (int k = 0; k < std::max(npd[2], 1); ++k)
         {
            const int id = i + j*npd[0] + k*npd[0]*npd[1];
            (*irp)[id].x = el[0][2*i];
            (*irp)[id].y = el[1][2*j];

            (*irp)[id].weight = el[0][(2*i)+1];
            (*irp)[id].weight *= el[1][(2*j)+1];

            if (npd[2] > 0)
            {
               (*irp)[id].z = el[2][2*k];
               (*irp)[id].weight *= el[2][(2*k)+1];
            }
         }
      }
   }

   return *irp;
}

void NURBSMeshRules::GetIntegrationPointFrom1D(const int patch, int i, int j,
                                               int k, IntegrationPoint & ip)
{
   MFEM_VERIFY(patchRules1D.NumRows() > 0,
               "Assuming patchRules1D is set.");

   ip.weight = (*patchRules1D(patch,0))[i].weight;
   ip.x = (*patchRules1D(patch,0))[i].x;

   if (dim > 1)
   {
      ip.weight *= (*patchRules1D(patch,1))[j].weight;
      ip.y = (*patchRules1D(patch,1))[j].x;  // 1D rule only has x
   }

   if (dim > 2)
   {
      ip.weight *= (*patchRules1D(patch,2))[k].weight;
      ip.z = (*patchRules1D(patch,2))[k].x;  // 1D rule only has x
   }
}

void NURBSMeshRules::Finalize(Mesh const& mesh)
{
   if ((int) pointToElem.size() == npatches) { return; }  // Already set

   MFEM_VERIFY(elementToRule.empty() && patchRules1D.NumRows() > 0
               && npatches > 0, "Assuming patchRules1D is set.");
   MFEM_VERIFY(mesh.NURBSext, "");
   MFEM_VERIFY(mesh.Dimension() == dim, "");

   pointToElem.resize(npatches);
   patchRules1D_KnotSpan.resize(npatches);

   // First, find all the elements in each patch.
   std::vector<std::vector<int>> patchElements(npatches);

   for (int e=0; e<mesh.GetNE(); ++e)
   {
      patchElements[mesh.NURBSext->GetElementPatch(e)].push_back(e);
   }

   Array<int> ijk(3);
   Array<int> maxijk(3);
   Array<int> np(3);  // Number of points in each dimension
   ijk = 0;

   Array<const KnotVector*> pkv;

   for (int p=0; p<npatches; ++p)
   {
      patchRules1D_KnotSpan[p].resize(dim);

      // For each patch, get the range of ijk.
      mesh.NURBSext->GetPatchKnotVectors(p, pkv);
      MFEM_VERIFY((int) pkv.Size() == dim, "");

      maxijk = 1;
      np = 1;
      for (int d=0; d<dim; ++d)
      {
         maxijk[d] = pkv[d]->GetNKS();
         np[d] = patchRules1D(p,d)->Size();
      }

      // For each patch, set a map from ijk to element index.
      Array3D<int> ijk2elem(maxijk[0], maxijk[1], maxijk[2]);
      ijk2elem = -1;

      for (auto elem : patchElements[p])
      {
         mesh.NURBSext->GetElementIJK(elem, ijk);
         MFEM_VERIFY(ijk2elem(ijk[0], ijk[1], ijk[2]) == -1, "");
         ijk2elem(ijk[0], ijk[1], ijk[2]) = elem;
      }

      // For each point, find its ijk and from that its element index.
      // It is assumed here that the NURBSFiniteElement kv the same as the
      // patch kv.

      for (int d=0; d<dim; ++d)
      {
         patchRules1D_KnotSpan[p][d].SetSize(patchRules1D(p,d)->Size());

         for (int r=0; r<patchRules1D(p,d)->Size(); ++r)
         {
            const IntegrationPoint& ip = (*patchRules1D(p,d))[r];

            const int order = pkv[d]->GetOrder();

            // Find ijk_d such that ip.x is in the corresponding knot-span.
            int ijk_d = 0;
            bool found = false;
            while (!found)
            {
               const real_t kv0 = (*pkv[d])[order + ijk_d];
               const real_t kv1 = (*pkv[d])[order + ijk_d + 1];

               const bool rightEnd = (order + ijk_d + 1) == (pkv[d]->Size() - 1);

               if (kv0 <= ip.x && (ip.x < kv1 || rightEnd))
               {
                  found = true;
               }
               else
               {
                  ijk_d++;
               }
            }

            patchRules1D_KnotSpan[p][d][r] = ijk_d;
         }
      }

      pointToElem[p].SetSize(np[0], np[1], np[2]);
      for (int i=0; i<np[0]; ++i)
         for (int j=0; j<np[1]; ++j)
            for (int k=0; k<np[2]; ++k)
            {
               const int elem = ijk2elem(patchRules1D_KnotSpan[p][0][i],
                                         patchRules1D_KnotSpan[p][1][j],
                                         patchRules1D_KnotSpan[p][2][k]);
               MFEM_VERIFY(elem >= 0, "");
               pointToElem[p](i,j,k) = elem;
            }
   } // Loop (p) over patches
}

void NURBSMeshRules::SetPatchRules1D(const int patch,
                                     std::vector<const IntegrationRule*> & ir1D)
{
   MFEM_VERIFY((int) ir1D.size() == dim, "Wrong dimension");

   for (int i=0; i<dim; ++i)
   {
      patchRules1D(patch,i) = ir1D[i];
   }
}

NURBSMeshRules::~NURBSMeshRules()
{
   for (int i=0; i<patchRules1D.NumRows(); ++i)
      for (int j=0; j<patchRules1D.NumCols(); ++j)
      {
         delete patchRules1D(i, j);
      }
}

}
