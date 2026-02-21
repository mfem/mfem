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

// Implementation of IntegrationRule(s) classes

// Acknowledgment: Some of the high-precision triangular and tetrahedral
// quadrature rules below were obtained from the Encyclopaedia of Cubature
// Formulas at http://nines.cs.kuleuven.be/research/ecf/ecf.html
//
// Witherden-Vincent quadrature rules:
//   F.D. Witherden, P.E. Vincent, "On the identification of symmetric
//   quadrature rules for finite element methods", Computers & Mathematics
//   with Applications, 69(10):1232-1241, 2015.
//   Data from PyFR (https://pyfr.org), CC-BY licensed.

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
      case ClosedGL:
         return type; // all types can work as open
      default:
         return Invalid;
   }
}


IntegrationRules IntRules(0, Quadrature1D::GaussLegendre);

IntegrationRules RefinedIntRules(1, Quadrature1D::GaussLegendre);

IntegrationRules::IntegrationRules(int ref, int type, SimplexQuadrature stype)
   : quad_type(type), simplex_type(stype)
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
   if (simplex_type == SimplexQuadrature::WitherdenVincent)
   {
      return WVTriangleIntegrationRule(Order);
   }

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
   if (simplex_type == SimplexQuadrature::WitherdenVincent)
   {
      return WVTetrahedronIntegrationRule(Order);
   }

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

IntegrationRule *IntegrationRules::WVTriangleIntegrationRule(int Order)
{
   IntegrationRule *ir = NULL;

   switch (Order)
   {
      case 0:
      case 1:
         ir = new IntegrationRule(1);
         ir->AddTriMidPoint(0, 5.00000000000000000000e-01);
         ir->SetOrder(1);
         TriangleIntRules[0] =
            TriangleIntRules[1] = ir;
         return ir;

      case 2:
         ir = new IntegrationRule(3);
         ir->AddTriPoints3(0, 1.66666666666666740682e-01, 1.66666666666666657415e-01);
         ir->SetOrder(2);
         TriangleIntRules[2] = ir;
         return ir;

      case 3:
      case 4:
         ir = new IntegrationRule(6);
         ir->AddTriPoints3(0, 4.45948490915964890213e-01, 1.11690794839005735906e-01);
         ir->AddTriPoints3(3, 9.15762135097707430376e-02, 5.49758718276609353870e-02);
         ir->SetOrder(4);
         TriangleIntRules[3] =
            TriangleIntRules[4] = ir;
         return ir;

      case 5:
         ir = new IntegrationRule(7);
         ir->AddTriMidPoint(0, 1.12500000000000002776e-01);
         ir->AddTriPoints3(1, 1.01286507323456342888e-01, 6.29695902724135697648e-02);
         ir->AddTriPoints3(4, 4.70142064105115109474e-01, 6.61970763942530959767e-02);
         ir->SetOrder(5);
         TriangleIntRules[5] = ir;
         return ir;

      case 6:
         ir = new IntegrationRule(12);
         ir->AddTriPoints3(0, 6.30890144915022266225e-02, 2.54224531851034094010e-02);
         ir->AddTriPoints3(3, 2.49286745170910428726e-01, 5.83931378631896841336e-02);
         ir->AddTriPoints6(6, 6.36502499121398668258e-01, 3.10352451033784393353e-01,
                           4.14255378091867854096e-02);
         ir->SetOrder(6);
         TriangleIntRules[6] = ir;
         return ir;

      case 7:
         ir = new IntegrationRule(15);
         ir->AddTriPoints3(0, 3.37306485545878498300e-02, 8.27252505539606552976e-03);
         ir->AddTriPoints3(3, 2.41577382595403566956e-01, 6.39720856150777922311e-02);
         ir->AddTriPoints3(6, 4.74309692504718327655e-01, 3.85433230929930342734e-02);
         ir->AddTriPoints6(9, 7.54280040550053154647e-01, 1.98683314797351684433e-01,
                           2.79393664515998896292e-02);
         ir->SetOrder(7);
         TriangleIntRules[7] = ir;
         return ir;

      case 8:
         ir = new IntegrationRule(16);
         ir->AddTriMidPoint(0, 7.21578038388935860681e-02);
         ir->AddTriPoints3(1, 4.59292588292723236165e-01, 4.75458171336423096598e-02);
         ir->AddTriPoints3(4, 1.70569307751760268488e-01, 5.16086852673591223173e-02);
         ir->AddTriPoints3(7, 5.05472283170309566458e-02, 1.62292488115990396480e-02);
         ir->AddTriPoints6(10, 7.28492392955404244326e-01, 2.63112829634638112353e-01,
                           1.36151570872174963733e-02);
         ir->SetOrder(8);
         TriangleIntRules[8] = ir;
         return ir;

      case 9:
         ir = new IntegrationRule(19);
         ir->AddTriMidPoint(0, 4.85678981413994181882e-02);
         ir->AddTriPoints3(1, 4.37089591492936690997e-01, 3.89137705023871391385e-02);
         ir->AddTriPoints3(4, 1.88203535619032802373e-01, 3.98238694636051243636e-02);
         ir->AddTriPoints3(7, 4.89682519198737620236e-01, 1.56673501135695357467e-02);
         ir->AddTriPoints3(10, 4.47295133944527467662e-02, 1.27888378293490156262e-02);
         ir->AddTriPoints6(13, 7.41198598784498008385e-01, 2.21962989160765733487e-01,
                           2.16417696886446880855e-02);
         ir->SetOrder(9);
         TriangleIntRules[9] = ir;
         return ir;

      case 10:
         ir = new IntegrationRule(25);
         ir->AddTriMidPoint(0, 4.08716645731429864541e-02);
         ir->AddTriPoints3(1, 3.20553732169435168231e-02, 6.67648440657478327992e-03);
         ir->AddTriPoints3(4, 1.42161101056564431744e-01, 2.29789818023723654838e-02);
         ir->AddTriPoints6(7, 5.30054118927343997925e-01, 3.21812995288835446139e-01,
                           3.19524531982120219009e-02);
         ir->AddTriPoints6(13, 6.01233328683459244957e-01, 3.69146781827810910315e-01,
                           1.70923240814797143539e-02);
         ir->AddTriPoints6(19, 8.07930600922879049719e-01, 1.63701733737182442141e-01,
                           1.26488788536441923438e-02);
         ir->SetOrder(10);
         TriangleIntRules[10] = ir;
         return ir;

      case 11:
         ir = new IntegrationRule(28);
         ir->AddTriMidPoint(0, 4.28805898661121093207e-02);
         ir->AddTriPoints3(1, 2.84854176143718995640e-02, 5.21593525644734826857e-03);
         ir->AddTriPoints3(4, 2.10219956703178278978e-01, 3.52578420558582877886e-02);
         ir->AddTriPoints3(7, 1.02635482712246428605e-01, 1.93153796185096607307e-02);
         ir->AddTriPoints3(10, 4.95891900965890919384e-01, 8.30313652729268436570e-03);
         ir->AddTriPoints3(13, 4.38465926764352253997e-01, 3.36580770397341480504e-02);
         ir->AddTriPoints6(16, 8.43349783661853091843e-01, 1.49324788652082374174e-01,
                           5.14514478647663895533e-03);
         ir->AddTriPoints6(22, 6.64408374196864159877e-01, 2.89581125637705882880e-01,
                           2.01662383202502772106e-02);
         ir->SetOrder(11);
         TriangleIntRules[11] = ir;
         return ir;

      case 12:
         ir = new IntegrationRule(33);
         ir->AddTriPoints3(0, 4.88203750945541581352e-01, 1.21334190407260157640e-02);
         ir->AddTriPoints3(3, 1.09257827659354322947e-01, 1.42430260344387719235e-02);
         ir->AddTriPoints3(6, 2.71462507014926135440e-01, 3.12706065979513822550e-02);
         ir->AddTriPoints3(9, 2.46463634363356387524e-02, 3.96582125498681943576e-03);
         ir->AddTriPoints3(12, 4.40111648658593201944e-01, 2.49591674640304711508e-02);
         ir->AddTriPoints6(15, 6.85310163906391878186e-01, 2.91655679738340944951e-01,
                           1.08917925193037796322e-02);
         ir->AddTriPoints6(21, 6.28249751683556123538e-01, 2.55454228638517299999e-01,
                           2.16136818297071042760e-02);
         ir->AddTriPoints6(27, 8.51337792510240110033e-01, 1.27279717233589384495e-01,
                           7.54183878825571887144e-03);
         ir->SetOrder(12);
         TriangleIntRules[12] = ir;
         return ir;

      case 13:
         ir = new IntegrationRule(37);
         ir->AddTriMidPoint(0, 3.39800182934158201409e-02);
         ir->AddTriPoints3(1, 4.89076946452539351728e-01, 1.19972009644473652512e-02);
         ir->AddTriPoints3(4, 2.21372286291832920391e-01, 2.91392425595999905730e-02);
         ir->AddTriPoints3(7, 4.26941414259800422482e-01, 2.78009837652266646180e-02);
         ir->AddTriPoints3(10, 2.15096811088433259584e-02, 3.02616855176958583773e-03);
         ir->AddTriPoints6(13, 7.48507115899952224503e-01, 1.63597401067850478640e-01,
                           1.20895199057969096601e-02);
         ir->AddTriPoints6(19, 8.64707770295442768038e-01, 1.10922042803463405392e-01,
                           7.48270055258283377231e-03);
         ir->AddTriPoints6(25, 6.23545995553675513889e-01, 3.08441760892117777804e-01,
                           1.73206380704241866275e-02);
         ir->AddTriPoints6(31, 7.22357793124188019007e-01, 2.72515817773429591675e-01,
                           4.79534050177163155559e-03);
         ir->SetOrder(13);
         TriangleIntRules[13] = ir;
         return ir;

      case 14:
         ir = new IntegrationRule(42);
         ir->AddTriPoints3(0, 1.77205532412543442788e-01, 2.10812943684965080349e-02);
         ir->AddTriPoints3(3, 4.17644719340453940415e-01, 1.63941767720626740967e-02);
         ir->AddTriPoints3(6, 6.17998830908725871325e-02, 7.21684983488833382143e-03);
         ir->AddTriPoints3(9, 4.88963910362178677538e-01, 1.09417906847144447147e-02);
         ir->AddTriPoints3(12, 2.73477528308838646609e-01, 2.58870522536457925433e-02);
         ir->AddTriPoints3(15, 1.93909612487010996063e-02, 2.46170180120004094063e-03);
         ir->AddTriPoints6(18, 6.86980167808087793802e-01, 2.98372882136257788765e-01,
                           7.21815405676692022074e-03);
         ir->AddTriPoints6(24, 7.70608554774996457049e-01, 1.72266687821355679588e-01,
                           1.23328766062818367955e-02);
         ir->AddTriPoints6(30, 5.70222290846683188548e-01, 3.36861459796344964168e-01,
                           1.92857553935303419057e-02);
         ir->AddTriPoints6(36, 8.79757171370171064950e-01, 1.18974497696956893478e-01,
                           2.50511441925033596229e-03);
         ir->SetOrder(14);
         TriangleIntRules[14] = ir;
         return ir;

      case 15:
         ir = new IntegrationRule(49);
         ir->AddTriMidPoint(0, 2.21676936910920364954e-02);
         ir->AddTriPoints3(1, 4.05362214133975495844e-01, 2.13568907857302828224e-02);
         ir->AddTriPoints3(4, 7.01735528999860580512e-02, 8.22236878131258133728e-03);
         ir->AddTriPoints3(7, 4.74170681438019769871e-01, 8.69807400038170690226e-03);
         ir->AddTriPoints3(10, 2.26378713420349653163e-01, 2.33916808643548149171e-02);
         ir->AddTriPoints3(13, 4.94996956769126195130e-01, 4.78692309123004283711e-03);
         ir->AddTriPoints3(16, 1.58117262509887002153e-02, 1.48038731895268772104e-03);
         ir->AddTriPoints6(19, 6.66975644801868106093e-01, 3.14648242812450851247e-01,
                           7.80128641528798211224e-03);
         ir->AddTriPoints6(25, 9.19912157726236134891e-01, 7.09486052364554087291e-02,
                           2.01492668600904969653e-03);
         ir->AddTriPoints6(31, 7.15222356931450642392e-01, 1.90535589476393929509e-01,
                           1.43602934626006709801e-02);
         ir->AddTriPoints6(37, 8.13292641049419229304e-01, 1.68068645222414381202e-01,
                           5.83631059078792285844e-03);
         ir->AddTriPoints6(43, 5.65252664877114230357e-01, 3.38950611475277163720e-01,
                           1.56577381424846430458e-02);
         ir->SetOrder(15);
         TriangleIntRules[15] = ir;
         return ir;

      case 16:
         ir = new IntegrationRule(55);
         ir->AddTriMidPoint(0, 2.26322830369093952463e-02);
         ir->AddTriPoints3(1, 2.45990070467141719313e-01, 2.05464615718494759966e-02);
         ir->AddTriPoints3(4, 4.15584896885420551627e-01, 2.03559166562126796218e-02);
         ir->AddTriPoints3(7, 8.53555665867003487968e-02, 7.39081734511220188322e-03);
         ir->AddTriPoints3(10, 1.61918644191271221544e-01, 1.47092048494940497855e-02);
         ir->AddTriPoints3(13, 5.00000000000000000000e-01, 2.20927315607528452004e-03);
         ir->AddTriPoints3(16, 4.75280727545942083268e-01, 1.29871666491385793357e-02);
         ir->AddTriPoints6(19, 7.54170061444767725334e-01, 1.91074763640529221576e-01,
                           9.46913623220784969603e-03);
         ir->AddTriPoints6(25, 9.68244368030958701965e-01, 2.32034277688137335893e-02,
                           8.27233357417524097638e-04);
         ir->AddTriPoints6(31, 6.49303698245446425652e-01, 3.31764523474147643434e-01,
                           7.50430089214290316213e-03);
         ir->AddTriPoints6(37, 9.00273703270429548340e-01, 8.06961669858730079596e-02,
                           3.97379696669624901673e-03);
         ir->AddTriPoints6(43, 5.89148840564247877616e-01, 3.08244969196354023921e-01,
                           1.59918050396850343342e-02);
         ir->AddTriPoints6(49, 8.06621867499395683865e-01, 1.87441782483782071189e-01,
                           2.69559355842440570919e-03);
         ir->SetOrder(16);
         TriangleIntRules[16] = ir;
         return ir;

      case 17:
         ir = new IntegrationRule(60);
         ir->AddTriPoints3(0, 4.17103444361599295931e-01, 1.36554632640510532210e-02);
         ir->AddTriPoints3(3, 1.47554916607539610141e-02, 1.38694378881882109979e-03);
         ir->AddTriPoints3(6, 4.65597871618890324363e-01, 1.25097254752486782697e-02);
         ir->AddTriPoints3(9, 1.80358116266370605008e-01, 1.31563152940089925225e-02);
         ir->AddTriPoints3(12, 6.66540634795969033632e-02, 6.22950040115272107855e-03);
         ir->AddTriPoints3(15, 2.85706502436586629035e-01, 1.88581185763976415248e-02);
         ir->AddTriPoints6(18, 8.24790070165088096132e-01, 1.59192287472792681768e-01,
                           3.98915010296479674579e-03);
         ir->AddTriPoints6(24, 6.26369030386452196879e-01, 3.06281591746186521164e-01,
                           1.12438862733455335191e-02);
         ir->AddTriPoints6(30, 5.71294867944684092720e-01, 4.15475459295228999324e-01,
                           5.19921997791976831654e-03);
         ir->AddTriPoints6(36, 7.53235145936458128091e-01, 1.68722513495259462957e-01,
                           1.02789491602272593102e-02);
         ir->AddTriPoints6(42, 7.15072259110642427515e-01, 2.71791870055354878311e-01,
                           4.34610725050059605590e-03);
         ir->AddTriPoints6(48, 9.15919353297816929427e-01, 7.25054707990024915887e-02,
                           2.29217420086793351869e-03);
         ir->AddTriPoints6(54, 5.43275579596159796658e-01, 2.99218942476970228839e-01,
                           1.30858129676684944304e-02);
         ir->SetOrder(17);
         TriangleIntRules[17] = ir;
         return ir;

      case 18:
         ir = new IntegrationRule(67);
         ir->AddTriMidPoint(0, 1.81778676507133342410e-02);
         ir->AddTriPoints3(1, 3.99955628067576229867e-01, 1.66522350166950668104e-02);
         ir->AddTriPoints3(4, 4.87580301574869645620e-01, 6.02332381699985548729e-03);
         ir->AddTriPoints3(7, 4.61809506406449243876e-01, 9.47458575338943308208e-03);
         ir->AddTriPoints3(10, 2.42264702514271956790e-01, 1.82375447044718190515e-02);
         ir->AddTriPoints3(13, 3.88302560886856218403e-02, 3.56466300985948522304e-03);
         ir->AddTriPoints3(16, 9.19477421216432500017e-02, 8.27957997600162372287e-03);
         ir->AddTriPoints6(19, 7.70372376214675247397e-01, 1.83822707925463957324e-01,
                           6.87980811747110256732e-03);
         ir->AddTriPoints6(25, 6.70953985194234547862e-01, 2.06349257433837918185e-01,
                           1.18909554500764153007e-02);
         ir->AddTriPoints6(31, 6.00418954634256873959e-01, 3.95683434332269712286e-01,
                           2.26526725112853252742e-03);
         ir->AddTriPoints6(37, 8.78342189467521738955e-01, 1.08195793791033278985e-01,
                           3.42005505980359086893e-03);
         ir->AddTriPoints6(43, 6.39988092004714625993e-01, 3.19751624525377309283e-01,
                           8.87374455101020212511e-03);
         ir->AddTriPoints6(49, 7.58929479855198430016e-01, 2.35772184958191743931e-01,
                           2.50533043728986106261e-03);
         ir->AddTriPoints6(55, 9.72360728962795684005e-01, 2.70909109951620319379e-02,
                           6.11474063480544911126e-04);
         ir->AddTriPoints6(61, 5.45918775386194599086e-01, 3.33493529449880754534e-01,
                           1.27410876559122202695e-02);
         ir->SetOrder(18);
         TriangleIntRules[18] = ir;
         return ir;

      case 19:
         ir = new IntegrationRule(73);
         ir->AddTriMidPoint(0, 1.72346988520061666916e-02);
         ir->AddTriPoints3(1, 5.25238903512089683190e-02, 3.55462829889906543543e-03);
         ir->AddTriPoints3(4, 4.92512675041336889237e-01, 5.16087757147214078873e-03);
         ir->AddTriPoints3(7, 1.11448873323021391268e-01, 7.61717554650914990128e-03);
         ir->AddTriPoints3(10, 4.59194201039543670184e-01, 1.14917950133708035576e-02);
         ir->AddTriPoints3(13, 4.03969722551901222474e-01, 1.57687674465774863020e-02);
         ir->AddTriPoints3(16, 1.78170104781764315760e-01, 1.23259574240954274116e-02);
         ir->AddTriPoints3(19, 1.16394611837894457196e-02, 8.82661388221423837477e-04);
         ir->AddTriPoints3(22, 2.55161632913607716588e-01, 1.58765096830015377261e-02);
         ir->AddTriPoints6(25, 8.30156464400275351245e-01, 1.30697676268032414448e-01,
                           4.84774224342752330097e-03);
         ir->AddTriPoints6(31, 5.59369805720300927732e-01, 3.11317629809541251973e-01,
                           1.31731609886953666272e-02);
         ir->AddTriPoints6(37, 6.33313293128784149388e-01, 3.64617780974611060962e-01,
                           1.64103827591790965915e-03);
         ir->AddTriPoints6(43, 7.04004819966042139079e-01, 2.21434885432331141075e-01,
                           9.05397246560622585843e-03);
         ir->AddTriPoints6(49, 8.52566954376889230005e-01, 1.42425757365756355810e-01,
                           1.46315755173510018451e-03);
         ir->AddTriPoints6(55, 6.05083979068707922266e-01, 3.54028009735275261960e-01,
                           8.05108138201205379703e-03);
         ir->AddTriPoints6(61, 7.43181368957436361278e-01, 2.41894578960579587079e-01,
                           4.22794374976824798712e-03);
         ir->AddTriPoints6(67, 9.30137698876805085746e-01, 6.00862753223067036501e-02,
                           1.66360068142969402642e-03);
         ir->SetOrder(19);
         TriangleIntRules[19] = ir;
         return ir;

      case 20:
         ir = new IntegrationRule(79);
         ir->AddTriMidPoint(0, 1.39101107014531159140e-02);
         ir->AddTriPoints3(1, 2.54579267673339160183e-01, 1.40832013075202471669e-02);
         ir->AddTriPoints3(4, 1.09761410283977789426e-02, 7.98840791066619858654e-04);
         ir->AddTriPoints3(7, 1.09383596711714603522e-01, 7.83023077607453328597e-03);
         ir->AddTriPoints3(10, 1.86294997744540946627e-01, 9.17346297425291473671e-03);
         ir->AddTriPoints3(13, 4.45551056955924895675e-01, 9.45239993323244813428e-03);
         ir->AddTriPoints3(16, 3.73108805988847103130e-02, 2.16127541066557732688e-03);
         ir->AddTriPoints3(19, 3.93425347817099924086e-01, 1.37880506290704585304e-02);
         ir->AddTriPoints3(22, 4.76245611540499047543e-01, 7.10182530340844071076e-03);
         ir->AddTriPoints6(25, 8.33295511838236246938e-01, 1.59133707657067247077e-01,
                           2.20289741855849742491e-03);
         ir->AddTriPoints6(31, 7.54921502863547422280e-01, 1.98518132228788335425e-01,
                           5.98639857895469015836e-03);
         ir->AddTriPoints6(37, 9.31054476783942153162e-01, 6.40905856084340586065e-02,
                           1.12986960212586558597e-03);
         ir->AddTriPoints6(43, 6.11877703547425655373e-01, 3.33134817309587605294e-01,
                           8.66722556721933289070e-03);
         ir->AddTriPoints6(49, 8.61684018936486717521e-01, 9.99522962881386756173e-02,
                           4.14571152761385782609e-03);
         ir->AddTriPoints6(55, 6.78165737889635522606e-01, 2.15607057390094447591e-01,
                           7.72260782209923009323e-03);
         ir->AddTriPoints6(61, 5.70144692890973359134e-01, 4.20023758816224113133e-01,
                           3.69568150025529782929e-03);
         ir->AddTriPoints6(67, 5.42331804172428100230e-01, 3.17860123835772001577e-01,
                           1.16917457318277372147e-02);
         ir->AddTriPoints6(73, 7.08681375720323636358e-01, 2.80581411423665327831e-01,
                           3.57820023845768515197e-03);
         ir->SetOrder(20);
         TriangleIntRules[20] = ir;
         return ir;

      default:
         // Grundmann-Moller fallback for orders beyond WV tables
         int i = (Order / 2) * 2 + 1;   // closest odd >= Order
         AllocIntRule(TriangleIntRules, i);
         ir = new IntegrationRule;
         ir->GrundmannMollerSimplexRule(i/2, 2);
         TriangleIntRules[i-1] = TriangleIntRules[i] = ir;
         return ir;
   }
}

IntegrationRule *IntegrationRules::WVTetrahedronIntegrationRule(int Order)
{
   IntegrationRule *ir = NULL;

   switch (Order)
   {
      case 0:
      case 1:
         ir = new IntegrationRule(1);
         ir->AddTetMidPoint(0, 1.66666666666666657415e-01);
         ir->SetOrder(1);
         TetrahedronIntRules[0] =
            TetrahedronIntRules[1] = ir;
         return ir;

      case 2:
         ir = new IntegrationRule(4);
         ir->AddTetPoints4(0, 1.38196601125010531952e-01, 4.16666666666666643537e-02);
         ir->SetOrder(2);
         TetrahedronIntRules[2] = ir;
         return ir;

      case 3:
         ir = new IntegrationRule(8);
         ir->AddTetPoints4(0, 3.28163302516381705232e-01, 2.27029737561812265667e-02);
         ir->AddTetPoints4(4, 1.08047249898428621151e-01, 1.89636929104854412564e-02);
         ir->SetOrder(3);
         TetrahedronIntRules[3] = ir;
         return ir;

      case 4:
      case 5:
         ir = new IntegrationRule(14);
         ir->AddTetPoints4(0, 3.10885919263300669613e-01, 1.87813209530026427319e-02);
         ir->AddTetPoints4(4, 9.27352503108912484819e-02, 1.22488405193936587129e-02);
         ir->AddTetPoints6(8, 4.54496295874350364485e-01, 7.09100346284691120807e-03);
         ir->SetOrder(5);
         TetrahedronIntRules[4] =
            TetrahedronIntRules[5] = ir;
         return ir;

      case 6:
         ir = new IntegrationRule(24);
         ir->AddTetPoints4(0, 4.06739585346113652342e-02, 1.67953517588677390775e-03);
         ir->AddTetPoints4(4, 3.22337890142275540484e-01, 9.22619692394245453915e-03);
         ir->AddTetPoints4(8, 2.14602871259152117034e-01, 6.65379170969458179352e-03);
         ir->AddTetPoints12(12, 6.36610018750174977420e-02, 6.03005664791649187428e-01,
                            8.03571428571428492127e-03);
         ir->SetOrder(6);
         TetrahedronIntRules[6] = ir;
         return ir;

      case 7:
         ir = new IntegrationRule(35);
         ir->AddTetMidPoint(0, 1.59142149106884754628e-02);
         ir->AddTetPoints4(1, 3.15701149778202794227e-01, 7.05493020166117132397e-03);
         ir->AddTetPoints6(5, 4.49510177401603649994e-01, 5.31615463880959638471e-03);
         ir->AddTetPoints12(11, 1.88833831026001153219e-01, 5.75171637586999962011e-01,
                            6.20118845472243662709e-03);
         ir->AddTetPoints12(23, 2.12654725414832546093e-02, 8.10830241098548620826e-01,
                            1.35179513831722359664e-03);
         ir->SetOrder(7);
         TetrahedronIntRules[7] = ir;
         return ir;

      case 8:
         ir = new IntegrationRule(46);
         ir->AddTetPoints4(0, 1.07952724962210866444e-01, 4.40444181806813866292e-03);
         ir->AddTetPoints4(4, 1.85109487782586568105e-01, 8.67195792728975463348e-03);
         ir->AddTetPoints4(8, 4.23165436847673381848e-02, 1.25420935892336655841e-03);
         ir->AddTetPoints4(12, 3.14181709124039088010e-01, 6.96063047615581593358e-03);
         ir->AddTetPoints6(16, 4.35591328583830206256e-01, 6.04682171021813687217e-03);
         ir->AddTetPoints12(22, 2.14339301271305737728e-02, 7.17464063426308307214e-01,
                            1.19281714847407210867e-03);
         ir->AddTetPoints12(34, 2.04139333876029116510e-01, 5.83797378302144398532e-01,
                            2.57558102516005586052e-03);
         ir->SetOrder(8);
         TetrahedronIntRules[8] = ir;
         return ir;

      case 9:
         ir = new IntegrationRule(59);
         ir->AddTetMidPoint(0, 9.66842481874670943431e-03);
         ir->AddTetPoints4(1, 6.19817086544571793638e-10, 1.07198802932093984424e-05);
         ir->AddTetPoints4(5, 1.60774535395261597426e-01, 3.86222307707090968185e-03);
         ir->AddTetPoints4(9, 3.22276521821420969260e-01, 4.92715205590488116577e-03);
         ir->AddTetPoints4(13, 4.51089183454135844720e-02, 1.34399666326936377374e-03);
         ir->AddTetPoints6(17, 3.87703453995623947836e-01, 6.35568001728374458448e-03);
         ir->AddTetPoints12(23, 4.58871448752459276665e-01, 7.97025232620401369310e-02,
                            1.39740369971642539558e-03);
         ir->AddTetPoints12(35, 3.37758706853386048152e-02, 7.18350326442074527122e-01,
                            1.70575989212422133613e-03);
         ir->AddTetPoints12(47, 1.83641369809927956780e-01, 5.98301349801968918030e-01,
                            3.42081932799802312939e-03);
         ir->SetOrder(9);
         TetrahedronIntRules[9] = ir;
         return ir;

      case 10:
         ir = new IntegrationRule(81);
         ir->AddTetMidPoint(0, 7.89996225933678984654e-03);
         ir->AddTetPoints4(1, 3.12250068695188676138e-01, 4.48950999871145037950e-03);
         ir->AddTetPoints4(5, 1.14309653857346149586e-01, 1.64485995279889710662e-03);
         ir->AddTetPoints12(9, 4.10430739218965501269e-01, 1.65486025619611065718e-01,
                            1.89898020336587186781e-03);
         ir->AddTetPoints12(21, 6.13800882479076381770e-03, 9.42988767345204870196e-01,
                            6.03240573898756009806e-05);
         ir->AddTetPoints12(33, 1.21050181145589408338e-01, 4.77190379904280370660e-01,
                            4.28995533007601147213e-03);
         ir->AddTetPoints12(45, 3.27794682164426753879e-02, 5.94256269480006982242e-01,
                            1.68931194662596552945e-03);
         ir->AddTetPoints12(57, 3.24852815648231096901e-02, 8.01177284658344368573e-01,
                            1.09602454617265063913e-03);
         ir->AddTetPoints12(69, 1.74979342183939068356e-01, 6.28071845475365986289e-01,
                            2.15117263314366490706e-03);
         ir->SetOrder(10);
         TetrahedronIntRules[10] = ir;
         return ir;

      case 11:
         ir = new IntegrationRule(96);
         ir->AddTetPoints4(0, 2.71527207067321363354e-02, 3.30755017786941475644e-04);
         ir->AddTetPoints4(4, 7.29513610462571016058e-02, 1.27724462275054500421e-03);
         ir->AddTetPoints4(8, 1.16306248902001030388e-01, 2.22195840281977797376e-03);
         ir->AddTetPoints4(12, 1.79873804986097840519e-01, 3.55549424791121128006e-03);
         ir->AddTetPoints4(16, 2.90224794862315171873e-01, 4.27767411104971236741e-03);
         ir->AddTetPoints4(20, 3.25420936748619160639e-01, 2.29560465583227143668e-03);
         ir->AddTetPoints6(24, 4.99998725049884129579e-01, 1.95152894059845476377e-04);
         ir->AddTetPoints6(30, 3.94300142842090972639e-01, 4.13762713030314983886e-03);
         ir->AddTetPoints12(36, 1.53994139264412854828e-02, 8.20202176629804657892e-01,
                            3.44113207868302869216e-04);
         ir->AddTetPoints12(48, 4.36843254717693696421e-02, 6.27516751622257062948e-01,
                            2.00540889524405963051e-03);
         ir->AddTetPoints12(60, 1.32316796082697751835e-01, 7.35366407834604496330e-01,
                            5.92160675031106853265e-04);
         ir->AddTetPoints12(72, 2.14430354900043917965e-01, 5.31595425719235903372e-01,
                            3.02373698028425954079e-03);
         ir->AddTetPoints12(84, 4.39586615093850330283e-01, 1.15789732843376125260e-01,
                            1.10416876556284249307e-03);
         ir->SetOrder(11);
         TetrahedronIntRules[11] = ir;
         return ir;

      case 12:
         ir = new IntegrationRule(123);
         ir->AddTetMidPoint(0, 3.73841522662751247000e-03);
         ir->AddTetPoints4(1, 1.87550512633127830497e-02, 1.42695998696545141987e-04);
         ir->AddTetPoints4(5, 1.08129536920462676619e-01, 2.21101382522646480733e-03);
         ir->AddTetPoints4(9, 2.00131676822545012673e-01, 1.02716311841773611131e-03);
         ir->AddTetPoints4(13, 3.00854293538076578152e-01, 3.84627572131096594557e-03);
         ir->AddTetPoints4(17, 3.33333333333333259318e-01, 3.93263480259983290444e-04);
         ir->AddTetPoints6(21, 4.61659950214442116323e-01, 3.64372815936188636666e-04);
         ir->AddTetPoints12(27, 1.44707549187619299857e-02, 8.12825119403836504617e-01,
                            2.78228183082975693598e-04);
         ir->AddTetPoints12(39, 1.93543398987769954545e-02, 6.01428742996530152354e-01,
                            5.44744358748572190913e-04);
         ir->AddTetPoints12(51, 7.79277628532308863640e-02, 8.27642519452021385717e-01,
                            4.96603607085240776955e-04);
         ir->AddTetPoints12(63, 1.22055870746741623734e-01, 4.77806520042316273944e-01,
                            3.50252598711278933380e-03);
         ir->AddTetPoints12(75, 2.47870739372197945727e-01, 4.77761799116294072487e-01,
                            1.92951947508030146987e-03);
         ir->AddTetPoints12(87, 4.29731509588804683197e-01, 1.17747435901101149547e-01,
                            1.59458000732484511328e-03);
         ir->AddTetPoints24(99, 6.53037808968305988344e-01, 2.26776739658831050228e-01,
                            9.77349816032284102185e-02, 1.25441443948160597475e-03);
         ir->SetOrder(12);
         TetrahedronIntRules[12] = ir;
         return ir;

      case 13:
         ir = new IntegrationRule(145);
         ir->AddTetMidPoint(0, 4.65163625751287973520e-03);
         ir->AddTetPoints4(1, 1.83047574861928130652e-02, 1.11328544338301012079e-04);
         ir->AddTetPoints4(5, 1.79015082630022803745e-01, 3.21788310144650391634e-03);
         ir->AddTetPoints4(9, 3.29615853754448240309e-01, 1.11613482625956810662e-03);
         ir->AddTetPoints6(13, 4.84258919196047465938e-01, 4.22777636660235864308e-04);
         ir->AddTetPoints6(19, 4.37693799377281589358e-01, 1.42741910394635984974e-03);
         ir->AddTetPoints12(25, 1.68580250819503341120e-02, 7.16622722155387803511e-01,
                            3.27226375772531215529e-04);
         ir->AddTetPoints12(37, 2.35257182448596058322e-02, 8.50600927605402956644e-01,
                            4.21131904215649394228e-04);
         ir->AddTetPoints12(49, 6.94418950495464537553e-02, 7.08574314820604622689e-01,
                            9.98940835257684659268e-04);
         ir->AddTetPoints12(61, 9.37005272821476720146e-02, 5.63456727822489344959e-01,
                            1.88623398086488818989e-03);
         ir->AddTetPoints12(73, 1.25885360164042447995e-01, 7.40105985667665167149e-01,
                            5.36847192210583730974e-04);
         ir->AddTetPoints12(85, 2.17955270547737667286e-01, 5.35290625276012344003e-01,
                            1.79931440889047528954e-03);
         ir->AddTetPoints12(97, 3.54663455472783883948e-01, 2.00014910210617791186e-01,
                            2.91862732938839245650e-03);
         ir->AddTetPoints12(109, 4.16689287657038387458e-01, 1.52054854976777675812e-01,
                            9.62750257012783515476e-04);
         ir->AddTetPoints24(121, 6.06652560730350010054e-01, 3.04158433676372519372e-01,
                            7.79465622318078477093e-02, 6.21649861415869242447e-04);
         ir->SetOrder(13);
         TetrahedronIntRules[13] = ir;
         return ir;

      case 14:
         ir = new IntegrationRule(180);
         ir->AddTetPoints4(0, 1.93487120802516010531e-02,
                        1.06067047649324128775e-04);
         ir->AddTetPoints4(4, 1.06052184776932997834e-01,
                        1.82377195375880009169e-03);
         ir->AddTetPoints4(8, 1.76306897411661001041e-01,
                        2.72552205602876254642e-03);
         ir->AddTetPoints4(12, 2.30898872883526007360e-01,
                        1.21224099235644122360e-03);
         ir->AddTetPoints4(16, 3.01314483404494015684e-01,
                        2.67068824986019985948e-03);
         ir->AddTetPoints4(20, 3.26708966297015013236e-01,
                        1.16628766415437885515e-03);
         ir->AddTetPoints6(24, 3.57855935072428979482e-02,
                        8.33790018893379968057e-04);
         ir->AddTetPoints6(30, 1.37206483887745006589e-01,
                        2.77309925846009990014e-03);
         ir->AddTetPoints12(36, 5.31986206598606976154e-03, 8.86841133422155009081e-02,
                            5.58583823423464966799e-05);
         ir->AddTetPoints12(48, 1.21092057944150992971e-02, 3.64320902432274995597e-01,
                            2.16867559229133740305e-04);
         ir->AddTetPoints12(60, 2.34347104796242995672e-02, 1.96142506670935995450e-01,
                            4.85640995603313732100e-04);
         ir->AddTetPoints12(72, 7.44714552020504932939e-02, 2.84848931445568998022e-01,
                            1.67428997401904990604e-03);
         ir->AddTetPoints12(84, 7.63912821970205019317e-02, 1.73597444016867011318e-02,
                            4.60798274250136274434e-04);
         ir->AddTetPoints12(96, 2.16987176174722989908e-01, 5.80150927227437024358e-02,
                            2.04381708293785002012e-03);
         ir->AddTetPoints12(108, 2.46389817533077010170e-01, 1.29672964789885993356e-03,
                            4.23286353630371228046e-04);
         ir->AddTetPoints12(120, 4.08360847823448003258e-01, 2.95698037900974999848e-02,
                            1.55922797825092497928e-03);
         ir->AddTetPoints24(132, 1.87368391416951993352e-03, 8.74018565728438973084e-02,
                            3.31958209684441007958e-01, 3.22902746408971271386e-04);
         ir->AddTetPoints24(156, 2.00450273643128992762e-02, 1.09443563180489006337e-01,
                            1.92537872734766996041e-01, 6.42496417930884990838e-04);
         ir->SetOrder(14);
         TetrahedronIntRules[14] = ir;
         return ir;

      case 15:
         ir = new IntegrationRule(213);
         ir->AddTetMidPoint(0, 1.09597814206725211600e-03);
         ir->AddTetPoints4(1, 6.13445374189439118773e-02, 6.08654145635489159279e-04);
         ir->AddTetPoints4(5, 1.56619848311296605559e-01, 1.59654095760468480933e-03);
         ir->AddTetPoints4(9, 2.08983416950805578338e-01, 2.37278041145122867150e-03);
         ir->AddTetPoints4(13, 3.02341424068938158243e-01, 1.86824828742607348085e-03);
         ir->AddTetPoints4(17, 3.29188903694042855896e-01, 8.00139152472053596064e-04);
         ir->AddTetPoints6(21, 4.84570056460385811814e-01, 3.21176512333682967915e-04);
         ir->AddTetPoints6(27, 3.67113125728904199363e-01, 2.30534931414932368204e-03);
         ir->AddTetPoints12(33, 1.14254053187452520035e-02, 9.31835064037604743348e-01,
                            6.74539727340275966890e-05);
         ir->AddTetPoints12(45, 1.75458438338256250688e-02, 6.73018705833061559041e-01,
                            3.43665155671541789131e-04);
         ir->AddTetPoints12(57, 1.76781044873532411366e-02, 8.20776317761974238962e-01,
                            2.50812012694799689547e-04);
         ir->AddTetPoints12(69, 6.06189467360996325773e-02, 5.17994269719641353689e-01,
                            1.00698272496308458869e-03);
         ir->AddTetPoints12(81, 7.95111143787260998828e-02, 8.34059424208046018556e-01,
                            2.00393629319816219932e-04);
         ir->AddTetPoints12(93, 8.13484076864854355193e-02, 6.73983322548355134884e-01,
                            8.31757527429042523709e-04);
         ir->AddTetPoints12(105, 2.26137812479195732251e-01, 5.39799048649219126439e-01,
                            6.17433070978635426679e-04);
         ir->AddTetPoints12(117, 2.43623481031443045453e-01, 4.32417328667378919604e-01,
                            1.12347184083411741166e-03);
         ir->AddTetPoints12(129, 3.95762442557879401406e-01, 1.83488453291837605441e-01,
                            1.22175368078889759645e-03);
         ir->AddTetPoints24(141, 5.45471540423582257340e-01, 3.59081001579021674708e-01,
                            8.81192856783694078437e-02, 4.08041207514277671619e-04);
         ir->AddTetPoints24(165, 6.93064510781163489739e-01, 1.95660234877701455503e-01,
                            9.29173257624009707456e-02, 6.55640487342973787496e-04);
         ir->AddTetPoints24(189, 5.34048287107432018139e-01, 2.57614777559765806281e-01,
                            1.47157090536597368047e-01, 1.13887657024173603003e-03);
         ir->SetOrder(15);
         TetrahedronIntRules[15] = ir;
         return ir;

      case 16:
         ir = new IntegrationRule(248);
         ir->AddTetPoints4(0, 1.52484867986518360e-02, 5.99504151841072482811e-05);
         ir->AddTetPoints4(4, 2.94565614715276806e-02, 5.99504407860089978742e-05);
         ir->AddTetPoints4(8, 1.14858334717906962e-01, 9.16153144855748770e-04);
         ir->AddTetPoints4(12, 1.67622967087127012e-01, 1.52109430849001258e-03);
         ir->AddTetPoints4(16, 2.09524276995675995e-01, 1.77324207014633748e-03);
         ir->AddTetPoints4(20, 2.73570592899956000e-01, 5.39659969854351293e-04);
         ir->AddTetPoints4(24, 3.02514911509892015e-01, 2.02348561033093748e-03);
         ir->AddTetPoints4(28, 3.27179256536997976e-01, 1.04275139723560245e-03);
         ir->AddTetPoints6(32, 6.82067828413315258e-02, 1.33730851717791242e-03);
         ir->AddTetPoints6(38, 1.48245803714745938e-01, 1.76246686094257490e-03);
         ir->AddTetPoints12(44, 8.29083396753205193e-03, 1.02799562245275977e-01, 7.67748485300563796e-05);
         ir->AddTetPoints12(56, 1.54521389813079835e-02, 3.89606562255856992e-01, 2.69895951724489992e-04);
         ir->AddTetPoints12(68, 1.58349833924095185e-02, 2.34000762147752006e-01, 2.28062018345420004e-04);
         ir->AddTetPoints12(80, 4.80151786100800060e-02, 1.31565941293946997e-01, 6.09836007633601210e-04);
         ir->AddTetPoints12(92, 5.22519420338140028e-02, 2.65214317841850011e-01, 5.21562485856138787e-04);
         ir->AddTetPoints12(104, 6.49951454829435227e-02, 1.08448532530004971e-02, 1.90537035385926240e-04);
         ir->AddTetPoints12(116, 9.54674612364609931e-02, 2.29759478443447018e-01, 9.08935512152296197e-04);
         ir->AddTetPoints12(128, 1.54104869766025970e-01, 3.00657244486740027e-02, 9.04193887085843768e-04);
         ir->AddTetPoints12(140, 2.33883852712881979e-01, 7.85426388526899721e-03, 5.40757076612427448e-04);
         ir->AddTetPoints12(152, 4.03170681447351242e-01, 1.61521517457074992e-02, 7.92865738300459959e-04);
         ir->AddTetPoints12(164, 4.62670359916073970e-01, 8.11096429888952253e-03, 2.79410938937282481e-04);
         ir->AddTetPoints24(176, 1.40491380816509803e-04, 7.52152961265974973e-02, 1.79100437644861943e-01, 1.50619939841640008e-04);
         ir->AddTetPoints24(200, 1.48035899046374819e-02, 9.60400303725589821e-02, 3.06302989797205549e-01, 5.64426496212163732e-04);
         ir->AddTetPoints24(224, 6.67738359900459932e-02, 1.77446596253103994e-01, 2.86183922421975934e-01, 1.47032385409803746e-03);
         ir->SetOrder(16);
         TetrahedronIntRules[16] = ir;
         return ir;

      case 17:
         ir = new IntegrationRule(286);
         ir->AddTetPoints4(0, 1.47598836609400541e-02, 5.28840890663946240e-05);
         ir->AddTetPoints4(4, 7.37497383607043783e-02, 5.06526532418637480e-04);
         ir->AddTetPoints4(8, 1.35219507338736000e-01, 1.33594214183261240e-03);
         ir->AddTetPoints4(12, 1.80928141422366989e-01, 6.77301689184768769e-04);
         ir->AddTetPoints4(16, 2.18587294962143991e-01, 1.66528884447029991e-03);
         ir->AddTetPoints4(20, 2.97435045949379029e-01, 2.08059092686700003e-03);
         ir->AddTetPoints4(24, 3.24756166038925986e-01, 4.64749046605275010e-04);
         ir->AddTetPoints6(28, 6.86081126537751995e-03, 8.38891886776328802e-05);
         ir->AddTetPoints12(34, 5.17514079785097447e-03, 3.12137975448285032e-01, 5.50039585256934980e-05);
         ir->AddTetPoints12(46, 1.43107480454737479e-02, 7.51465455972474827e-02, 1.10920406938030380e-04);
         ir->AddTetPoints12(58, 1.46965971812722529e-02, 1.76369978485476975e-01, 1.66199984599868744e-04);
         ir->AddTetPoints12(70, 5.05669142090789947e-02, 1.81815229162291980e-01, 5.44972393992090051e-04);
         ir->AddTetPoints12(82, 7.54675824272149909e-02, 1.45445445812680263e-02, 2.68609986903531231e-04);
         ir->AddTetPoints12(94, 8.34224757015799845e-02, 2.71248484515970001e-01, 1.11678505498206504e-03);
         ir->AddTetPoints12(106, 1.41757340264034998e-01, 2.99627907108787006e-01, 1.46253132158004990e-03);
         ir->AddTetPoints12(118, 1.44466835027020984e-01, 4.24572771457780163e-02, 7.87423391530311196e-04);
         ir->AddTetPoints12(130, 1.89876572061487003e-01, 6.45663045352651288e-03, 2.42012075851381247e-04);
         ir->AddTetPoints12(142, 2.18840964673858884e-01, 7.16655871049595161e-02, 1.43848904399491257e-03);
         ir->AddTetPoints12(154, 2.73588604110502009e-01, 1.08752193152130139e-02, 5.61097271893722474e-04);
         ir->AddTetPoints12(166, 3.79114196918969304e-01, 4.86860544613335056e-02, 1.18674138121477746e-03);
         ir->AddTetPoints12(178, 4.23858883682162957e-01, 2.76763044898487021e-04, 2.27576613337623749e-04);
         ir->AddTetPoints12(190, 4.28577279910766995e-01, 4.85444074331429776e-02, 7.23697457154796257e-04);
         ir->AddTetPoints12(202, 4.67262072050318000e-01, 1.25258273808724896e-02, 3.16470333662152497e-04);
         ir->AddTetPoints24(214, 4.81132159695801809e-03, 7.86466828328645229e-02, 1.84988832037595974e-01, 2.00835373652671261e-04);
         ir->AddTetPoints24(238, 1.37993139820000166e-02, 4.89791122093244935e-02, 3.22491732038402035e-01, 3.39888029876185018e-04);
         ir->AddTetPoints24(262, 1.97833665261955005e-02, 1.34676387955150023e-01, 2.92734467158131006e-01, 6.47936193924848725e-04);
         ir->SetOrder(17);
         TetrahedronIntRules[17] = ir;
         return ir;

      case 18:
         ir = new IntegrationRule(360);
         ir->AddTetPoints4(0, 1.21612991820346048e-02, 3.36004769125282509e-05);
         ir->AddTetPoints4(4, 5.98609857989486419e-02, 3.83749797619453736e-04);
         ir->AddTetPoints4(8, 1.28935410926582006e-01, 5.81241076165449951e-04);
         ir->AddTetPoints4(12, 1.56237978338631067e-01, 9.24513809123521293e-04);
         ir->AddTetPoints4(16, 2.89634952526186040e-01, 1.06154448758938617e-03);
         ir->AddTetPoints4(20, 3.11600727301163005e-01, 4.25145760007678735e-04);
         ir->AddTetPoints6(24, 9.97963224578402031e-03, 1.02900277564123006e-04);
         ir->AddTetPoints6(30, 2.21815979195565949e-01, 4.62638070041924976e-04);
         ir->AddTetPoints12(36, 8.73999261671876138e-03, 8.41859233755140046e-02, 5.24953929310302471e-05);
         ir->AddTetPoints12(48, 1.11168051162787718e-02, 1.97973163583385015e-01, 1.06883197473858744e-04);
         ir->AddTetPoints12(60, 1.15549853782892509e-02, 3.38413050134774962e-01, 1.25365338330555009e-04);
         ir->AddTetPoints12(72, 4.62447491622127449e-02, 2.50468700990438997e-01, 4.43519657525103775e-04);
         ir->AddTetPoints12(84, 4.70543628996780150e-02, 9.21396510194050222e-03, 8.97142980378936287e-05);
         ir->AddTetPoints12(96, 6.39238531342868344e-02, 3.62763112507323959e-01, 7.22696004472126222e-04);
         ir->AddTetPoints12(108, 6.78992531211482508e-02, 1.60467857399488989e-01, 3.94408567904652512e-04);
         ir->AddTetPoints12(120, 1.30089538063895005e-01, 4.66996705048839944e-02, 4.82708709827315023e-04);
         ir->AddTetPoints12(132, 1.41320331283104006e-01, 3.64401095365052274e-03, 1.30243221008284989e-04);
         ir->AddTetPoints12(144, 1.50612919662131972e-01, 2.86717179253101029e-01, 1.17122588240265628e-03);
         ir->AddTetPoints12(156, 2.33328072545731124e-01, 1.05055084109579866e-02, 4.31598497998792483e-04);
         ir->AddTetPoints12(168, 2.35047063999292982e-01, 1.33163536577812991e-01, 7.77261475417438718e-04);
         ir->AddTetPoints12(180, 2.62096353140561011e-01, 5.36700494133749983e-02, 1.01287673593292876e-03);
         ir->AddTetPoints12(192, 3.72603856057497962e-01, 1.09749466550385089e-02, 4.98829517748433757e-04);
         ir->AddTetPoints12(204, 3.90708677155744977e-01, 6.42554201700920036e-02, 1.13805518813424127e-03);
         ir->AddTetPoints24(216, 6.13198129842651429e-03, 5.91505365908949754e-02, 2.60923072826566016e-01, 1.62228049762157495e-04);
         ir->AddTetPoints24(240, 1.22084325856364750e-02, 5.56968253699064997e-02, 4.00652272386217645e-01, 2.63517464527631264e-04);
         ir->AddTetPoints24(264, 1.27497248166335009e-02, 1.34732789360122007e-01, 3.52113535493879248e-01, 4.42329352552116230e-04);
         ir->AddTetPoints24(288, 1.34252305987554998e-02, 5.64345001370845090e-02, 1.31366537726315069e-01, 2.43011886996146245e-04);
         ir->AddTetPoints24(312, 1.62863187851640112e-02, 1.25716381708101010e-01, 2.27564097123100995e-01, 3.39528162745711241e-04);
         ir->AddTetPoints24(336, 6.68099593141014969e-02, 1.42305898758797011e-01, 2.45600896140753922e-01, 9.95204863816846145e-04);
         ir->SetOrder(18);
         TetrahedronIntRules[18] = ir;
         return ir;

      case 19:
         ir = new IntegrationRule(420);
         ir->AddTetPoints4(0, 1.31788192421319703e-02, 3.82193395791433771e-05);
         ir->AddTetPoints4(4, 5.66892885709266645e-02, 2.98231164316413758e-04);
         ir->AddTetPoints4(8, 1.65015397838813999e-01, 9.28419959035332474e-04);
         ir->AddTetPoints12(12, 7.58546556894754143e-03, 1.64575203645211998e-01, 5.09947115152120032e-05);
         ir->AddTetPoints12(24, 8.37198046062501144e-03, 4.18632374633719995e-01, 7.14038428681792439e-05);
         ir->AddTetPoints12(36, 1.18322276808884852e-02, 6.84269434961595202e-02, 6.92763497814434932e-05);
         ir->AddTetPoints12(48, 1.20675846620199867e-02, 2.81006744545988008e-01, 1.07931488911448375e-04);
         ir->AddTetPoints12(60, 4.78905350727200085e-02, 2.51867401423586990e-01, 4.54149924126790003e-04);
         ir->AddTetPoints12(72, 5.35979829045493128e-02, 3.70849448695308004e-01, 4.67207825801212488e-04);
         ir->AddTetPoints12(84, 6.16489404148549969e-02, 1.00122457895789885e-02, 1.22189479527250134e-04);
         ir->AddTetPoints12(96, 6.34266569447892448e-02, 1.40714637888080973e-01, 3.25558721186596271e-04);
         ir->AddTetPoints12(108, 1.00918491996615983e-01, 1.78307909384450014e-01, 6.78670533286543755e-04);
         ir->AddTetPoints12(120, 1.21684785885215002e-01, 3.51203816294214932e-02, 3.56520264560966269e-04);
         ir->AddTetPoints12(132, 1.34096203033947980e-01, 2.13286177894700391e-03, 1.15596177693075378e-04);
         ir->AddTetPoints12(144, 1.37895625303441988e-01, 3.13886615101924005e-01, 9.38850071082121280e-04);
         ir->AddTetPoints12(156, 1.84306899674015978e-01, 4.16061221228199996e-02, 5.56849613378066215e-04);
         ir->AddTetPoints12(168, 2.19741345150853001e-01, 1.00090080901258982e-01, 9.60977160044931204e-04);
         ir->AddTetPoints12(180, 2.23704038070779015e-01, 1.22296652272979589e-04, 1.61895616674714990e-04);
         ir->AddTetPoints12(192, 2.46643098734524002e-01, 1.64206059730428011e-01, 1.00546714458886878e-03);
         ir->AddTetPoints12(204, 3.36142941168669052e-01, 1.50698963700999777e-02, 1.83851627947964987e-04);
         ir->AddTetPoints12(216, 3.47281736613307968e-01, 7.26919784981269745e-02, 1.01930598139808246e-03);
         ir->AddTetPoints12(228, 3.90055469241282982e-01, 1.12239395228247441e-03, 1.58649251067613739e-04);
         ir->AddTetPoints12(240, 4.12516048930301027e-01, 5.26538976479284937e-02, 7.07797896418448720e-04);
         ir->AddTetPoints24(252, 5.07279440210001153e-03, 5.12886988018579904e-02, 2.68091359552455000e-01, 1.13353019333235124e-04);
         ir->AddTetPoints24(276, 1.03601284811329908e-02, 1.19466435769161994e-01, 3.62381285495233019e-01, 3.18506849543317505e-04);
         ir->AddTetPoints24(300, 1.08142178458284777e-02, 4.81720747827449891e-02, 4.03305150141510471e-01, 2.09612481742775010e-04);
         ir->AddTetPoints24(324, 1.27306717754545051e-02, 5.01791990679790190e-02, 1.48766292999399974e-01, 2.05445681944725010e-04);
         ir->AddTetPoints24(348, 1.56976315246900033e-02, 1.14848896039848991e-01, 2.37716733271631009e-01, 3.69743822937373750e-04);
         ir->AddTetPoints24(372, 2.60822191842460249e-02, 2.11059696887617998e-01, 3.03881062339131869e-01, 5.80741058295128777e-04);
         ir->AddTetPoints24(396, 6.68088684784480247e-02, 1.28700331992339989e-01, 2.78895524704544706e-01, 6.79657945896311217e-04);
         ir->SetOrder(19);
         TetrahedronIntRules[19] = ir;
         return ir;

      case 20:
         ir = new IntegrationRule(454);
         ir->AddTetPoints4(0, 1.14982590351778349e-02, 2.54898045929234989e-05);
         ir->AddTetPoints4(4, 6.89530857566646588e-02, 2.92331437530762492e-04);
         ir->AddTetPoints4(8, 1.14101964191234015e-01, 3.75557040499648769e-04);
         ir->AddTetPoints4(12, 2.26623670903950969e-01, 8.49894786496254969e-04);
         ir->AddTetPoints4(16, 2.89243653771868003e-01, 8.34055660362309969e-04);
         ir->AddTetPoints4(20, 3.17004432174104012e-01, 8.63167800196846255e-04);
         ir->AddTetPoints4(24, 3.31028638344827042e-01, 2.01603302126462507e-04);
         ir->AddTetPoints6(28, 1.49182829833814923e-02, 9.60079473556843690e-05);
         ir->AddTetPoints6(34, 1.01675467118900964e-01, 2.73059370749955017e-04);
         ir->AddTetPoints6(40, 1.68941981405750008e-01, 7.21080316181650009e-04);
         ir->AddTetPoints12(46, 7.56521776567550819e-03, 6.10924008816329800e-02, 2.94475128262334993e-05);
         ir->AddTetPoints12(58, 8.36107667864674498e-03, 3.94544679191236980e-01, 6.98556838916828689e-05);
         ir->AddTetPoints12(70, 1.06658621945099896e-02, 2.49977524602618018e-01, 9.73872572951808694e-05);
         ir->AddTetPoints12(82, 3.45794612092844567e-02, 1.12019812423493015e-01, 2.23013263172452513e-04);
         ir->AddTetPoints12(94, 4.89319360614865195e-02, 1.12784042718049871e-02, 1.02343433134437502e-04);
         ir->AddTetPoints12(106, 4.97033000780047196e-02, 2.43257586121019997e-01, 2.42974400168919995e-04);
         ir->AddTetPoints12(118, 5.10037542007290012e-02, 3.67945291810742980e-01, 4.80875338274760003e-04);
         ir->AddTetPoints12(130, 7.48599119077760222e-02, 1.64236407332499001e-01, 3.88757286540629974e-04);
         ir->AddTetPoints12(142, 9.10115850848747521e-02, 2.23714676430063986e-01, 3.19409292359203731e-04);
         ir->AddTetPoints12(154, 9.87353111996785160e-02, 6.16531449401497289e-03, 1.33021564866297494e-04);
         ir->AddTetPoints12(166, 1.11850388225890024e-01, 3.01228466943312012e-01, 9.03070354254908786e-04);
         ir->AddTetPoints12(178, 1.22653801145481023e-01, 3.32316986358039967e-02, 3.50335277933808739e-04);
         ir->AddTetPoints12(190, 1.77043987282103021e-01, 9.44259663703654883e-02, 8.17122629336931297e-04);
         ir->AddTetPoints12(202, 1.78066784887388002e-01, 3.09343296668529999e-02, 4.12917879681161252e-04);
         ir->AddTetPoints12(214, 2.19082479256664991e-01, 1.33745856552868025e-01, 9.83920067171401144e-04);
         ir->AddTetPoints12(226, 2.42586884433407063e-01, 9.08113791779802115e-03, 3.73278155708182513e-04);
         ir->AddTetPoints12(238, 3.47791919849646969e-01, 1.00715819343872026e-01, 7.47700772680592483e-04);
         ir->AddTetPoints12(250, 3.81024201303649002e-01, 1.17406613337495136e-02, 4.17492565127586268e-04);
         ir->AddTetPoints12(262, 4.19411663188390016e-01, 4.27658513732990242e-02, 6.22612033427441297e-04);
         ir->AddTetPoints12(274, 4.66198324022163990e-01, 9.76525149374452006e-03, 2.29289621993751243e-04);
         ir->AddTetPoints24(286, 1.41129559109548497e-03, 2.34856005772224874e-02, 1.39035106491765037e-01, 4.67151353243301269e-05);
         ir->AddTetPoints24(310, 3.23413549940498868e-03, 1.26849866338819006e-01, 2.16420089002580029e-01, 1.43432619652048753e-04);
         ir->AddTetPoints24(334, 7.31682891540552660e-03, 1.31770989026243013e-01, 3.51730617481239849e-01, 2.66336041058678748e-04);
         ir->AddTetPoints24(358, 9.21108705827600183e-03, 4.96903454030804825e-02, 3.23819064968342007e-01, 1.99857167436717493e-04);
         ir->AddTetPoints24(382, 1.44914937006905276e-02, 5.96740629000345191e-02, 1.96024452862043452e-01, 2.23981236019175011e-04);
         ir->AddTetPoints24(406, 3.60854317950489722e-02, 1.18445031971756987e-01, 2.70400786211849986e-01, 5.04716790352354957e-04);
         ir->AddTetPoints24(430, 5.13097598889180051e-02, 2.01435523695688012e-01, 2.96487931834384044e-01, 7.40773045805666244e-04);
         ir->SetOrder(20);
         TetrahedronIntRules[20] = ir;
         return ir;

      default:
         // Grundmann-Moller fallback for orders beyond WV tables
         int i = (Order / 2) * 2 + 1;   // closest odd >= Order
         AllocIntRule(TetrahedronIntRules, i);
         ir = new IntegrationRule;
         ir->GrundmannMollerSimplexRule(i/2, 3);
         TetrahedronIntRules[i-1] = TetrahedronIntRules[i] = ir;
         return ir;
   }
}


// Integration rules for reference pyramid
IntegrationRule *IntegrationRules::PyramidIntegrationRule(int Order)
{
   // This is a simple integration rule adapted from an integration
   // rule for a cube which seems to be adequate for now. We should continue
   // to search for a more appropriate integration rule designed specifically
   // for pyramid elements.
   const IntegrationRule &irc = Get(Geometry::CUBE, Order);
   int npts = irc.GetNPoints();
   AllocIntRule(PyramidIntRules, Order);
   PyramidIntRules[Order] = new IntegrationRule(npts);
   PyramidIntRules[Order]->SetOrder(Order);

   if (npts == 1)
   {
      // We handle this as a special case because with only one integration
      // point we cannot accurately integrate the quadratic factor
      // pow(1.0 - ipc.z, 2) and the resulting weight does not match the volume
      // of the reference element.
      IntegrationPoint &ipp = PyramidIntRules[Order]->IntPoint(0);
      ipp.x = 0.375;
      ipp.y = 0.375;
      ipp.z = 0.25;
      ipp.weight = 1.0 / 3.0;
   }
   else
   {
      for (int k=0; k<npts; k++)
      {
         const IntegrationPoint &ipc = irc.IntPoint(k);
         IntegrationPoint &ipp = PyramidIntRules[Order]->IntPoint(k);
         ipp.x = ipc.x * (1.0 - ipc.z);
         ipp.y = ipc.y * (1.0 - ipc.z);
         ipp.z = ipc.z;
         ipp.weight = ipc.weight * pow(1.0 - ipc.z, 2);
      }
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
                                                Array<const KnotVector*> const& kv) const
{
   // First check whether a rule has been assigned to element index elem.
   auto search = elementToRule.find(elem);
   if (search != elementToRule.end())
   {
      return *elementRule[search->second];
   }

#ifndef MFEM_THREAD_SAFE
   // If no prescribed rule is given for the current element, a temporary one is
   // formed by restricting a tensor-product of 1D rules to the element. The
   // ownership model for this temporary rule is not thread-safe.

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

   temporaryElementRule.SetSize(np);

   // Set temporaryElementRule[i + j*npd[0] + k*npd[0]*npd[1]] =
   //     (el[0][2*i], el[1][2*j], el[2][2*k])

   MFEM_VERIFY(npd[0] > 0 && npd[1] > 0, "Assuming 2D or 3D");

   for (int i = 0; i < npd[0]; ++i)
   {
      for (int j = 0; j < npd[1]; ++j)
      {
         for (int k = 0; k < std::max(npd[2], 1); ++k)
         {
            const int id = i + j*npd[0] + k*npd[0]*npd[1];
            temporaryElementRule[id].x = el[0][2*i];
            temporaryElementRule[id].y = el[1][2*j];

            temporaryElementRule[id].weight = el[0][(2*i)+1];
            temporaryElementRule[id].weight *= el[1][(2*j)+1];

            if (npd[2] > 0)
            {
               temporaryElementRule[id].z = el[2][2*k];
               temporaryElementRule[id].weight *= el[2][(2*k)+1];
            }
         }
      }
   }

   return temporaryElementRule;
#else
   MFEM_ABORT("Temporary integration rules on NURBS elements "
              "are not thread-safe.");
#endif
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
