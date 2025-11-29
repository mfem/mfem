//                  MFEM Example 33 - Serial/Parallel Shared Code
//                      (Implementation of the AAA algorithm)
//
//  Here, we implement the triple-A algorithm [1] for the rational approximation
//  of complex-valued functions,
//
//          p(z)/q(z) ≈ f(z).
//
//  In this file, we always assume f(z) = z^{-α}. The triple-A algorithm
//  provides a robust, accurate approximation in rational barycentric form.
//  This representation must be transformed into a partial fraction
//  representation in order to be used to solve a spectral FPDE.
//
//  More specifically, we first expand the numerator in terms of the zeros of
//  the rational approximation,
//
//          p(z) ∝ Π_i (z - z_i),
//
//  and expand the denominator in terms of the poles of the rational
//  approximation,
//
//          q(z) ∝ Π_i (z - p_i).
//
//  We then use these zeros and poles to derive the partial fraction expansion
//
//          f(z) ≈ p(z)/q(z) = Σ_i c_i / (z - p_i).
//
//  [1] Nakatsukasa, Y., Sète, O., & Trefethen, L. N. (2018). The AAA algorithm
//      for rational approximation. SIAM Journal on Scientific Computing, 40(3),
//      A1494-A1522.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <string>

using namespace std;
using namespace mfem;

/** RationalApproximation_AAA: compute the rational approximation (RA) of data
    @a val [in] at the set of points @a pt [in].

    @param[in]  val        Vector of data values
    @param[in]  pt         Vector of sample points
    @param[in]  tol        Relative tolerance
    @param[in]  max_order  Maximum number of terms (order) of the RA
    @param[out] z          Support points of the RA in rational barycentric form
    @param[out] f          Data values at support points @a z
    @param[out] w          Weights of the RA in rational barycentric form

    See pg. A1501 of Nakatsukasa et al. [1]. */
void RationalApproximation_AAA(const Vector &val, const Vector &pt,
                               Array<real_t> &z, Array<real_t> &f, Vector &w,
                               real_t tol, int max_order)
{

   // number of sample points
   int size = val.Size();
   MFEM_VERIFY(pt.Size() == size, "size mismatch");

   // Initializations
   Array<int> J(size);
   for (int i = 0; i < size; i++) { J[i] = i; }
   z.SetSize(0);
   f.SetSize(0);

   DenseMatrix C, Ctemp, A, Am;
   // auxiliary arrays and vectors
   Vector f_vec;
   Array<real_t> c_i;

   // mean of the value vector
   Vector R(val.Size());
   real_t mean_val = val.Sum()/size;

   for (int i = 0; i<R.Size(); i++) { R(i) = mean_val; }

   for (int k = 0; k < max_order; k++)
   {
      // select next support point
      int idx = 0;
      real_t tmp_max = 0;
      for (int j = 0; j < size; j++)
      {
         real_t tmp = abs(val(j)-R(j));
         if (tmp > tmp_max)
         {
            tmp_max = tmp;
            idx = j;
         }
      }

      // Append support points and data values
      z.Append(pt(idx));
      f.Append(val(idx));

      // Update index vector
      J.DeleteFirst(idx);

      // next column in Cauchy matrix
      Array<real_t> C_tmp(size);
      for (int j = 0; j < size; j++)
      {
         C_tmp[j] = 1.0/(pt(j)-pt(idx));
      }
      c_i.Append(C_tmp);
      int h_C = C_tmp.Size();
      int w_C = k+1;
      C.UseExternalData(c_i.GetData(),h_C,w_C);

      Ctemp = C;

      f_vec.SetDataAndSize(f.GetData(),f.Size());
      Ctemp.InvLeftScaling(val);
      Ctemp.RightScaling(f_vec);

      A.SetSize(C.Height(), C.Width());
      Add(C,Ctemp,-1.0,A);
      A.LeftScaling(val);

      int h_Am = J.Size();
      int w_Am = A.Width();
      Am.SetSize(h_Am,w_Am);
      for (int i = 0; i<h_Am; i++)
      {
         int ii = J[i];
         for (int j = 0; j<w_Am; j++)
         {
            Am(i,j) = A(ii,j);
         }
      }

#ifdef MFEM_USE_LAPACK
      DenseMatrixSVD svd(Am,'N','A');
      svd.Eval(Am);
      DenseMatrix &v = svd.RightSingularvectors();
      v.GetRow(k,w);
#else
      mfem_error("Compiled without LAPACK");
#endif

      // N = C*(w.*f); D = C*w; % numerator and denominator
      Vector aux(w);
      aux *= f_vec;
      Vector N(C.Height()); // Numerator
      C.Mult(aux,N);
      Vector D(C.Height()); // Denominator
      C.Mult(w,D);

      R = val;
      for (int i = 0; i<J.Size(); i++)
      {
         int ii = J[i];
         R(ii) = N(ii)/D(ii);
      }

      Vector verr(val);
      verr-=R;

      if (verr.Normlinf() <= tol*val.Normlinf()) { break; }
   }
}

/** ComputePolesAndZeros: compute the @a poles [out] and @a zeros [out] of the
    rational function f(z) = C p(z)/q(z) from its ration barycentric form.

    @param[in]  z      Support points in rational barycentric form
    @param[in]  f      Data values at support points @a z
    @param[in]  w      Weights in rational barycentric form
    @param[out] poles  Array of poles (roots of p(z))
    @param[out] zeros  Array of zeros (roots of q(z))
    @param[out] scale  Scaling constant in f(z) = C p(z)/q(z)

    See pg. A1501 of Nakatsukasa et al. [1]. */
void ComputePolesAndZeros(const Vector &z, const Vector &f, const Vector &w,
                          Array<real_t> & poles, Array<real_t> & zeros, real_t &scale)
{
   // Initialization
   poles.SetSize(0);
   zeros.SetSize(0);

   // Compute the poles
   int m = w.Size();
   DenseMatrix B(m+1); B = 0.;
   DenseMatrix E(m+1); E = 0.;
   for (int i = 1; i<=m; i++)
   {
      B(i,i) = 1.;
      E(0,i) = w(i-1);
      E(i,0) = 1.;
      E(i,i) = z(i-1);
   }

#ifdef MFEM_USE_LAPACK
   DenseMatrixGeneralizedEigensystem eig1(E,B);
   eig1.Eval();
   Vector & evalues = eig1.EigenvaluesRealPart();
   for (int i = 0; i<evalues.Size(); i++)
   {
      if (IsFinite(evalues(i)))
      {
         poles.Append(evalues(i));
      }
   }
#else
   mfem_error("Compiled without LAPACK");
#endif
   // compute the zeros
   B = 0.;
   E = 0.;
   for (int i = 1; i<=m; i++)
   {
      B(i,i) = 1.;
      E(0,i) = w(i-1) * f(i-1);
      E(i,0) = 1.;
      E(i,i) = z(i-1);
   }

#ifdef MFEM_USE_LAPACK
   DenseMatrixGeneralizedEigensystem eig2(E,B);
   eig2.Eval();
   evalues = eig2.EigenvaluesRealPart();
   for (int i = 0; i<evalues.Size(); i++)
   {
      if (IsFinite(evalues(i)))
      {
         zeros.Append(evalues(i));
      }
   }
#else
   mfem_error("Compiled without LAPACK");
#endif

   scale = w * f / w.Sum();
}

/** PartialFractionExpansion: compute the partial fraction expansion of the
    rational function f(z) = Σ_i c_i / (z - p_i) from its @a poles [in] and
    @a zeros [in].

    @param[in]  poles   Array of poles (same as p_i above)
    @param[in]  zeros   Array of zeros
    @param[in]  scale   Scaling constant
    @param[out] coeffs  Coefficients c_i */
void PartialFractionExpansion(real_t scale, Array<real_t> & poles,
                              Array<real_t> & zeros, Array<real_t> & coeffs)
{
   int psize = poles.Size();
   int zsize = zeros.Size();
   coeffs.SetSize(psize);
   coeffs = scale;

   // Note: C p(z)/q(z) = Σ_i c_i / (z - p_i) results in an system of equations
   // where the N unknowns are the coefficients c_i. After multiplying the
   // system with q(z), the coefficients c_i can be computed analytically by
   // choosing N values for z. Choosing z_j = = p_j diagonalizes the system and
   // one can obtain an analytic form for the c_i coefficients. The result is
   // implemented in the code block below.

   for (int i=0; i<psize; i++)
   {
      real_t tmp_numer=1.0;
      for (int j=0; j<zsize; j++)
      {
         tmp_numer *= poles[i]-zeros[j];
      }

      real_t tmp_denom=1.0;
      for (int k=0; k<psize; k++)
      {
         if (k != i) { tmp_denom *= poles[i]-poles[k]; }
      }
      coeffs[i] *= tmp_numer / tmp_denom;
   }
}


/** ComputePartialFractionApproximation: compute a rational approximation (RA)
    in partial fraction form, e.g., f(z) ≈ Σ_i c_i / (z - p_i), from sampled
    values of the function f(z) = z^{-a}, 0 < a < 1.

    @param[in]  alpha         Exponent a in f(z) = z^-a
    @param[in] lmax, npoints  f(z) is uniformly sampled @a npoints times in the
                              interval [ 0, @a lmax ]
    @param[in]  tol           Relative tolerance
    @param[in]  max_order     Maximum number of terms (order) of the RA
    @param[out] coeffs        Coefficients c_i
    @param[out] poles         Poles p_i

    NOTES: When MFEM is not built with LAPACK support, only @a alpha = 0.33,
           0.5, and 0.99 are possible. In this case, if @a alpha != 0.33 and
           @a alpha != 0.99, then @a alpha = 0.5 is used by default.

   See pg. A1501 of Nakatsukasa et al. [1]. */
void ComputePartialFractionApproximation(real_t & alpha,
                                         Array<real_t> & coeffs, Array<real_t> & poles,
                                         real_t lmax = 1000.,
                                         real_t tol=1e-10, int npoints = 1000,
                                         int max_order = 100)
{
   MFEM_VERIFY(alpha < 1., "alpha must be less than 1");
   MFEM_VERIFY(alpha > 0., "alpha must be greater than 0");
   MFEM_VERIFY(npoints > 2, "npoints must be greater than 2");
   MFEM_VERIFY(lmax > 0,  "lmin must be greater than 0");
   MFEM_VERIFY(tol > 0,  "tol must be greater than 0");

   bool print_warning = true;
#ifdef MFEM_USE_MPI
   if ((Mpi::IsInitialized() && !Mpi::Root())) { print_warning = false; }
#endif

#ifndef MFEM_USE_LAPACK
   if (print_warning)
   {
      mfem::out
            << "\n" << string(80, '=')
            << "\nMFEM is compiled without LAPACK."
            << "\nUsing precomputed values for PartialFractionApproximation."
            << "\nOnly alpha = 0.33, 0.5, and 0.99 are available."
            << "\nThe default is alpha = 0.5.\n" << string(80, '=') << "\n"
            << endl;
   }
   const real_t eps = std::numeric_limits<real_t>::epsilon();

   if (abs(alpha - 0.33) < eps)
   {
      coeffs = Array<real_t> ({1.821898e+03, 9.101221e+01, 2.650611e+01,
                               1.174937e+01, 6.140444e+00, 3.441713e+00,
                               1.985735e+00, 1.162634e+00, 6.891560e-01,
                               4.111574e-01, 2.298736e-01});
      poles = Array<real_t> ({-4.155583e+04, -2.956285e+03, -8.331715e+02,
                              -3.139332e+02, -1.303448e+02, -5.563385e+01,
                              -2.356255e+01, -9.595516e+00, -3.552160e+00,
                              -1.032136e+00, -1.241480e-01});
   }
   else if (abs(alpha - 0.99) < eps)
   {
      coeffs = Array<real_t>({2.919591e-02, 1.419750e-02, 1.065798e-02,
                              9.395094e-03, 8.915329e-03, 8.822991e-03,
                              9.058247e-03, 9.814521e-03, 1.180396e-02,
                              1.834554e-02, 9.840482e-01});
      poles = Array<real_t> ({-1.069683e+04, -1.769370e+03, -5.718374e+02,
                              -2.242095e+02, -9.419132e+01, -4.031012e+01,
                              -1.701525e+01, -6.810088e+00, -2.382810e+00,
                              -5.700059e-01, -1.384324e-03});
   }
   else
   {
      if (abs(alpha - 0.5) > eps)
      {
         alpha = 0.5;
      }
      coeffs = Array<real_t>({2.290262e+02, 2.641819e+01, 1.005566e+01,
                              5.390411e+00, 3.340725e+00, 2.211205e+00,
                              1.508883e+00, 1.049474e+00, 7.462709e-01,
                              5.482686e-01, 4.232510e-01, 3.578967e-01});
      poles = Array<real_t>({-3.168211e+04, -3.236077e+03, -9.868287e+02,
                             -3.945597e+02, -1.738889e+02, -7.925178e+01,
                             -3.624992e+01, -1.629196e+01, -6.982956e+00,
                             -2.679984e+00, -7.782607e-01, -7.649166e-02});
   }

   if (print_warning)
   {
      mfem::out << "=> Using precomputed values for alpha = "
                << alpha << "\n" << std::endl;
   }


   return;
#else
   MFEM_CONTRACT_VAR(print_warning);
#endif

   Vector x(npoints);
   Vector val(npoints);
   real_t dx = lmax / (real_t)(npoints-1);
   for (int i = 0; i<npoints; i++)
   {
      x(i) = dx * (real_t)i;
      val(i) = pow(x(i),1.-alpha);
   }

   // Apply triple-A algorithm to f(x) = x^{1-a}
   Array<real_t> z, f;
   Vector w;
   RationalApproximation_AAA(val,x,z,f,w,tol,max_order);

   Vector vecz, vecf;
   vecz.SetDataAndSize(z.GetData(), z.Size());
   vecf.SetDataAndSize(f.GetData(), f.Size());

   // Compute poles and zeros for RA of f(x) = x^{1-a}
   real_t scale;
   Array<real_t> zeros;
   ComputePolesAndZeros(vecz, vecf, w, poles, zeros, scale);

   // Remove the zero at x=0, thus, delivering a RA for f(x) = x^{-a}
   zeros.DeleteFirst(0.0);

   // Compute partial fraction approximation of f(x) = x^{-a}
   PartialFractionExpansion(scale, poles, zeros, coeffs);
}
